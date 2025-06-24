import asyncio
import socket
import struct
import ctypes
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import joblib
from collections import defaultdict

IDLE_TIMEOUT = 1
TIMEOUT = 1
FEAT_DIM = 128

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIB_PATH = os.path.abspath("./v1/packet_parser.so")
c_parser = ctypes.CDLL(LIB_PATH)
c_parser.parse_packet.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)
]
c_parser.parse_packet.restype = ctypes.c_int

class StudentEncoder(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),         # 첫 conv: 4채널
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(),         # 두 번째 conv: 8채널
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(8 * 2 * 2, 32), nn.ReLU(),              # FC는 축소
            nn.BatchNorm1d(32), nn.Dropout(0.3),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.net(x)

DEVICE = torch.device("cpu")

class Proxy:
    def __init__(self, rootPID, PID, isDefectDetected,
                POD_NAME, POD_IP, REPLICA_INFO,
                pause_event, TARGET_PORT, PROXY_PORT):
        self.rootPID = rootPID
        self.PID = PID
        self.isDefectDetected = isDefectDetected
        self.POD_NAME = POD_NAME
        self.POD_IP = POD_IP
        self.REPLICA_INFO = REPLICA_INFO
        self.pause_event = pause_event
        self.PROXY_PORT = PROXY_PORT
        self.TARGET_PORT = TARGET_PORT
        self.payload_len = 1460
        self.window_size = 15
        self.vector_len = self.payload_len + 24
        self.session_buffers = defaultdict(lambda: torch.zeros((self.window_size, self.vector_len), dtype=torch.uint8))
        self.buffer_indices = defaultdict(int)

        self.encoder = StudentEncoder().to(DEVICE)
        self.encoder.load_state_dict(torch.load('./Model/student_encoder_kd_k8s_10_2x8.pth', map_location=DEVICE))
        self.encoder.eval()
        self.ocsvm = joblib.load('./Model/ocsvm.pkl')

    def parse_packet(self, raw_bytes):
        raw_buf = (ctypes.c_uint8 * len(raw_bytes)).from_buffer_copy(raw_bytes)
        out_buf = (ctypes.c_uint8 * self.vector_len)()
        result = c_parser.parse_packet(raw_buf, len(raw_bytes), out_buf)
        if result == 0:
            return np.ctypeslib.as_array(out_buf)
        return None

    async def get_target(self, addr, client_writer):
        src_ip, src_port = addr
        try:
            SO_ORIGINAL_DST = 80
            sock = client_writer.get_extra_info('socket')
            dst = sock.getsockopt(socket.SOL_IP, SO_ORIGINAL_DST, 16)
            dst_port, dst_ip = struct.unpack("!2xH4s8x", dst)
            dst_ip = socket.inet_ntoa(dst_ip)

            if dst_port in [22, 23]:
                return dst_ip, dst_port, 'interactive'

            if src_ip == self.POD_IP.value:
                return dst_ip, dst_port, 'internal'

            return dst_ip, self.TARGET_PORT, 'external'
        except Exception as e:
            logger.error(f"Get original destination error: {e}")
            raise e

        
    async def handle_client(self, client_reader, client_writer):
        addr = client_writer.get_extra_info('peername')
        logger.info(f"Connection from {addr}")
        transfer_tasks = []
        remote_writer = None

        try:
            target_ip, target_port, origin = await self.get_target(addr, client_writer)
            if target_ip == self.POD_IP.value:
                target_ip = "127.0.0.1"

            remote_reader, remote_writer = await asyncio.open_connection(target_ip, target_port)

            transfer_tasks = [
                asyncio.create_task(self.transfer(client_reader, remote_writer)),
                asyncio.create_task(self.transfer(remote_reader, client_writer))
            ]

            done, pending = await asyncio.wait(transfer_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=TIMEOUT)
            for task in pending:
                task.cancel()
        finally:
            for task in transfer_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=IDLE_TIMEOUT)
                    except:
                        pass
            if remote_writer:
                await self.close_connection(remote_writer)
            if client_writer:
                await self.close_connection(client_writer)


    async def transfer(self, reader, writer):
        try:
            while not reader.at_eof():
                data = await reader.read(16384)
                if not data:
                    break
                # vec = self.parse_packet(data)

                # if vec is not None:
                #     logger.debug(f"Parsed vector: shape={vec.shape}")

                writer.write(data)
                await writer.drain()
        except Exception as e:
            logger.error(f"Error transferring data: {e}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Error transferring data: {e}")
                

    async def close_connection(self, writer):
        if writer and not writer.is_closing():
            try:
                writer.write_eof()
            except:
                pass
            try:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
            except Exception as e:
                logger.error(f"Error in close_connection: {e}")

    def predict(self, stacked_tensor):
        flat = stacked_tensor.flatten()
        if flat.shape[0] < 34 * 44:
            flat = torch.cat([flat, torch.zeros(34 * 44 - flat.shape[0])])
        img = flat[:34 * 44].reshape(1, 1, 34, 44).to(DEVICE)

        with torch.no_grad():
            feat = self.encoder(img).cpu().numpy()
            score = self.ocsvm.decision_function(feat)[0]
            pred = int(score < 0)
            logger.info(f"{'🚨 Anomaly' if pred else '✅ Normal'}: {score:.4f}")
            return pred

import uvloop

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--target-port', type=int, default=80)
    parser.add_argument('--proxy-port', type=int, default=8080)
    parser.add_argument('--pod-name', type=str, default='test')
    parser.add_argument('--pod-ip', type=str, default='172.16.200.75')
    args = parser.parse_args()

    pause_event = asyncio.Event()

    proxy_instance = Proxy(
        rootPID=None,
        PID=os.getpid(),
        isDefectDetected=None,
        POD_NAME=args.pod_name,
        POD_IP=type("obj", (object,), {"value": args.pod_ip}),
        REPLICA_INFO=None,
        pause_event=pause_event,
        TARGET_PORT=args.target_port,
        PROXY_PORT=args.proxy_port,
    )

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    async def main():
        server = await asyncio.start_server(
            proxy_instance.handle_client, host='0.0.0.0', port=args.proxy_port
        )
        logger.info(f"Proxy server listening on port {args.proxy_port}")
        async with server:
            await server.serve_forever()

    asyncio.run(main())
