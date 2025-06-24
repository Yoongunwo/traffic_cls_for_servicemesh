import asyncio
import socket
import struct
import ctypes
import logging
import os
import numpy as np
import torch
import joblib
import argparse
import uvloop

# === Config ===
IDLE_TIMEOUT = 1
TIMEOUT = 1
FEAT_DIM = 128
MAX_SESSIONS = 65536
VEC_LEN = 1479
WIN_SIZE = 5
H, W = 1479, 5  # input shape

# === Logging 설정 최소화 ===
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# === TorchScript Model ===
model = torch.jit.load('./Model/student_encoder_1x8_ts.pt').eval().to(torch.float32)
ocsvm = joblib.load('./Model/ocsvm.pkl')

@torch.jit.script
def preprocess(flat: torch.Tensor) -> torch.Tensor:
    return flat.reshape(1, 1, 1479, 5)

# === C parser setup ===
c_parser = ctypes.CDLL('./Proxy/v2_real/packet_parser_stack_v2.so')
c_parser.parse_and_stack.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float), ctypes.c_uint32
]
c_parser.parse_and_stack.restype = ctypes.c_int
c_parser.init_session_storage.restype = ctypes.c_int
assert c_parser.init_session_storage() == 0

# === Proxy Class ===
class Proxy:
    def __init__(self, TARGET_PORT, PROXY_PORT, POD_IP):
        self.TARGET_PORT = TARGET_PORT
        self.PROXY_PORT = PROXY_PORT
        self.POD_IP = POD_IP

    async def get_target(self, addr, client_writer):
        src_ip, _ = addr
        try:
            SO_ORIGINAL_DST = 80
            sock = client_writer.get_extra_info('socket')
            dst = sock.getsockopt(socket.SOL_IP, SO_ORIGINAL_DST, 16)
            dst_port, dst_ip = struct.unpack("!2xH4s8x", dst)
            dst_ip = socket.inet_ntoa(dst_ip)
            if dst_port in [22, 23]:
                return dst_ip, dst_port, 'interactive'
            if src_ip == self.POD_IP:
                return dst_ip, dst_port, 'internal'
            return dst_ip, self.TARGET_PORT, 'external'
        except Exception as e:
            logger.error(f"Get original destination error: {e}")
            raise

    async def handle_client(self, client_reader, client_writer):
        addr = client_writer.get_extra_info('peername')
        remote_writer = None
        try:
            target_ip, target_port, _ = await self.get_target(addr, client_writer)
            if target_ip == self.POD_IP:
                target_ip = "127.0.0.1"
            remote_reader, remote_writer = await asyncio.open_connection(target_ip, target_port)
            tasks = [
                asyncio.create_task(self.transfer(client_reader, remote_writer)),
                asyncio.create_task(self.transfer(remote_reader, client_writer))
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=TIMEOUT)
            for task in pending:
                task.cancel()
        finally:
            if remote_writer:
                await self.close_connection(remote_writer)
            if client_writer:
                await self.close_connection(client_writer)

    async def transfer(self, reader, writer):
        while not reader.at_eof():
            data = await reader.read(16384)
            if not data:
                break
            try:
                src_ip = int.from_bytes(data[26:30], 'big')
                dst_ip = int.from_bytes(data[30:34], 'big')
                src_port = struct.unpack("!H", data[34:36])[0]
                dst_port = struct.unpack("!H", data[36:38])[0]
                proto = data[23]
                session_id = (src_ip ^ dst_ip ^ src_port ^ dst_port ^ proto) % MAX_SESSIONS

                out_stack = (ctypes.c_float * (WIN_SIZE * VEC_LEN))()
                raw_buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
                ret = c_parser.parse_and_stack(raw_buf, len(data), out_stack, session_id)

                if ret == 1:
                    stacked = np.ctypeslib.as_array(out_stack, shape=(H * W,))
                    stacked_tensor = torch.from_numpy(stacked).to(dtype=torch.float32).div(255.0)
                    img = preprocess(stacked_tensor).contiguous().unsqueeze(0)

                    with torch.no_grad():
                        feat = model(img).cpu().numpy()
                        score = ocsvm.decision_function(feat)[0]
                    # with torch.no_grad():
                    #     feat = model(img).cpu().numpy()
                    #     score = (
                    #         np.dot(feat, ocsvm.coef_.T)[0] + ocsvm.intercept_[0]
                    #         if USE_LINEAR_KERNEL else ocsvm.decision_function(feat)[0]
                    #     )
                        # pred = int(score < 0)  # 사용할 경우
                        # logging suppressed for performance
            except Exception:
                pass  # 오류 무시하고 다음 요청 처리
            writer.write(data)
            await writer.drain()

    async def close_connection(self, writer):
        if writer and not writer.is_closing():
            try:
                writer.write_eof()
            except:
                pass
            try:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
            except:
                pass

# === Main ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-port', type=int, default=80)
    parser.add_argument('--proxy-port', type=int, default=8080)
    parser.add_argument('--pod-ip', type=str, default='172.16.200.107')
    args = parser.parse_args()

    proxy = Proxy(
        TARGET_PORT=args.target_port,
        PROXY_PORT=args.proxy_port,
        POD_IP=args.pod_ip
    )

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    async def main():
        server = await asyncio.start_server(proxy.handle_client, host='0.0.0.0', port=args.proxy_port)
        logger.info(f"Proxy server listening on port {args.proxy_port}")
        async with server:
            await server.serve_forever()

    asyncio.run(main())
