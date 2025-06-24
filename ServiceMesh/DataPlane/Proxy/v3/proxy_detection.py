import asyncio
import socket
import struct
import ctypes
import logging
import os
import numpy as np
import torch
from multiprocessing import Process, Queue

# === Config ===
IDLE_TIMEOUT = 1
TIMEOUT = 1
FEAT_DIM = 128
MAX_SESSIONS = 65536
VEC_LEN = 1479
WIN_SIZE = 15
LIB_PATH = os.path.abspath("./Proxy/v3/packet_parser_stack.so")
NUM_WORKERS = 4
MODEL_PATH = './Model/student_encoder_ts.pt'

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === C parser setup ===
c_parser = ctypes.CDLL(LIB_PATH)
c_parser.parse_and_stack.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float), ctypes.c_uint32
]
c_parser.parse_and_stack.restype = ctypes.c_int

# === TorchScript Inference Worker ===
def inference_worker(queue):
    torch.set_num_threads(1)
    model = torch.jit.load(MODEL_PATH, map_location='cpu')
    model.eval()

    while True:
        buf = queue.get()
        if buf is None:
            break
        try:
            stacked = np.frombuffer(buf, dtype=np.float32).reshape(WIN_SIZE, VEC_LEN) / 255.0
            flat = stacked.flatten()
            padded = np.pad(flat, (0, max(0, 34 * 44 - flat.shape[0])), constant_values=0)
            img = torch.from_numpy(padded[:34 * 44].reshape(1, 1, 34, 44)).float()
            with torch.no_grad():
                _ = model(img)
        except Exception as e:
            logger.error(f"Inference error: {e}")

# === Proxy ===
class Proxy:
    def __init__(self, TARGET_PORT, PROXY_PORT, POD_IP, queue):
        self.TARGET_PORT = TARGET_PORT
        self.PROXY_PORT = PROXY_PORT
        self.POD_IP = POD_IP
        self.queue = queue

    async def get_target(self, addr, client_writer):
        src_ip, _ = addr
        try:
            SO_ORIGINAL_DST = 80
            sock = client_writer.get_extra_info('socket')
            dst = sock.getsockopt(socket.SOL_IP, SO_ORIGINAL_DST, 16)
            dst_port, dst_ip = struct.unpack("!2xH4s8x", dst)
            dst_ip = socket.inet_ntoa(dst_ip)
            if dst_port in [22, 23]: return dst_ip, dst_port, 'interactive'
            if src_ip == self.POD_IP: return dst_ip, dst_port, 'internal'
            return dst_ip, self.TARGET_PORT, 'external'
        except Exception as e:
            logger.error(f"Get target error: {e}")
            raise

    async def handle_client(self, client_reader, client_writer):
        addr = client_writer.get_extra_info('peername')
        transfer_tasks = []
        remote_writer = None
        try:
            target_ip, target_port, _ = await self.get_target(addr, client_writer)
            if target_ip == self.POD_IP:
                target_ip = "127.0.0.1"
            remote_reader, remote_writer = await asyncio.open_connection(target_ip, target_port)
            transfer_tasks = [
                asyncio.create_task(self.transfer(client_reader, remote_writer)),
                asyncio.create_task(self.transfer(remote_reader, client_writer))
            ]
            await asyncio.wait(transfer_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=TIMEOUT)
        finally:
            for task in transfer_tasks:
                if not task.done(): task.cancel()
            if remote_writer: await self.close_connection(remote_writer)
            if client_writer: await self.close_connection(client_writer)

    async def transfer(self, reader, writer):
        while not reader.at_eof():
            data = await reader.read(16384)
            if not data: break
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
                    arr_bytes = bytearray(out_stack)
                    self.queue.put(arr_bytes)
            except Exception as e:
                logger.warning(f"Parse failed: {e}")
            writer.write(data)
            await writer.drain()

    async def close_connection(self, writer):
        if writer and not writer.is_closing():
            try: writer.write_eof()
            except: pass
            try:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
            except Exception as e:
                logger.error(f"Close error: {e}")

# === Main ===
if __name__ == '__main__':
    import argparse
    import uvloop

    parser = argparse.ArgumentParser()
    parser.add_argument('--target-port', type=int, default=80)
    parser.add_argument('--proxy-port', type=int, default=8080)
    parser.add_argument('--pod-ip', type=str, default='172.16.184.244')
    args = parser.parse_args()

    queue = Queue()
    workers = [Process(target=inference_worker, args=(queue,)) for _ in range(NUM_WORKERS)]
    for p in workers: p.start()

    proxy = Proxy(
        TARGET_PORT=args.target_port,
        PROXY_PORT=args.proxy_port,
        POD_IP=args.pod_ip,
        queue=queue
    )

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    async def main():
        server = await asyncio.start_server(proxy.handle_client, host='0.0.0.0', port=args.proxy_port)
        logger.info(f"Proxy server listening on port {args.proxy_port}")
        async with server:
            await server.serve_forever()

    try:
        asyncio.run(main())
    finally:
        for _ in workers: queue.put(None)
        for p in workers: p.join()
