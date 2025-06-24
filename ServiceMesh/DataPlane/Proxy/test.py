# -*- coding: utf-8 -*-

import asyncio
import aiohttp
from aiohttp import web
import json
import hashlib
from yarl import URL
import socket
import struct
from ctypes import *
import logging
import base64
import re
import numpy as np
import time
import joblib
from collections import defaultdict

from scapy.all import Ether, IP, TCP

import os
import sys

IDLE_TIMEOUT = 1  # 1 second
TIMEOUT = 1
FEAT_DIM = 128

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.svm import OneClassSVM

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
        self.session_table = {}
        self.payload_len = 1460
        self.window_size = 15
        self.vector_len = self.payload_len + 24
        
        self.encoder = StudentEncoder().to(DEVICE)
        self.encoder.load_state_dict(torch.load('./Model/student_encoder_kd_k8s_10_2x8.pth', map_location=DEVICE))
        self.encoder.eval()

        self.ocsvm = joblib.load('./Model/ocsvm.pkl')
        self.session_buffers = defaultdict(lambda: torch.zeros((self.window_size, self.payload_len + 24), dtype=torch.uint8))
        self.buffer_indices = defaultdict(int)

    async def get_target(self, addr, client_writer):
        src_ip, src_port = addr
        try:
            SO_ORIGINAL_DST = 80
            sock = client_writer.get_extra_info('socket')
            dst = sock.getsockopt(socket.SOL_IP, SO_ORIGINAL_DST, 16)
            dst_port, dst_ip = struct.unpack("!2xH4s8x", dst)
            dst_ip = socket.inet_ntoa(dst_ip)
            
            # Interactive protocols ports
            interactive_ports = [22, 23]  # SSH and Telnet
            
            if dst_port in interactive_ports:
                return dst_ip, dst_port, 'interactive'
                
            if src_ip == self.POD_IP.value:
                return dst_ip, dst_port, 'internal'
            
            
            return dst_ip, self.TARGET_PORT, 'external'
            
        except Exception as e:
            print(f"Get original destination error: {e}")
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

            sock = remote_writer.get_extra_info('socket')
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)

            # 태스크 생성
            transfer_tasks = [
                asyncio.create_task(self.transfer(client_reader, remote_writer, origin, target_ip, target_port, client_writer, True)),
                asyncio.create_task(self.transfer(remote_reader, client_writer, origin, target_ip, target_port, client_writer, False))
            ]

            # 태스크 실행 및 대기
            try:
                done, pending = await asyncio.wait(
                    transfer_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=TIMEOUT
                )

                for task in pending:
                    task.cancel()

                # Handle completed tasks
                for task in done:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        pass

            except asyncio.TimeoutError:
                passa

        except asyncio.CancelledError:
            logger.debug("Client connection cancelled normally") 
        except Exception as e:
            print(f"Error in handle_client: {e}")
        finally:
            # 리소스 정리
            for task in transfer_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=IDLE_TIMEOUT)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    except Exception as e:
                        print(f"Error in task cleanup: {e}")

            # 연결 종료
            if remote_writer:
                try:
                    await self.close_connection(remote_writer)
                except Exception as e:
                    logger.error(f"Error closing remote connection: {e}")
            
            if client_writer:
                try:
                    await self.close_connection(client_writer)
                except Exception as e:
                    logger.error(f"Error closing client connection: {e}")


    async def transfer(self, reader, writer, origin, target_ip, target_port, client_writer, is_client_to_remote):
        try:
            while not reader.at_eof():
                try:
                    data = await reader.read(16384) # 18.82ms
                    # data = await asyncio.wait_for(reader.read(16384), timeout=TIMEOUT) # 25.25ms
                    if not data:
                        break

                    processed_data = self.preprocess(data)
                    
                    # if processed_data is not None:
                    #     result = self.predict(processed_data)

                    writer.write(data)
                    await writer.drain()
                
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"Error transferring data: {e}")
                    break

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Error transferring data: {e}")
                

    async def close_connection(self, writer):
        if writer and not writer.is_closing():
            try:
                writer.write_eof()
            except (OSError, AttributeError):
                pass

            try:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.error(f"Error in close_connection: {e}")
    
    def connect_to_remote(self, target_ip, target_port):
        self.remote = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.remote.settimeout(TIMET)
            self.remote.connect((target_ip, target_port))
            self.remote.settimeout(None)
        except ConnectionRefusedError as e:
            print(f"Connnection Refused: {e}")
            self.client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            self.client.close()
            return False
        except (socket.timeout, TimeoutError) as e:
            print(f"Connection Timeout: {target_ip}:{target_port}")
            self.client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            self.client.close()
            return False
        except Exception as e:
            print(f"Error: {e}")
            self.client.close()
            return False
        
        return True

    def fast_parse_packet(self, raw_data):
        try:
            # 최소 길이 검사 (이더넷 + IP + TCP 헤더)
            min_len = 14 + 20 + 20
            if len(raw_data) < min_len:
                return None

            # IP 헤더 필드 파싱
            ttl = raw_data[22]
            proto = raw_data[23]
            flags_frag = struct.unpack("!H", raw_data[20:22])[0]
            ip_flags = (flags_frag >> 13) & 0x7
            frag_offset = flags_frag & 0x1FFF

            # TCP 헤더 필드 파싱
            tcp_offset = 14 + 20
            data_offset = (raw_data[tcp_offset + 12] >> 4) & 0xF
            tcp_header_len = data_offset * 4

            if len(raw_data) < tcp_offset + tcp_header_len:
                return None

            flags = raw_data[tcp_offset + 13]
            window = struct.unpack("!H", raw_data[tcp_offset + 14:tcp_offset + 16])[0]
            urgptr = struct.unpack("!H", raw_data[tcp_offset + 18:tcp_offset + 20])[0]
            seq = raw_data[tcp_offset + 4:tcp_offset + 8]
            ack = raw_data[tcp_offset + 8:tcp_offset + 12]

            # Payload 추출 및 padding
            payload_start = tcp_offset + tcp_header_len
            payload = raw_data[payload_start:payload_start + self.payload_len]
            if len(payload) < self.payload_len:
                payload += b'\x00' * (self.payload_len - len(payload))

            # Feature vector 구성
            f = [
                ttl, proto, ip_flags,
                (frag_offset >> 8) & 0xFF, frag_offset & 0xFF,
                data_offset, flags,
                (window >> 8) & 0xFF, window & 0xFF,
                (urgptr >> 8) & 0xFF, urgptr & 0xFF
            ]
            f.extend(seq)
            f.extend(ack)
            f.extend(payload)

            vec = torch.tensor(f, dtype=torch.uint8)

            # 길이 검증 및 보정 (정확히 self.payload_len + 24여야 함)
            expected_len = self.payload_len + 24
            if vec.shape[0] != expected_len:
                if vec.shape[0] > expected_len:
                    vec = vec[:expected_len]
                else:
                    vec = F.pad(vec, (0, expected_len - vec.shape[0]))

            return vec

        except Exception as e:
            logger.warning(f"⚠️ Low-level parse failed: {e}")
            return None


    def preprocess(self, data):
        src_ip = socket.inet_ntoa(data[26:30])
        dst_ip = socket.inet_ntoa(data[30:34])
        src_port = struct.unpack("!H", data[34:36])[0]
        dst_port = struct.unpack("!H", data[36:38])[0]
        proto = data[23]

        key = (src_ip, dst_ip, src_port, dst_port, proto)
        vec = self.fast_parse_packet(data)
        if vec is None:
            return None

        idx = self.buffer_indices[key]
        self.session_buffers[key][idx] = vec
        self.buffer_indices[key] = (idx + 1) % self.window_size

        if (self.session_buffers[key] != 0).all():
            stacked = self.session_buffers[key].clone().float() /255.0
            return stacked
        return None

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

    pause_event = asyncio.Event()  # 기본 event

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