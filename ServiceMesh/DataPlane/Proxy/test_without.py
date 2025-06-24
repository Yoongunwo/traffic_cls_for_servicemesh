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
from scapy.all import Ether, IP, TCP

import os
import sys

BUFFER_SIZE = 65536  # 64KB buffer
FLUSH_INTERVAL = 0.01  # 0.5 seconds
IDLE_TIMEOUT = 1  # 1 second
TIMEOUT = 1
CHECK_GET_IPS = True

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
                pass

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

                    # processed_data = self.preprocess(data)
                    
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

    def extract_packet_vector(self, raw_data):
        try:
            from scapy.all import Ether, IP, TCP
            pkt = Ether(raw_data)
            f = []

            if IP in pkt:
                ip = pkt[IP]
                f.extend([
                    ip.ttl,
                    ip.proto,
                    ip.flags.value,
                    (ip.frag >> 8) & 0xFF,
                    ip.frag & 0xFF
                ])

            if TCP in pkt:
                tcp = pkt[TCP]
                f.extend([
                    tcp.dataofs,
                    int(tcp.flags),
                    (tcp.window >> 8) & 0xFF,
                    tcp.window & 0xFF,
                    (tcp.urgptr >> 8) & 0xFF,
                    tcp.urgptr & 0xFF
                ])
                f.extend(tcp.seq.to_bytes(4, 'big'))
                f.extend(tcp.ack.to_bytes(4, 'big'))

                payload = bytes(tcp.payload)[:self.payload_len]
                if len(payload) < self.payload_len:
                    payload += b'\x00' * (self.payload_len - len(payload))
                f.extend(payload)

            return np.array(f, dtype=np.uint8)
        except Exception as e:
            logger.warning(f"Failed to extract packet vector: {e}")
            return None

    def preprocess(self, data):
        try:
            pkt = Ether(data)
            if not (IP in pkt and TCP in pkt):
                return None

            key = (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, pkt[IP].proto)

            if key not in self.session_table:
                self.session_table[key] = deque(maxlen=self.window_size)

            vector = self.extract_packet_vector(data)
            if vector is None:
                return None

            self.session_table[key].append(vector)

            if len(self.session_table[key]) < self.window_size:
                return None  # 아직 window가 준비 안 됨

            return np.stack(self.session_table[key], axis=0)
        except Exception as e:
            logger.warning(f"Preprocess failed: {e}")
            return None

    def predict(self, data):
        with torch.no_grad():
            data = np.nan_to_num(data)
            data = np.clip(data, 0, 255).astype(np.float32) / 255.0
            flat = data.flatten()

            if flat.shape[0] < 34 * 44:
                flat = np.pad(flat, (0, 34 * 44 - flat.shape[0]))
            img = flat[:34 * 44].reshape(1, 34, 44)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            feat = self.encoder(img).detach().cpu().numpy()
            score = self.ocsvm.decision_function(feat)[0]
            pred = 1 if score < 0 else 0
            if pred == 1:
                logger.info(f"🚨 Anomaly detected: {score:.4f}")
                return 1
            else:
                logger.info(f"✅ Normal traffic: {score:.4f}")
                return 0

import uvloop

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--target-port', type=int, default=80)
    parser.add_argument('--proxy-port', type=int, default=8080)
    parser.add_argument('--pod-name', type=str, default='test')
    parser.add_argument('--pod-ip', type=str, default='172.16.200.73')
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