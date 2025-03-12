# -*- coding: utf-8 -*-

import asyncio
import aiohttp
from aiohttp import web
import json
import hashlib
import aiofiles
from yarl import URL
import socket
import struct
from ctypes import *
import logging
import base64
import re
import numpy as np
import time

import os
import sys

from Model import model
from Model import preprocess
from Detect import detecting

BUFFER_SIZE = 65536  # 64KB buffer
FLUSH_INTERVAL = 0.01  # 0.5 seconds
IDLE_TIMEOUT = 1  # 1 second
TIMEOUT = 1
PROXY_API_PORT = 9011
CHECK_GET_IPS = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Proxy:
    def __init__(self, rootPID, PID, isDefectDetected,
                POD_NAME, POD_IP, REPLICA_INFO, 
                pause_event, TARGET_PORT, PROXY_PORT, MASTER_NODE_IP,):
        self.rootPID = rootPID
        self.PID = PID
        self.isDefectDetected = isDefectDetected
        self.POD_NAME = POD_NAME
        self.POD_IP = POD_IP
        self.REPLICA_INFO = REPLICA_INFO
        self.pause_event = pause_event
        self.PROXY_PORT = PROXY_PORT
        self.TARGET_PORT = TARGET_PORT
        self.MASTER_NODE_IP = MASTER_NODE_IP
        self.app = web.Application(logger=None, client_max_size=0)

        self.clf_model = detecting.get_model()
    
    def setup_routes(self):
        self.app.router.add_post('/get/response', self.handle_response)
        self.app.router.add_post('/receive/model', self.handle_model)
        self.app.router.add_post('/receive/pods_ip', self.handle_pods_ip)
        # self.app.router.add_get('/flush', self.handle_flush)

    async def start_server(self):
        self.setup_routes()
        runner = web.AppRunner(self.app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', PROXY_API_PORT)
        await site.start()
        print(f'HTTP API server running on port {PROXY_API_PORT}')

        server = await asyncio.start_server(
            self.handle_client, '0.0.0.0', self.PROXY_PORT)
        addr = server.sockets[0].getsockname()
        print(f'Serving on {addr}')
        
        async with server:
            try:
                await server.serve_forever()
            finally:
                await runner.cleanup()

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

                    packet_bytes = preprocess.packet_to_bytes(data)
                    image = preprocess.packet_to_image(packet_bytes, 32)

                    predicted = self.clf_model.classify(image) # 0 normal, 1 attack

                    if predicted == 1:
                        print('\033[91m' + "Attack Detected" + '\033[0m')
                        self.isDefectDetected.value = True
                        # connection close
                        await self.close_connection(writer)
                        await self.close_connection(client_writer)
                        await preprocess.image_save(image, 'model/attack_traffic_image/')
                        break
                    else: 
                        writer.write(data)
                        await writer.drain()
                        await preprocess.image_save(image, 'model/normal_traffic_image/')
                
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


    async def validate_response(self, request_data):
        try:
            # 원래 remote 서버로 요청을 보내고 응답을 받습니다
            original_response = await self.get_remote_response(request_data)
            
            # 복제 서버(relay)로 같은 요청을 보내고 응답을 받습니다
            relay_response = await self.get_relay_response(request_data)

            if original_response is None or relay_response is None:
                print('\033[91m' + "Failed to get response from either original or relay server" + '\033[0m')
                if original_response is None:
                    print("Original Response is None")
                if relay_response is None:
                    print("Relay Response is None")
                return False, None

            if self.is_http_message(request_data):
                if self.get_http_status_code(original_response) == 502:
                    print("Original Response is 502")
                    original_response = await self.get_remote_response(request_data)
                if self.get_http_status_code(relay_response) == 502:
                    print("Relay Response is 502")
                    relay_response = await self.get_relay_response(request_data)
                
                original_response = await self.normalize_http_response(original_response)
                relay_response = await self.normalize_http_response(relay_response)
                print(f"Original Response: {original_response}")
                print(f"Relay Response: {relay_response}")

            if original_response is None or relay_response is None:
                print('\033[31m' + "Failed to get response from either original or relay server" + '\033[0m')
                # self.isDefectDetected.value = True
                return False, None
            elif original_response != relay_response:
                print('\033[91m' + "Respond with Relay Response" + '\033[0m')
                self.isDefectDetected.value = True
                return False, relay_response
            else:
                print('\033[92m' + "Respond with Original Response" + '\033[0m')
                return True, original_response

        except Exception as e:
            print(f"Error in validate_response: {e}")
            self.isDefectDetected.value = True
            return False, None

    async def get_remote_response(self, request_data):
        try:
            reader, writer = await asyncio.open_connection('0.0.0.0', self.TARGET_PORT)
            if isinstance(request_data, bytes):
                writer.write(request_data)
            else:
                writer.write(request_data.encode('utf-8'))
            await writer.drain()
            response = await reader.read(16384)  # 적절한 버퍼 크기를 사용하세요
            writer.close()
            await writer.wait_closed()
            return response
        except Exception as e:
            print(f"Error getting remote response: {e}")
            return None
        except Exception as e:
            print(f"Error getting remote response: {e}")
            return None
    OU
    async def get_relay_response(self, data):
        target_pod_ip = next((replica['ip'] for replica in self.REPLICA_INFO), '')
        if not target_pod_ip:
            return None
        print('\033[94m' + f"Relay request to {target_pod_ip}" + '\033[0m')
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=f"http://{target_pod_ip}:{PROXY_API_PORT}/get/response",
                    json={'request': data.decode('utf-8')}
                ) as response:
                    if response.status == 200:
                        try:
                            json_response = await response.json()
                            return json_response['response'].encode('utf-8')
                        except aiohttp.ContentTypeError:
                            return None
                    else:
                        return None
        except aiohttp.ClientError as e:
            print(f"Error in get_relay_response: {e}")
            return None
        except asyncio.TimeoutError as e:
            print(f"Timeout error in get_relay_response: {e}")
            return None
        except Exception as e:
            print(f"Error in get_relay_response: {e}")
            return None
        
    def is_http_message(self, data):
        try:
            # 바이트 데이터를 문자열로 변환
            message = data.decode('utf-8')
            
            # HTTP 요청 확인
            request_pattern = r'^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH|TRACE) .+ HTTP/\d\.\d\r\n'
            
            # HTTP 응답 확인
            response_pattern = r'^HTTP/\d\.\d \d{3} .+\r\n'
            
            # 정규식 매칭
            if (re.match(request_pattern, message) or 
                re.match(response_pattern, message)):
                return True
            return False
        except (UnicodeDecodeError, AttributeError):
            return False
    
    def get_http_status_code(self, response_data):
        try:
            first_line = response_data.decode('utf-8').split('\r\n')[0]
            if first_line.startswith('HTTP/'):
                return int(first_line.split(' ')[1])
        except (UnicodeError, IndexError, ValueError):
            pass
        return None

    async def normalize_http_response(self, response_data):
        """HTTP 응답에서 동적인 부분을 제거하고 정규화"""
        try:
            # HTTP 메시지 정규화 로직
            response_str = response_data.decode('utf-8')
            
            if '\r\n\r\n' in response_str:
                headers, body = response_str.split('\r\n\r\n', 1)
                header_lines = headers.split('\r\n')
                
                normalized_headers = []
                for line in header_lines:
                    if not any(h in line.lower() for h in [
                        'date:', 
                        'last-modified:', 
                        'expires:', 
                        'etag:',
                        'set-cookie:',
                        'server:',
                        'cf-ray:',
                        'x-request-id:',
                        'x-runtime:',
                        'x-served-by:',
                    ]):
                        normalized_headers.append(line)
                
                normalized_response = '\r\n'.join(normalized_headers) + '\r\n\r\n' + body
                return normalized_response.encode('utf-8')
            
            return response_data
        except Exception as e:
            print(f"Error normalizing HTTP response: {e}")
            return response_data
        
    async def validate_request(self, data, target_ip, target_port):
        try:
            signature_data = await get_request_signature(data, target_ip, target_port)
            headers = {
                'Content-Type': 'application/json',
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=f"http://{self.MASTER_NODE_IP}:8080/send/internal_request_body",
                    json={
                        'pod_ip': self.POD_IP.value,
                        'signature_data': signature_data,
                    },
                    headers=headers
                ) as response:
                    if response.status == 200:
                        try:
                            json_response = await response.json()
                            print(f"result : {json_response['result']}")
                            if 'result' in json_response:
                                if json_response['result'] == "valid":
                                    print('\033[92m' + "Allow Internal Request" + '\033[0m')
                                    return True
                                else:
                                    print('\033[91m' + "Drop Internal Request" + '\033[0m')
                                    return False
                            else:
                                # # #logger.error(f"Unexpected response format: {json_response}")
                                return False
                        except aiohttp.ContentTypeError:
                            return False
                    else:
                        return False
        except aiohttp.ClientError as e:
            return False
        except Exception as e:
            return False
        
    async def handle_response(self, request):
        try:
            data = await request.json()
            request  = data['request']

            response = await self.get_remote_response(request)

            if isinstance(response, bytes):
                response = response.decode('utf-8')

            return web.Response(
                text=json.dumps({'response': response}),
                content_type='application/json'
            )
        except Exception as e:
            print(f"Error in handle_response: {e}")
            return web.Response(text=str(e), status=500)
        
    async def handle_model(self, request):
        try:
            self.pause_event.clear()
            # sleep 5second
            await asyncio.sleep(5)
            data = await request.json()
            model_base64 = data['model']
            vectorizer_base64 = data['vectorizer']

            model_data = base64.b64decode(model_base64)
            vectorizer_data = base64.b64decode(vectorizer_base64)

            async with aiofiles.open('./Detect/model.pkl', 'wb') as model_file:
                await model_file.write(model_data)

            async with aiofiles.open('./Detect/vectorizer.pkl', 'wb') as vectorizer_file:
                await vectorizer_file.write(vectorizer_data)

            print('\033[96m' + "Model and vectorizer have been changed" + '\033[0m')        
            self.pause_event.set() 
            
            return web.Response(text='Model and vectorizer have been saved successfully', status=200)
        except Exception as e:
            self.pause_event.set()
            return web.Response(text=str(e), status=500)

    async def handle_pods_ip(self, request):
        try:
            global CHECK_GET_IPS
            if CHECK_GET_IPS:
                print("\033[94m" + "Pods IP data received" + "\033[0m")
                CHECK_GET_IPS = False

            data = await request.json()
            if 'pods_ip' not in data:
                return web.Response(text='Invalid data format. "pods_ip" key is required.', status=400)

            self.POD_NAME.value = data['name']
            self.POD_IP.value = data['ip']
            self.REPLICA_INFO[:] = data['pods_ip']

            return web.Response(
                text=json.dumps({'status': 'success', 'message': 'Pods IP data received and stored.'}),
                content_type='application/json'
            )   
        except Exception as e:
            return web.Response(text=str(e), status=500)

async def get_forwarding_url(request, TARGET_URL):
    target_path = request.path_qs
    if "relay" in target_path:
        target_path = target_path.split("relay")[-1]

    full_url = f"{TARGET_URL}{target_path}"
    # #logger.info(f"Forwarding request to {full_url}")

    return full_url


async def get_external_ip(request):
    try:
        # request.url은 yarl.URL 객체입니다
        original_url = request.url

        # 원래 요청의 스킴(http 또는 https)을 유지합니다
        scheme = original_url.scheme

        # host와 path_qs를 결합하여 새 URL을 만듭니다
        new_url = URL.build(scheme=scheme, host=request.host, path=request.path_qs)

        print(f"External IP: {new_url}")
        return str(new_url)
    except Exception as e:
        print(e)

    return None

async def get_request_signature(data, target_ip, target_port):
    # 서로 다른 Dst 해쉬 구분 위해
    dst_ip = target_ip + ':' + str(target_port)

    signature_data = data + dst_ip.encode()
    
    return hashlib.md5(signature_data).hexdigest()
