# -*- coding: utf-8 -*-

import multiprocessing
import asyncio
import os
import pwd
import subprocess
import subprocess
from ctypes import *

import Proxy.proxy as Proxy
import Detect.detecting as detect

# 로깅 설정
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

async def set_proxy(rootPID, PID, isDefectDetected,
                    POD_NAME, POD_IP, REPLICA_INFO,
                    pause_event):
    TARGET_PORT = os.environ.get('TARGET_PORT', '3000')
    PROXY_PORT = os.environ.get('PROXY_PORT', '8080')
    MASTER_NODE_IP = '10.125.37.77'

    proxy = Proxy.Proxy(rootPID, PID, isDefectDetected, 
                        POD_NAME, POD_IP, REPLICA_INFO,
                        pause_event, TARGET_PORT, PROXY_PORT, MASTER_NODE_IP)
    await proxy.start_server()

def start_proxy(rootPID, PID, isDefectDetected,
                POD_NAME, POD_IP, REPLICA_INFO,
                pause_event):
    
    asyncio.run(set_proxy(rootPID, PID, isDefectDetected,
                          POD_NAME, POD_IP, REPLICA_INFO,
                          pause_event))

class UserProcess(multiprocessing.Process):
    def __init__(self, user_name, *args, **kwargs):
        self.user_name = user_name
        super().__init__(*args, **kwargs)

    def run(self):
        try:
            # 사용자 ID 가져오기
            pw_record = pwd.getpwnam(self.user_name)
            user_uid = pw_record.pw_uid
            user_gid = pw_record.pw_gid

            # 그룹 ID 설정
            os.setgid(user_gid)

            # 사용자 ID 설정
            os.setuid(user_uid)
        except PermissionError:
            print(f"Warning: Unable to set UID/GID to {self.user_name}. Make sure the script is run with root privileges.")
        except KeyError:
            print(f"Warning: User {self.user_name} not found. Make sure the user exists.")

        super().run()

def create_shared_resources(user_name):
    try:
        ip = subprocess.check_output("hostname -I", shell=True).decode().split()[0]
        print(f"IP: {ip}")
    except Exception as e:
        print(f"Error getting IP: {e}")
        ip = ''
    try:
        pw_record = pwd.getpwnam(user_name)
        user_uid = pw_record.pw_uid
        user_gid = pw_record.pw_gid

        # Temporarily change to proxyuser
        os.setegid(user_gid)
        os.seteuid(user_uid)

        # Create shared resources
        manager = multiprocessing.Manager()
        rootPID = manager.Value('i', 0)
        PID = manager.list()
        isDefectDetected = manager.Value('b', False)
        POD_NAME = manager.Value('s', '')
        POD_IP = manager.Value('s', ip)
        REPLICA_INFO = manager.list()
        pause_event = multiprocessing.Event()

        # Change back to root
        os.seteuid(0)
        os.setegid(0)

        return rootPID, PID, isDefectDetected, POD_NAME, POD_IP, REPLICA_INFO, pause_event
    except Exception as e:
        print(f"Error creating shared resources: {e}")
        return None, None, None, None, None, None

##################################### Main ###################################################

if __name__ == '__main__':
    if os.geteuid() != 0:
        print("This script must be run as root.")
        exit(1)
    
    if not os.path.exists("./model/data.txt"):
        open("./model/data.txt", 'w').close()
        os.chmod("./model/data.txt", 0o777)

    user_name = "proxyuser"

    rootPID, PID, isDefectDetected, POD_NAME, POD_IP, REPLICA_INFO, pause_event = create_shared_resources(user_name)

    pause_event.set()

    web_server_process = UserProcess(user_name, target=start_proxy,
                                     args=(rootPID, PID, isDefectDetected, 
                                           POD_NAME, POD_IP, REPLICA_INFO,
                                           pause_event))

    web_server_process.start()

    try:
        web_server_process.join()
    except KeyboardInterrupt:
        print("Terminating processes...")
        web_server_process.terminate()