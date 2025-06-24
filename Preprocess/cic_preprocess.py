import pandas as pd
import os
from tqdm import tqdm
from scapy.all import RawPcapReader, Ether, IP, TCP, PcapReader
from PIL import Image
import numpy as np
from datetime import datetime

from hilbertcurve.hilbertcurve import HilbertCurve

def debug_timestamps(pcap_path):
    """PCAP 파일의 실제 타임스탬프 범위 확인 (UTC와 ADT 둘 다 표시)"""
    reader = PcapReader(pcap_path)
    timestamps = []
    count = 0
    
    for pkt in reader:
        if count < 10:  # 처음 10개만 확인
            ts = pkt.time
            dt_utc = datetime.fromtimestamp(ts)
            dt_adt = datetime.fromtimestamp(ts - 3*3600)  # UTC-3 = ADT
            timestamps.append((ts, dt_utc, dt_adt))
            print(f"Timestamp: {ts}")
            print(f"  UTC: {dt_utc}")
            print(f"  ADT: {dt_adt}")
            print()
            count += 1
        else:
            break
    
    reader.close()
    return timestamps

# === 1. 이미지 변환 방식 정의 ===
def row_wise_mapping(packet_bytes, width=32):
    normalized = np.array([int(b) for b in packet_bytes], dtype=np.uint8)
    if len(normalized) < width * width:
        padding = np.zeros(width * width - len(normalized), dtype=np.uint8)
        normalized = np.concatenate([normalized, padding])
    return normalized[:width * width].reshape(width, width)

def spiral_inward_mapping(byte_array, image_size=32):
    pad_len = max(0, image_size * image_size - len(byte_array))
    padded = np.pad(byte_array, (0, pad_len), 'constant')
    data = padded[:image_size * image_size]
    mat = np.zeros((image_size, image_size), dtype=np.uint8)
    top, bottom, left, right = 0, image_size-1, 0, image_size-1
    idx = 0
    while top <= bottom and left <= right:
        for i in range(left, right+1): mat[top][i] = data[idx]; idx += 1
        top += 1
        for i in range(top, bottom+1): mat[i][right] = data[idx]; idx += 1
        right -= 1
        if top <= bottom:
            for i in range(right, left-1, -1): mat[bottom][i] = data[idx]; idx += 1
            bottom -= 1
        if left <= right:
            for i in range(bottom, top-1, -1): mat[i][left] = data[idx]; idx += 1
            left += 1
    return mat

def diagonal_zigzag_mapping(byte_array, image_size=32):
    pad_len = max(0, image_size * image_size - len(byte_array))
    padded = np.pad(byte_array, (0, pad_len), 'constant')
    data = padded[:image_size * image_size]
    mat = np.zeros((image_size, image_size), dtype=np.uint8)
    index = 0
    for s in range(2 * image_size - 1):
        if s % 2 == 0:
            for i in range(s, -1, -1):
                if i < image_size and s - i < image_size:
                    mat[i][s - i] = data[index]; index += 1
        else:
            for i in range(0, s + 1):
                if i < image_size and s - i < image_size:
                    mat[i][s - i] = data[index]; index += 1
    return mat

def hilbert_mapping(byte_array, image_size=32):
    pad_len = max(0, image_size * image_size - len(byte_array))
    padded = np.pad(byte_array, (0, pad_len), 'constant')
    data = padded[:image_size * image_size]
    mat = np.zeros((image_size, image_size), dtype=np.uint8)
    p = int(np.log2(image_size))
    hilbert_curve = HilbertCurve(p, 2)
    for i in range(image_size * image_size):
        x, y = hilbert_curve.point_from_distance(i)
        mat[y][x] = data[i]
    return mat

# 선택할 매핑 방식
mapping_methods = {
    "row": row_wise_mapping,
    "spiral": spiral_inward_mapping,
    "zigzag": diagonal_zigzag_mapping,
    "hilbert": hilbert_mapping,
}

selected_mapping = "hilbert"  # 🔁 원하는 방식 선택 가능: row, spiral, zigzag, hilbert
image_size = 32

# === 2. 데이터 경로 설정 ===
csv_path = './Data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv'
pcap_path = './Data/cic_data/Wednesday-WorkingHours.pcap'
output_dir = f'./Data_CIC/Wed_{selected_mapping}_{image_size}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'benign'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'attack'), exist_ok=True)


# 공격 정의
# 날짜 및 공격자/피해자 IP
date_str = "2017-07-05"
attacker_ip = "205.174.165.73"
nat_ip = "205.174.165.80"  # 공격 대상 NAT IP

attack_windows = [
    # (공격이름, 시작시간, 종료시간)
    ("Slowloris",      "09:47", "10:10", "192.168.10.50"),
    ("Slowhttptest",   "10:14", "10:35", "192.168.10.50"),
    ("Hulk",           "10:43", "11:00", "192.168.10.50"),
    ("GoldenEye",      "11:10", "11:23", "192.168.10.50"),
    ("Heartbleed",     "15:12", "15:32", "192.168.10.51"),
]

import pytz

# def to_epoch(dt_str, date_str="2017-07-05"):
#     """ADT (UTC-3) 시간을 UTC epoch으로 변환"""
#     dt = datetime.strptime(f"{date_str} {dt_str}", "%Y-%m-%d %H:%M")
#     # 7월 5일은 일광절약시간 (ADT = UTC-3)
#     # 로컬 시간에 3시간을 더해서 UTC로 변환
#     utc_timestamp = int(dt.timestamp()) + (3 * 3600)
#     return utc_timestamp

# 또는 pytz 사용 (더 정확함):

# def to_epoch(dt_str, date_str="2017-07-05"):
#     """pytz를 사용한 정확한 변환"""
#     adt = pytz.timezone("America/Halifax")
#     adt = pytz.timezone("America/Halifax")
#     dt = datetime.strptime(f"{date_str} {dt_str}", "%Y-%m-%d %H:%M")
#     dt_local = adt.localize(dt)
#     return int(dt_local.timestamp())

def to_epoch(dt_str, date_str="2017-07-05"):
    dt = datetime.strptime(f"{date_str} {dt_str}", "%Y-%m-%d %H:%M")
    return int(dt.timestamp())  + 2 * 3600

print("=== PCAP 타임스탬프 디버깅 ===")
debug_timestamps(pcap_path)

attack_ranges = []
for name, start, end, dst in attack_windows:
    attack_ranges.append((name, to_epoch(start), to_epoch(end), dst))
    os.makedirs(os.path.join(output_dir, 'attack', name), exist_ok=True)

for name, _, _, _ in attack_ranges:
    os.makedirs(os.path.join(output_dir, 'attack', name), exist_ok=True)

print("📦 Streaming from PCAP using RawPcapReader...")
reader = PcapReader(pcap_path)
count = 0

MARGIN = 180

for pkt in tqdm(reader):
    try:
        if not pkt.haslayer(IP) or not pkt.haslayer(TCP):
            continue

        ts = pkt.time
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst

        if dst_ip == "192.168.10.50":
            # print(f"Skipping packet to NAT IP: {dst_ip}")
            for atk_name, start, end, dst in attack_ranges:
                if start <= ts <= end:
                    print(f"Skipping packet during attack {atk_name} to {dst_ip}")
                    break
                # else:
                    
            print(f"ts: {datetime.utcfromtimestamp(ts)}")

            # print(f"Packet time: {ts}")

        # label = "benign"
        # for atk_name, start, end, dst in attack_ranges:
        #     if start <= ts <= end and dst == dst_ip:
        #         label = atk_name
        #         break

        # byte_array = np.frombuffer(bytes(pkt), dtype=np.uint8)
        # image = mapping_methods[selected_mapping](byte_array, image_size)

        # if label == "benign":
        #     save_path = os.path.join(output_dir, "benign", f"benign_{count}.png")
        # else:
        #     save_path = os.path.join(output_dir, "attack", label, f"{label}_{count}.png")

        # Image.fromarray(image).save(save_path)
        # count += 1

    except Exception as e:
        continue

reader.close()
print(f"✅ Done! Total saved packets: {count}")
