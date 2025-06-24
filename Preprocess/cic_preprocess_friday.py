import os
import sys
from scapy.all import rdpcap, IP, TCP, PcapReader, RawPcapReader, Ether
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
from hilbertcurve.hilbertcurve import HilbertCurve

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

mapping_methods = {
    "row": row_wise_mapping,
    "spiral": spiral_inward_mapping,
    "zigzag": diagonal_zigzag_mapping,
    "hilbert": hilbert_mapping,
}

selected_mapping = "hilbert"  # 🔁 원하는 방식 선택 가능: row, spiral, zigzag, hilbert
image_size = 32

date_str = "2017-07-07"
pcap_path = './Data/cic_data/Friday-WorkingHours.pcap'
output_dir = f'./Data_CIC/Fri_{selected_mapping}_{image_size}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'benign'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'attack'), exist_ok=True)

def to_epoch(dt_str, date_str="2017-07-07"):
    dt = datetime.strptime(f"{date_str} {dt_str}", "%Y-%m-%d %H:%M")
    return int(dt.timestamp()) + 3 * 3600  # UTC-4 보정

# (공격명, 시작시간, 끝시간, [공격자IP 리스트])
attack_windows = [
    ("Botnet", "10:02", "11:02", ["192.168.10.15", "192.168.10.9", "192.168.10.14", "192.168.10.5", "192.168.10.8"]),
    ("PortScan", "13:55", "13:57", ["192.168.10.50"]),
    ("PortScan", "13:58", "14:00", ["192.168.10.50"]),
    ("PortScan", "14:01", "14:04", ["192.168.10.50"]),
    ("PortScan", "14:05", "14:07", ["192.168.10.50"]),
    ("PortScan", "14:08", "14:10", ["192.168.10.50"]),
    ("PortScan", "14:11", "14:13", ["192.168.10.50"]),
    ("PortScan", "14:14", "14:16", ["192.168.10.50"]),
    ("PortScan", "14:17", "14:19", ["192.168.10.50"]),
    ("PortScan", "14:20", "14:21", ["192.168.10.50"]),
    ("PortScan", "14:22", "14:24", ["192.168.10.50"]),
    ("PortScan", "14:33", "14:33", ["192.168.10.50"]),
    ("PortScan", "14:35", "14:35", ["192.168.10.50"]),
    ("sS", "14:51", "14:53", ["192.168.10.50"]),
    ("sT", "14:54", "14:56", ["192.168.10.50"]),
    ("sF", "14:57", "14:59", ["192.168.10.50"]),
    ("sX", "15:00", "15:02", ["192.168.10.50"]),
    ("sN", "15:03", "15:05", ["192.168.10.50"]),
    ("sP", "15:06", "15:07", ["192.168.10.50"]),
    ("sV", "15:08", "15:10", ["192.168.10.50"]),
    ("sU", "15:11", "15:12", ["192.168.10.50"]),
    ("sO", "15:13", "15:15", ["192.168.10.50"]),
    ("sA", "15:16", "15:18", ["192.168.10.50"]),
    ("sW", "15:19", "15:21", ["192.168.10.50"]),
    ("sR", "15:22", "15:24", ["192.168.10.50"]),
    ("sL", "15:25", "15:25", ["192.168.10.50"]),
    ("sI", "15:26", "15:27", ["192.168.10.50"]),
    ("b",  "15:28", "15:29", ["192.168.10.50"]),
    ("LOIT", "15:56", "16:16", ["192.168.10.50"])
]

attack_ranges = []
for name, start, end, dst_list in attack_windows:
    attack_ranges.append((name, to_epoch(start), to_epoch(end), dst_list))
    os.makedirs(os.path.join(output_dir, 'attack', name), exist_ok=True)

# === pcap 로딩 및 필터링 ===
print("📦 Loading packets...")
reader = PcapReader(pcap_path)
count = 6_040_000

from itertools import islice

for pkt in tqdm(islice(reader, count, None)):
# for pkt in tqdm(reader):
    if IP in pkt and TCP in pkt:
        ts = pkt.time
        src = pkt[IP].src
        dst = pkt[IP].dst

        label = "benign"
        for atk_name, start, end, dst_list in attack_ranges:
            if start <= ts <= end and dst in dst_list:
                label = atk_name
                print(f"ts: {datetime.utcfromtimestamp(ts)}")
                break

        byte_arr = np.frombuffer(bytes(pkt), dtype=np.uint8)
        image = mapping_methods[selected_mapping](byte_arr, image_size)

        if label == "benign":
            continue
            # save_path = os.path.join(output_dir, 'benign', f"benign_{count}.png")
        else:
            save_path = os.path.join(output_dir, 'attack', label, f"{label}_{count}.png")
            Image.fromarray(image).save(save_path)
            count += 1

print(f"✅ Done! Total saved packets: {count}")
