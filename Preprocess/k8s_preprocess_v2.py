import os
import numpy as np
from scapy.all import PcapReader, IP, TCP
from datetime import datetime
from tqdm import tqdm

# === 설정 ===
pcap_path = './Data/benign/save_front.pcap'
# pcap_path = './Data/attack'
output_dir = './Data_k8s/Session_Windows_15'
os.makedirs(output_dir, exist_ok=True)

# === 패킷에서 row 추출 ===
def extract_packet_vector(pkt, payload_len=1460):
    f = []

    if IP in pkt:
        ip = pkt[IP]
        f.append(ip.ttl)
        f.append(ip.proto)
        f.append(ip.flags.value)
        f.extend([(ip.frag >> 8) & 0xFF, ip.frag & 0xFF])

    if TCP in pkt:
        tcp = pkt[TCP]
        f.append(tcp.dataofs)
        f.append(int(tcp.flags))
        f.extend([(tcp.window >> 8) & 0xFF, tcp.window & 0xFF])
        f.extend([(tcp.urgptr >> 8) & 0xFF, tcp.urgptr & 0xFF])
        f.extend(list(tcp.seq.to_bytes(4, 'big')))
        f.extend(list(tcp.ack.to_bytes(4, 'big')))

    payload = bytes(pkt.payload)[:payload_len]
    payload_arr = np.frombuffer(payload, dtype=np.uint8)
    if len(payload_arr) < payload_len:
        payload_arr = np.pad(payload_arr, (0, payload_len - len(payload_arr)), constant_values=0)
    f.extend(payload_arr.tolist())

    return np.array(f, dtype=np.uint8)

# === 세션에서 윈도우 생성 ===
def build_sliding_windows_from_session(packets, payload_len=1460, window_size=15):
    packets = sorted(packets, key=lambda pkt: pkt.time)
    feature_rows = []
    for pkt in packets:
        if not (IP in pkt and TCP in pkt): continue
        row = extract_packet_vector(pkt, payload_len=payload_len)
        feature_rows.append(row)
    if len(feature_rows) < window_size:
        return []
    feature_array = np.stack(feature_rows, axis=0)
    return [
        feature_array[i:i+window_size]
        for i in range(len(feature_array) - window_size + 1)
    ]

# === 세션 키 ===
def session_key(pkt):
    if IP in pkt and TCP in pkt:
        return (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, pkt[IP].proto)
    return None

# === 세션별 패킷 그룹화 ===
reader = PcapReader(pcap_path)
sessions = {}
timestamps = {}

for pkt in tqdm(reader):
    if not (IP in pkt and TCP in pkt): continue
    key = session_key(pkt)
    if not key:
        continue
    sessions.setdefault(key, []).append(pkt)
    timestamps.setdefault(key, pkt.time)

# === 슬라이딩 윈도우 이미지 저장 === 정상
benign_idx = 0
window_size = 15

for key, packets in sessions.items():
    windows = build_sliding_windows_from_session(packets, payload_len=1460, window_size=window_size)
    for i, win in enumerate(windows):
        np.save(os.path.join(output_dir, 'save_front', f'benign_{benign_idx}_{i}.npy'), win)
    benign_idx += 1

print(f"✅ Saved benign sessions: {benign_idx}")

# === 슬라이딩 윈도우 이미지 저장 === 공격
# pcap_dir = './Data/attack'  # 원래 pcap_path → pcap_dir로 수정
# output_dir = './Data_k8s/Session_Windows_15'
# os.makedirs(output_dir, exist_ok=True)

# for pcap_file in os.listdir(pcap_dir):
#     if not pcap_file.endswith(".pcap"):
#         continue

#     pcap_file_path = os.path.join(pcap_dir, pcap_file)  # 🔁 안전한 경로
#     attack_name = os.path.splitext(pcap_file)[0]
#     output_attack_dir = os.path.join(output_dir, attack_name)
#     os.makedirs(output_attack_dir, exist_ok=True)

#     print(f"🚀 Processing {pcap_file} ...")

#     reader = PcapReader(pcap_file_path)  # ⬅️ 여기도 변경

#     sessions = {}
#     for pkt in tqdm(reader):
#         if not (IP in pkt and TCP in pkt): continue
#         key = session_key(pkt)
#         if not key: continue
#         sessions.setdefault(key, []).append(pkt)

#     attack_idx = 0
#     for key, packets in sessions.items():
#         windows = build_sliding_windows_from_session(packets, payload_len=1460, window_size=15)
#         for i, win in enumerate(windows):
#             np.save(os.path.join(output_attack_dir, f'{attack_name}_{attack_idx}_{i}.npy'), win)
#         attack_idx += 1

#     print(f"✅ {attack_name}: Saved {attack_idx} sessions\n")