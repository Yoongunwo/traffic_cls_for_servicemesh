import os
import numpy as np
from scapy.all import PcapReader, IP, TCP
from datetime import datetime
from tqdm import tqdm

# === 설정 ===
pcap_path = './Data/cic_data/Friday-WorkingHours.pcap'
output_dir = './Data_CIC/Session_Windows_15'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'benign'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'attack'), exist_ok=True)

def to_epoch(dt_str, date_str="2017-07-07"):
    dt = datetime.strptime(f"{date_str} {dt_str}", "%Y-%m-%d %H:%M")
    return int(dt.timestamp()) + 3 * 3600

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
    ("PortScan", "14:51", "14:53", ["192.168.10.50"]),
    ("PortScan", "14:54", "14:56", ["192.168.10.50"]),
    ("PortScan", "14:57", "14:59", ["192.168.10.50"]),
    ("PortScan", "15:00", "15:02", ["192.168.10.50"]),
    ("PortScan", "15:03", "15:05", ["192.168.10.50"]),
    ("PortScan", "15:06", "15:07", ["192.168.10.50"]),
    ("PortScan", "15:08", "15:10", ["192.168.10.50"]),
    ("PortScan", "15:11", "15:12", ["192.168.10.50"]),
    ("PortScan", "15:13", "15:15", ["192.168.10.50"]),
    ("PortScan", "15:16", "15:18", ["192.168.10.50"]),
    ("PortScan", "15:19", "15:21", ["192.168.10.50"]),
    ("PortScan", "15:22", "15:24", ["192.168.10.50"]),
    ("PortScan", "15:25", "15:25", ["192.168.10.50"]),
    ("PortScan", "15:26", "15:27", ["192.168.10.50"]),
    ("PortScan",  "15:28", "15:29", ["192.168.10.50"]),
    ("LOIT", "15:56", "16:16", ["192.168.10.50"])
]

attack_ranges = [(name, to_epoch(start), to_epoch(end), dsts) for name, start, end, dsts in attack_windows]

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

# === 슬라이딩 윈도우 이미지 저장 ===
benign_idx, attack_idx = 0, 0
window_size = 15

for key, packets in sessions.items():
    label = 'benign'
    for pkt in packets:
        ts = pkt.time
        dst = pkt[IP].dst
        for atk_name, start, end, dst_list in attack_ranges:
            if start <= ts <= end and dst in dst_list:
                label = atk_name
                break
        if label != 'benign':
            break

    windows = build_sliding_windows_from_session(packets, payload_len=1460, window_size=window_size)
    for i, win in enumerate(windows):
        if label == 'benign':
            np.save(os.path.join(output_dir, 'benign', f'benign_{benign_idx}_{i}.npy'), win)
        else:
            atk_dir = os.path.join(output_dir, 'attack', label)
            os.makedirs(atk_dir, exist_ok=True)
            np.save(os.path.join(atk_dir, f'{label}_{attack_idx}_{i}.npy'), win)
    if label == 'benign':
        benign_idx += 1
    else:
        attack_idx += 1


print(f"✅ Saved benign sessions: {benign_idx}, attack sessions: {attack_idx}")
