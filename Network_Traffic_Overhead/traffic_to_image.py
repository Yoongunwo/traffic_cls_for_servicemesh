import os
import numpy as np
from scapy.all import PcapReader, IP, TCP
from tqdm import tqdm
from collections import deque
import time

# === 설정 ===
pcap_path = './Data/benign/save_front.pcap'
output_dir = './Data_k8s/Session_Windows_15/save_front'
os.makedirs(output_dir, exist_ok=True)

# === 파라미터 설정 ===
payload_len = 1460
window_size = 15

# === 패킷에서 row 추출 ===
def extract_packet_vector(pkt):
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

    payload = bytes(pkt.payload)[:payload_len]
    if len(payload) < payload_len:
        payload += b'\x00' * (payload_len - len(payload))
    f.extend(payload)

    return np.array(f, dtype=np.uint8)

# === 세션 키 ===
def session_key(pkt):
    if IP in pkt and TCP in pkt:
        return (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, pkt[IP].proto)
    return None

# === Reader 준비 ===
reader = PcapReader(pcap_path)

sessions = {}
detection_latencies = []
benign_idx = 0

# === 본 처리 루프 ===
for pkt in tqdm(reader):
    if not (IP in pkt and TCP in pkt):
        continue

    start_time = time.perf_counter()
    key = session_key(pkt)
    if not key:
        continue

    if key not in sessions:
        sessions[key] = deque(maxlen=window_size)
    sessions[key].append(extract_packet_vector(pkt))

    if len(sessions[key]) == window_size:

        feature_array = np.stack(sessions[key], axis=0)
        elapsed = time.perf_counter() - start_time
        detection_latencies.append(elapsed)

        # np.save(os.path.join(output_dir, f'benign_{benign_idx}_0.npy'), feature_array)
        # benign_idx += 1

# === 결과 출력 ===
if detection_latencies:
    detection_latencies_ms = np.array(detection_latencies) * 1000
    print(f"\n✅ {benign_idx} sliding windows saved.")
    print(f"📈 Average latency: {detection_latencies_ms.mean():.3f} ms")
    print(f"📉 Std deviation: {detection_latencies_ms.std():.3f} ms")
else:
    print("⚠️ No session reached the required window size.")
