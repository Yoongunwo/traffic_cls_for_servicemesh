import os
import numpy as np
from scapy.all import rdpcap
from PIL import Image

def save_packet_image(image_array, output_path):
    img_array = image_array.astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(output_path, format='PNG')

def process_pcap_stream_to_images(pcap_path, output_dir, width=32, max_images=50):
    os.makedirs(output_dir, exist_ok=True)
    packets = rdpcap(pcap_path)

    current_bytes = []
    packet_lens = []
    image_count = 0
    i = 0
    max_packets_per_image = 4
    target_bytes = width * width

    # 초기 최대 4개 패킷 누적
    while len(packet_lens) < max_packets_per_image:
        payload = bytes(packets[i])
        if payload:
            current_bytes.extend(payload)

            if len(current_bytes) > target_bytes:
                excess = len(current_bytes) - target_bytes
                trim_len = len(payload) - excess
                if trim_len > 0:
                    current_bytes = current_bytes[:target_bytes]
                    packet_lens.append(trim_len)
                break
            else:
                packet_lens.append(len(payload))
        i += 1

    # 반복: 이후부터는 슬라이딩 방식
    while i < len(packets) and image_count < max_images:
        # 이미지로 만들 수 있는 충분한 바이트가 있는가?
        if len(current_bytes) >= target_bytes:
            byte_data = current_bytes[:target_bytes]
        else:
            byte_data = current_bytes + [0] * (target_bytes - len(current_bytes))  # 제로 패딩

        byte_array = np.array(byte_data, dtype=np.uint8)
        image = byte_array.reshape((width, width))
        image_path = os.path.join(output_dir, f'image_{image_count:05d}.png')
        save_packet_image(image, image_path)
        image_count += 1

        # 가장 오래된 패킷 제거 (슬라이딩)
        removed_len = packet_lens.pop(0)
        current_bytes = current_bytes[removed_len:]

        # 최대 4개 유지 위해 다음 패킷 추가
        while i < len(packets) and len(packet_lens) < max_packets_per_image:
            payload = bytes(packets[i])
            if payload:
                current_bytes.extend(payload)

                if len(current_bytes) > target_bytes:
                    excess = len(current_bytes) - target_bytes
                    trim_len = len(payload) - excess
                    if trim_len > 0:
                        current_bytes = current_bytes[:target_bytes]
                        packet_lens.append(trim_len)
                    break
                else:
                    packet_lens.append(len(payload))
            i += 1

    print(f"✅ 총 {image_count}개 이미지 저장 완료 at {output_dir}")

# BENIGN_PCAP_PATH = ['grafana', 'prometheus', 'jenkins', 'pgadmin', 'postgres']
BENIGN_PCAP_PATH = ['save_front', 'save_back']
ATTACK_PCAP_PATHS = ['brute_force', 'kubernetes_enum', 'kubernetes_escape', 'kubernetes_manipulate', 'remote_access']

if __name__ == "__main__":
    for benign_pcap in BENIGN_PCAP_PATH:
        PCAP_PATH = f"./Data/benign/{benign_pcap}.pcap"  # <- 여기에 pcap 경로
        OUTPUT_DIR = f"./Data/32x32/{benign_pcap}"
        process_pcap_stream_to_images(PCAP_PATH, OUTPUT_DIR, width=32, max_images=50)
    # for attack_pcap in ATTACK_PCAP_PATHS:
    #     PCAP_PATH = f"./Data/attack/{attack_pcap}.pcap"  # <- 여기에 pcap 경로
    #     OUTPUT_DIR = f"./Data/32x32/attack/{attack_pcap}"
    #     process_pcap_stream_to_images(PCAP_PATH, OUTPUT_DIR, width=32, max_images=50)
