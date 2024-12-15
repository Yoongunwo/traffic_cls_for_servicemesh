import os
import sys
from scapy.all import rdpcap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import binascii

current_dir = os.getcwd()  # C:\Users\gbwl3\Desktop\SourceCode\k8s_research
sys.path.append(current_dir)

def packet_to_bytes(packet):
    # 패킷을 raw 바이트로 변환
    return [int(b) for b in bytes(packet)]

def packet_to_image(packet_bytes, width=32):
    # 패킷 바이트를 0-255 사이의 값으로 정규화
    normalized = np.array([int(b) for b in packet_bytes], dtype=np.uint8)
    
    # 패딩 추가 (필요한 경우)
    if len(normalized) < width * width:
        padding = np.zeros(width * width - len(normalized), dtype=np.uint8)
        normalized = np.concatenate([normalized, padding])
    
    image = normalized[:width * width].reshape(width, width)

    return image

def save_packet_image(image_array, output_path, format='PNG', mode='single'):
    """
    이미지 배열을 파일로 저장
    
    Parameters:
    - image_array: 변환된 이미지 배열
    - output_path: 저장할 경로
    - format: 이미지 형식 (PNG, JPEG 등)
    - mode: 'single' - 흑백 이미지로 저장
            'colormap' - matplotlib 컬러맵 적용하여 저장
    """
    if mode == 'single':
        img_array = image_array.astype(np.uint8)
        # PIL Image로 변환하여 저장
        img = Image.fromarray(img_array)
        img.save(output_path, format)
    
    elif mode == 'colormap':
        # matplotlib 컬러맵 적용하여 저장
        plt.figure(figsize=(10, 10))
        plt.imshow(image_array, cmap='viridis')
        plt.colorbar(label='Byte value')
        plt.axis('off')
        plt.savefig(output_path, format=format, bbox_inches='tight', pad_inches=0)
        plt.close()

def process_pcap_to_images(pcap_file, output_dir, width=32, start_idx=0, max_packets=None):
    """
    PCAP 파일의 패킷들을 이미지로 변환하여 저장
    
    Parameters:
    - pcap_file: PCAP 파일 경로
    - output_dir: 이미지 저장할 디렉토리
    - width: 이미지 너비(정사각형)
    - start_idx: 시작할 패킷 인덱스
    - max_packets: 처리할 최대 패킷 수
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # PCAP 파일 읽기
    packets = rdpcap(pcap_file)
    
    # 처리할 패킷 범위 결정
    end_idx = len(packets) if max_packets is None else min(start_idx + max_packets, len(packets))
    
    for i in range(start_idx, end_idx):
        # 패킷을 이미지로 변환
        packet_bytes = packet_to_bytes(packets[i])
        image = packet_to_image(packet_bytes, width)
        
        # 이미지 파일 경로
        image_path = os.path.join(output_dir, f'packet_{i}.png')
        
        # 이미지 저장 (기본 흑백)
        save_packet_image(image, image_path)
        
        # 컬러맵 버전도 저장하고 싶다면:
        # colormap_path = os.path.join(output_dir, f'packet_{i}_colormap.png')
        # save_packet_image(image, colormap_path, mode='colormap')
        
        if (i - start_idx + 1) % 100 == 0:
            print(f'Processed {i - start_idx + 1} packets')

if __name__ == "__main__":
    data_root_path = "./Data/attack"
    pcap_file = os.path.join(data_root_path, "remote_access.pcap")
    output_dir = "./Data/attack/attack_to_byte/remote_access"  # 저장할 디렉토리
    
    # 전체 PCAP 파일 처리
    process_pcap_to_images(pcap_file, output_dir, width=32)