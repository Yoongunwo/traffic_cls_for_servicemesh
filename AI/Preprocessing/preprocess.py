import os
import sys
from scapy.all import rdpcap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

from hilbertcurve.hilbertcurve import HilbertCurve

current_dir = os.getcwd()  # C:\Users\gbwl3\Desktop\SourceCode\k8s_research
sys.path.append(current_dir)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

BENIGN_PCAP_PATH = ['save_front', 'save_back', 'grafana', 'prometheus', 'jenkins', 'pgadmin', 'postgres']
# BENIGN_PCAP_PATH = ['grafana', 'prometheus', 'jenkins', 'pgadmin', 'postgres']
ATTACK_PCAP_PATHS = ['brute_force', 'kubernetes_enum', 'kubernetes_escape', 'kubernetes_manipulate', 'remote_access']

def packet_to_bytes(packet):
    # 패킷을 raw 바이트로 변환
    return [int(b) for b in bytes(packet)]

def row_wise_mapping(packet_bytes, width=32): # row-major
    normalized = np.array([int(b) for b in packet_bytes], dtype=np.uint8)
    
    # padding
    if len(normalized) < width * width:
        padding = np.zeros(width * width - len(normalized), dtype=np.uint8)
        normalized = np.concatenate([normalized, padding])
    
    image = normalized[:width * width].reshape(width, width)

    return image

def spiral_inward_mapping(byte_array, image_size=16):
    pad_len = max(0, image_size * image_size - len(byte_array))
    padded = np.pad(byte_array, (0, pad_len), 'constant')
    data = padded[:image_size * image_size]

    mat = np.zeros((image_size, image_size), dtype=np.uint8)

    top, bottom, left, right = 0, image_size-1, 0, image_size-1
    idx = 0
    while top <= bottom and left <= right:
        for i in range(left, right+1):  # Top row
            mat[top][i] = data[idx]; idx += 1
        top += 1
        for i in range(top, bottom+1):  # Right column
            mat[i][right] = data[idx]; idx += 1
        right -= 1
        if top <= bottom:
            for i in range(right, left-1, -1):  # Bottom row
                mat[bottom][i] = data[idx]; idx += 1
            bottom -= 1
        if left <= right:
            for i in range(bottom, top-1, -1):  # Left column
                mat[i][left] = data[idx]; idx += 1
            left += 1
    return mat

def diagonal_zigzag_mapping(byte_array, image_size=16):
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


def hilbert_mapping(byte_array, image_size=16):
    pad_len = max(0, image_size * image_size - len(byte_array))
    padded = np.pad(byte_array, (0, pad_len), 'constant')

    data = padded[:image_size * image_size]
    mat = np.zeros((image_size, image_size), dtype=np.uint8)

    p = int(np.log2(image_size))  # image_size = 2^p
    hilbert_curve = HilbertCurve(p, 2)
    for i in range(image_size * image_size):
        x, y = hilbert_curve.point_from_distance(i)
        mat[y][x] = data[i]  # y,x because PIL uses row,col
    return mat


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

def process_pcap_to_images_for_bengin_seq(pcap_file, output_root_dir, width=16, start_idx=0, max_packets=None, seed=42):
    """
    PCAP 파일의 패킷을 이미지로 변환하여 train/val/test 폴더에 무작위로 분할 저장
    """
    packets = rdpcap(pcap_file)
    total_packets = len(packets)
    if max_packets is not None:
        total_packets = min(total_packets, max_packets)

    # 순차적으로 index 분할
    # num_train = int(total_packets * train_ratio)
    # num_val = int(total_packets * val_ratio)
    # num_test = total_packets - num_train - num_val
    num_train = 50000
    num_val = 5000
    num_test = 5000

    split_sets = {
        'train': range(0, num_train),
        'val': range(num_train, num_train + num_val),
        'test': range(num_train + num_val, total_packets)
    }

    print(f"Sequential split: train={num_train}, val={num_val}, test={num_test}")

    for split, idx_list in split_sets.items():
        split_dir = os.path.join(output_root_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for i in idx_list:
            packet_bytes = packet_to_bytes(packets[i])
            image = hilbert_mapping(packet_bytes, width)
            image_path = os.path.join(split_dir, f'packet_{i}.png')
            save_packet_image(image, image_path)

        print(f"[{split.upper()}] Saved {len(idx_list)} images to: {split_dir}")

def process_pcap_to_images_for_bengin_shuffle(pcap_file, output_root_dir, width=16, start_idx=0, max_packets=None, seed=42):
    """
    PCAP 파일의 패킷을 이미지로 변환하여 train/val/test 폴더에 무작위로 분할 저장
    """
    packets = rdpcap(pcap_file)
    total_packets = len(packets)
    if max_packets is not None:
        total_packets = min(total_packets, max_packets)

    indices = list(range(total_packets))
    random.seed(seed)
    random.shuffle(indices)  # ✅ 셔플!

    num_train = int(total_packets * train_ratio)
    num_val = int(total_packets * val_ratio)
    num_test = total_packets - num_train - num_val

    split_sets = {
        'train': indices[:num_train],
        'val': indices[num_train:num_train + num_val],
        'test': indices[num_train + num_val:]
    }

    print(f"Shuffled split: train={num_train}, val={num_val}, test={num_test}")

    for split, idx_list in split_sets.items():
        split_dir = os.path.join(output_root_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for i in idx_list:
            packet_bytes = packet_to_bytes(packets[i])
            image = hilbert_mapping(packet_bytes, width)
            image_path = os.path.join(split_dir, f'packet_{i}.png')
            save_packet_image(image, image_path)

        print(f"[{split.upper()}] Saved {len(idx_list)} images to: {split_dir}")

def process_pcap_to_images_for_attack(pcap_file, output_dir, width=32, start_idx=0, max_packets=None):
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
        image = hilbert_mapping(packet_bytes, width)
        
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
    # for benign_pcap in BENIGN_PCAP_PATH:
    #     PCAP_PATH = f'./Data/benign/{benign_pcap}.pcap'
    #     DIR_PATH = f'./Data/byte_16_hilbert/{benign_pcap}'

    #     process_pcap_to_images_for_bengin(PCAP_PATH, DIR_PATH, width=16)
    
    for benign_pcap in BENIGN_PCAP_PATH:
        PCAP_PATH = f'./Data/benign/{benign_pcap}.pcap'
        DIR_PATH = f'./Data/byte_32_hilbert_seq/{benign_pcap}'

        process_pcap_to_images_for_bengin_seq(PCAP_PATH, DIR_PATH, width=32)

    # for attack_pcap in ATTACK_PCAP_PATHS:
    #     PCAP_PATH = f'./Data/attack/{attack_pcap}.pcap'
    #     DIR_PATH = f'./Data/byte_16_hilbert_attack/{attack_pcap}'
    #     process_pcap_to_images_for_attack(PCAP_PATH, DIR_PATH, width=16)      