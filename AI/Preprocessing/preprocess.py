import os
import sys
from scapy.all import rdpcap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from hilbertcurve.hilbertcurve import HilbertCurve

current_dir = os.getcwd()  # C:\Users\gbwl3\Desktop\SourceCode\k8s_research
sys.path.append(current_dir)

def packet_to_bytes(packet):
    # 패킷을 raw 바이트로 변환
    return [int(b) for b in bytes(packet)]

def packet_to_image(packet_bytes, width=32): # row-major
    normalized = np.array([int(b) for b in packet_bytes], dtype=np.uint8)
    
    # padding
    if len(normalized) < width * width:
        padding = np.zeros(width * width - len(normalized), dtype=np.uint8)
        normalized = np.concatenate([normalized, padding])
    
    image = normalized[:width * width].reshape(width, width)

    return image

def spiral_inward_mapping(byte_array, image_size=16):
    padded = np.pad(byte_array, (0, image_size * image_size - len(byte_array)), 'constant')
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
    padded = np.pad(byte_array, (0, image_size * image_size - len(byte_array)), 'constant')
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
    padded = np.pad(byte_array, (0, image_size * image_size - len(byte_array)), 'constant')
    data = padded[:image_size * image_size]
    mat = np.zeros((image_size, image_size), dtype=np.uint8)

    p = int(np.log2(image_size))  # image_size = 2^p
    hilbert_curve = HilbertCurve(p, 2)
    for i in range(image_size * image_size):
        x, y = hilbert_curve.coordinates_from_distance(i)
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
    data_root_path = "./Data"
    pcap_file = os.path.join(data_root_path, "save_front.pcap")
    output_dir = "./Data/save/save_packet_to_byte_16/front_image"  # 저장할 디렉토리
    
    # 전체 PCAP 파일 처리
    process_pcap_to_images(pcap_file, output_dir, width=16)