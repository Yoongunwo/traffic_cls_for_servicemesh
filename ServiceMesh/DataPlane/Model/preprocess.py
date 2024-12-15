import os
import sys
import numpy as np
from PIL import Image
import time
import torch

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

    return image_to_tensor_image(Image.fromarray(image))

def image_to_tensor_image(image):
    image_tensor = torch.from_numpy(np.array(image)).float()
    image_tensor = image_tensor.unsqueeze(0)  # 채널 추가
    image_tensor = image_tensor.unsqueeze(0)  # 배치 추가
    return image_tensor.to('cpu')

async def image_save(image, base_path, format='PNG'):
    image_array = image.cpu().detach().numpy()
    image = Image.fromarray(image_array[0][0])

    timestamp = int(time.time() * 1000)  # 밀리초 단위
    filename = f"image_{timestamp}.{format.lower()}"
    full_path = os.path.join(base_path, filename)
    image.save(full_path, format)

# if __name__ == "__main__":
#     data_root_path = "./Data/attack"
#     pcap_file = os.path.join(data_root_path, "remote_access.pcap")
#     output_dir = "./Data/attack/attack_to_byte/remote_access"  # 저장할 디렉토리
    
#     # 전체 PCAP 파일 처리
#     process_pcap_to_images(pcap_file, output_dir, width=32)