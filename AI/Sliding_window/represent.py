import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# 설정
DATA_DIR = './Data_CIC/Session_Windows_15'
OUTPUT_DIR = './AI/Sliding_window/represent'
os.makedirs(os.path.join(OUTPUT_DIR, 'benign'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'attack'), exist_ok=True)

# npy → 이미지 저장 함수
def save_npy_as_image(npy_path, save_path):
    data = np.load(npy_path)  # shape: [15, 1479]
    print(f"Processing {npy_path} with shape {data.shape}")
    
    height, width = data.shape
    aspect_ratio = width / height

    plt.figure(figsize=(aspect_ratio * 4, 4))  # 가로/세로 비율 맞게
    plt.imshow(data, cmap='gray', aspect='auto')  # or aspect='equal'도 가능
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# benign 10개 시각화
benign_files = sorted(glob(os.path.join(DATA_DIR, 'benign', '*.npy')))[:10]
for i, path in enumerate(benign_files):
    save_npy_as_image(path, os.path.join(OUTPUT_DIR, 'benign', f'benign_{i}.png'))

# attack 10개 시각화 (모든 attack 타입에서)
# attack_files = []
# attack_dirs = sorted(glob(os.path.join(DATA_DIR, 'attack', '*')))

# for d in attack_dirs:
#     npy_files = sorted(glob(os.path.join(d, '*.npy')))
#     if npy_files:
#         attack_files.extend(npy_files[:5])  # 각 공격 폴더에서 하나만 선택

# for i, path in enumerate(attack_files):
#     attack_type = path.split(os.sep)[-2]
#     save_path = os.path.join(OUTPUT_DIR, 'attack', f'{attack_type}_{i}.png')
#     save_npy_as_image(path, save_path)

print("✅ 10 benign + 10 attack samples saved as PNG.")
