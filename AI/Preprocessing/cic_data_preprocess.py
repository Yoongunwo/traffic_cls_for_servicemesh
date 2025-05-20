from scapy.all import RawPcapReader, Raw
import os, numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

TYPE = 'spiral'

CSV_PATH = './Data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv'
PCAP_PATH = './Data/cic_data/Wednesday-workingHours.pcap'
OUTPUT_DIR = f'./Data/cic_data/Wednesday-workingHours/{TYPE}'
IMAGE_SIZE = 16

def packet_to_image(packet_bytes, width=32): # row-major
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
        x, y = hilbert_curve.coordinates_from_distance(i)
        mat[y][x] = data[i]  # y,x because PIL uses row,col
    return mat

# 라벨 분할
BENIGN_SPLIT = {'train': 0.7, 'val': 0.15, 'test': 0.15}
for split in BENIGN_SPLIT:
    os.makedirs(f'{OUTPUT_DIR}/benign_{split}', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/attack', exist_ok=True)

df = pd.read_csv(CSV_PATH).reset_index(drop=True)

# benign 샘플 분할
benign_indices = [i for i, lbl in enumerate(df[' Label']) if str(lbl).strip().lower() == 'benign']
np.random.seed(42)
np.random.shuffle(benign_indices)
total = len(benign_indices)
train_cut = int(total * BENIGN_SPLIT['train'])
val_cut = int(total * BENIGN_SPLIT['val'])

benign_map = {}
for i in benign_indices[:train_cut]:
    benign_map[i] = 'train'
for i in benign_indices[train_cut:train_cut + val_cut]:
    benign_map[i] = 'val'
for i in benign_indices[train_cut + val_cut:]:
    benign_map[i] = 'test'

# 저장용 카운터
counter = {'benign_train': 0, 'benign_val': 0, 'benign_test': 0, 'attack': 0}

# ✅ 스트리밍 방식으로 하나씩 처리
reader = RawPcapReader(PCAP_PATH)

for i, (pkt_data, _) in enumerate(tqdm(reader, total=len(df))):
    if i >= len(df):  # CSV보다 pcap이 더 길면 중단
        break

    label_raw = str(df.loc[i, ' Label']).strip().lower()

    pkt = None
    try:
        pkt = Raw(pkt_data)
    except:
        continue

    if not pkt or not pkt.load:
        continue

    payload = bytes(pkt.load)
    byte_array = np.frombuffer(payload, dtype=np.uint8)
    
    img = spiral_inward_mapping(byte_array, IMAGE_SIZE)

    im = Image.fromarray(img.astype(np.uint8), mode='L')

    if label_raw == 'benign':
        split = benign_map.get(i)
        save_path = f"{OUTPUT_DIR}/benign_{split}/benign_{counter[f'benign_{split}']:05d}.png"
        counter[f'benign_{split}'] += 1
    else:
        save_path = f"{OUTPUT_DIR}/attack/attack_{counter['attack']:05d}.png"
        counter['attack'] += 1

    im.save(save_path)

reader.close()

# ✅ 결과 출력
print("\n✅ 이미지 변환 완료")
for k, v in counter.items():
    print(f"{k}: {v}장")
