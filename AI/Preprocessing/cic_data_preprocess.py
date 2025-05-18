from scapy.all import RawPcapReader, Raw
import os, numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

CSV_PATH = './Data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv'
PCAP_PATH = './Data/cic_data/Wednesday-workingHours.pcap'
OUTPUT_DIR = './Data/cic_data/Wednesday-workingHours'
IMAGE_SIZE = 16

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
    if len(byte_array) < IMAGE_SIZE * IMAGE_SIZE:
        byte_array = np.pad(byte_array, (0, IMAGE_SIZE * IMAGE_SIZE - len(byte_array)), 'constant')
    else:
        byte_array = byte_array[:IMAGE_SIZE * IMAGE_SIZE]

    img = byte_array.reshape((IMAGE_SIZE, IMAGE_SIZE))
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
