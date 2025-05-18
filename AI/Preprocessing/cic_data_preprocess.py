import os
import pandas as pd
from scapy.all import rdpcap, Raw
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

# ✅ 설정
CSV_PATH = './Data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv'
PCAP_PATH = './Data/cic_data/Wednesday-workingHours.pcap'
OUTPUT_DIR = './Data/cic_data/Wednesday-workingHours'
IMAGE_SIZE = 16

BENIGN_SPLIT = {'train': 0.7, 'val': 0.15, 'test': 0.15}

for split in BENIGN_SPLIT:
    os.makedirs(f'{OUTPUT_DIR}/benign_{split}', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/attack', exist_ok=True)

df = pd.read_csv(CSV_PATH).reset_index(drop=True)

packets = rdpcap(PCAP_PATH)
print(f"총 패킷 수: {len(packets)}")

benign_indices = [i for i, lbl in enumerate(df['label']) if str(lbl).strip().lower() == 'benign']
random.seed(42)
random.shuffle(benign_indices)
total = len(benign_indices)
train_cut = int(BENIGN_SPLIT['train'] * total)
val_cut = int(BENIGN_SPLIT['val'] * total)

benign_split_map = {}
for i in benign_indices[:train_cut]:
    benign_split_map[i] = 'train'
for i in benign_indices[train_cut:train_cut+val_cut]:
    benign_split_map[i] = 'val'
for i in benign_indices[train_cut+val_cut:]:
    benign_split_map[i] = 'test'

counter = {'benign_train': 0, 'benign_val': 0, 'benign_test': 0, 'attack': 0}

for i in tqdm(range(min(len(df), len(packets)))):
    label_raw = str(df.loc[i, 'label']).strip().lower()
    pkt = packets[i]

    if not pkt.haslayer(Raw):
        continue

    payload = bytes(pkt[Raw].load)
    byte_array = np.frombuffer(payload, dtype=np.uint8)
    if len(byte_array) < IMAGE_SIZE * IMAGE_SIZE:
        byte_array = np.pad(byte_array, (0, IMAGE_SIZE * IMAGE_SIZE - len(byte_array)), 'constant')
    else:
        byte_array = byte_array[:IMAGE_SIZE * IMAGE_SIZE]

    img = byte_array.reshape((IMAGE_SIZE, IMAGE_SIZE))
    im = Image.fromarray(img.astype(np.uint8), mode='L')

    if label_raw == 'benign':
        split = benign_split_map.get(i)
        save_path = f"{OUTPUT_DIR}/benign_{split}/benign_{counter[f'benign_{split}']:05d}.png"
        counter[f'benign_{split}'] += 1
    else:
        save_path = f"{OUTPUT_DIR}/attack/attack_{counter['attack']:05d}.png"
        counter['attack'] += 1

    im.save(save_path)

print("\nImage saved successfully")
for k, v in counter.items():
    print(f"{k}: {v}")
