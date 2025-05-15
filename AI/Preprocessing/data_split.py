import os
import random
import shutil

source_dir = "./images"  # or your actual path
all_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]

random.seed(42)
random.shuffle(all_files)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

total = len(all_files)
train_cut = int(total * train_ratio)
val_cut = int(total * (train_ratio + val_ratio))

train_files = all_files[:train_cut]
val_files = all_files[train_cut:val_cut]
test_files = all_files[val_cut:]

output_base = "./Data/"
os.makedirs(output_base, exist_ok=True)

for split_name, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
    split_dir = os.path.join(output_base, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    for f in split_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(split_dir, f)
        shutil.copy2(src, dst)

print(f"✅ result : {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")
