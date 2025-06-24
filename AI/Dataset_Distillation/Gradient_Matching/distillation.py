import os, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from glob import glob
from copy import deepcopy

# Config
DATA_DIR = './Data_CIC/Session_Windows_15'
H, W = 34, 44
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_SYN_IMAGES = 100
DISTILL_STEPS = 1000
INNER_LR = 0.1
OUT_DIM = 2

# === 파일 이름 파싱 ===
def load_image_paths(data_dir, per_class=1000):
    files = sorted(glob(os.path.join(data_dir, 'benign/*.npy')))[:per_class]
    return files

# === Custom Dataset ===
class CICBenignDataset(Dataset):
    def __init__(self, file_paths):
        self.frames = []
        for path in file_paths:
            x = np.load(path)
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            for t in range(x.shape[0]):
                vec = x[t]
                if vec.shape[0] < H * W:
                    vec = np.pad(vec, (0, H * W - vec.shape[0]))
                else:
                    vec = vec[:H * W]
                img = vec.reshape(1, H, W)
                self.frames.append(img)
        self.frames = torch.tensor(np.array(self.frames), dtype=torch.float32)
        self.labels = torch.zeros(len(self.frames), dtype=torch.long)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.labels[idx]

# === CNN Model ===
class SimpleCNN(nn.Module):
    def __init__(self, out_dim=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64 * 4 * 4, out_dim)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# === Distillation Logic ===
def compute_gradients(model, loss):
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    return grads

def set_model_weights(model, grads, lr):
    new_model = deepcopy(model)
    with torch.no_grad():
        for p, g in zip(new_model.parameters(), grads):
            p.data -= lr * g
    return new_model

def dataset_distillation(real_loader, model_fn, image_shape=(1, H, W), num_syn_images=10, steps=1000, lr=0.1):
    syn_images = torch.randn((num_syn_images,) + image_shape, requires_grad=True, device=DEVICE)
    syn_labels = torch.zeros(num_syn_images, dtype=torch.long, device=DEVICE)
    syn_optimizer = torch.optim.SGD([syn_images], lr=lr)

    for step in range(steps):
        model = model_fn().to(DEVICE)
        model.apply(init_weights)

        # Synthetic → weight update
        pred_syn = model(syn_images)
        loss_syn = F.cross_entropy(pred_syn, syn_labels)
        grads_syn = compute_gradients(model, loss_syn)
        model_updated = set_model_weights(model, grads_syn, lr=INNER_LR)

        # Real data → loss from updated model
        real_x, real_y = next(iter(real_loader))
        real_x, real_y = real_x.to(DEVICE), real_y.to(DEVICE)
        loss_real = F.cross_entropy(model_updated(real_x), real_y)

        syn_optimizer.zero_grad()
        loss_real.backward()
        syn_optimizer.step()

        if step % 50 == 0:
            print(f"[Step {step}] Real Loss: {loss_real.item():.4f}")

    return syn_images.detach().cpu()

# === 실행 ===
if __name__ == '__main__':
    print("Loading benign data...")
    benign_paths = load_image_paths(DATA_DIR, per_class=500)
    benign_dataset = CICBenignDataset(benign_paths)
    benign_loader = DataLoader(benign_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Running Dataset Distillation...")
    syn_data = dataset_distillation(
        real_loader=benign_loader,
        model_fn=lambda: SimpleCNN(out_dim=OUT_DIM),
        image_shape=(1, H, W),
        num_syn_images=NUM_SYN_IMAGES,
        steps=DISTILL_STEPS,
        lr=0.1
    )

    torch.save(syn_data, './AI/Dataset_Distillation/Gradient_Matching/distilled_images.pt')

    import matplotlib.pyplot as plt
    # save .png
    OUTPUT_DIR = './AI/Dataset_Distillation/Gradient_Matching/synthetic_images'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    syn_images = syn_data.squeeze(1).cpu().numpy()
    for i, img in enumerate(syn_images):
        plt.imsave(f'{OUTPUT_DIR}/syn_{i}.png', img, cmap='gray', vmin=0, vmax=1)

    print("Distilled synthetic images saved.")
