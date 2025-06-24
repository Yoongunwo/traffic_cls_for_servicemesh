import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import re
import random

# ✅ Dataset with 4-channel stat input
class StatisticalChannelDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, window_size=4):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.window_size = window_size

    def __len__(self):
        return len(self.image_paths) - self.window_size + 1

    def __getitem__(self, idx):
        imgs = [self.transform(Image.open(self.image_paths[idx+i]).convert('L')) for i in range(self.window_size)]
        stack = torch.stack(imgs, dim=0).squeeze(1)
        stat = torch.stack([
            stack.mean(dim=0),
            stack.std(dim=0),
            stack.min(dim=0).values,
            stack.max(dim=0).values
        ], dim=0)  # [4, H, W]
        return stat, self.labels[idx + self.window_size - 1]

# ✅ MAE for 4-channel 16x16 input
class MAE(nn.Module):
    def __init__(self, in_ch=4, embed_dim=64, patch_size=4, img_size=16, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio

        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True), 
            num_layers=4
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * in_ch),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, D]

        # Random mask
        N = patches.size(1)
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        patches_keep = torch.gather(patches, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, patches.size(2)))

        enc = self.encoder(patches_keep)
        dec = self.decoder(enc)  # [B, keep, patch_dim]

        # reconstruct and place patches
        recon_patches = torch.zeros(B, N, self.patch_size**2 * C, device=x.device)
        recon_patches.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, dec.size(2)), dec)
        recon = recon_patches.view(B, H//self.patch_size, W//self.patch_size, C, self.patch_size, self.patch_size)
        recon = recon.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
        return recon

# ✅ 평가
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        score = ((x - out) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        all_scores.extend(score)
        all_labels.extend(y.numpy())

    threshold = np.percentile(all_scores, 95)
    preds = (np.array(all_scores) > threshold).astype(int)
    print(classification_report(all_labels, preds, digits=4))
    print("ROC AUC:", roc_auc_score(all_labels, all_scores))

# ✅ 정렬 함수
def natural_key(string):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def get_paths(path, label):
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')], key=natural_key)
    return [os.path.join(path, f) for f in files], [label] * len(files)

ROOT = './Data/cic_data/Wednesday-workingHours/hilbert_seq'
WIN = 4
BATCH = 1024 * 16
EPOCHS = 25
# ✅ 전체 파이프라인
def run_mae_train_eval():
    print(f"Statistic MAE Window size: {WIN}, Data: {ROOT}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)
    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])

    benign_ds = StatisticalChannelDataset(benign_p[:55000+WIN-1], benign_l[:55000+WIN-1], transform, WIN)
    attack_ds = StatisticalChannelDataset(attack_p[:5000+WIN-1], attack_l[:5000+WIN-1], transform, WIN)

    train_idx, val_idx = train_test_split(list(range(len(benign_ds))), test_size=5000, random_state=42)
    train_ds = Subset(benign_ds, train_idx)
    val_ds = Subset(benign_ds, val_idx)

    test_ds = ConcatDataset([val_ds, attack_ds])
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH)

    model = MAE(in_ch=4, img_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.to(device)
    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for x, _ in train_loader:
            x = x.to(device)
            recon = model(x)
            loss = loss_fn(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1} Loss: {total / len(train_loader):.4f}")

    torch.save(model.state_dict(), './AI/Statistic/MAE/Model/mae_model_{WIN}.pth')
    evaluate(model, test_loader, device)

def start_mae_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)
    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])

    benign_ds = StatisticalChannelDataset(benign_p[:55000+WIN-1], benign_l[:55000+WIN-1], transform, WIN)
    attack_ds = StatisticalChannelDataset(attack_p[:5000+WIN-1], attack_l[:5000+WIN-1], transform, WIN)

    train_idx, val_idx = train_test_split(list(range(len(benign_ds))), test_size=5000, random_state=42)
    val_ds = Subset(benign_ds, val_idx)

    test_ds = ConcatDataset([val_ds, attack_ds])
    test_loader = DataLoader(test_ds, batch_size=BATCH)

    model = MAE(in_ch=4, img_size=16)
    model.load_state_dict(torch.load('./AI/Statistic/MAE/Model/mae_model_{WIN}_{EPOCHS}.pth'))
    model.to(device)
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    run_mae_train_eval()
    # start_mae_eval()
