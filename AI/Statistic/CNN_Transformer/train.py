import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import ConcatDataset

# ✅ Dataset with statistical augmentation (mean, std, min, max)
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
        stack = torch.stack(imgs, dim=0).squeeze(1)  # [T, H, W]
        mean_img = stack.mean(dim=0, keepdim=True)
        std_img = stack.std(dim=0, keepdim=True)
        min_img = stack.min(dim=0, keepdim=True).values
        max_img = stack.max(dim=0, keepdim=True).values
        stat_stack = torch.cat([mean_img, std_img, min_img, max_img], dim=0)
        label = self.labels[idx + self.window_size - 1]
        return stat_stack, label

# ✅ CNN + Transformer Autoencoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class CNNTransformerAE(nn.Module):
    def __init__(self, input_channels=4, d_model=128, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Conv2d(input_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.recon = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_channels * patch_size * patch_size)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.embed(x)  # [B, d_model, H', W']
        patches = patches.flatten(2).transpose(1, 2)  # [B, N, d_model]
        patches = self.pos_enc(patches)
        encoded = self.transformer(patches)  # [B, N, d_model]

        decoded = self.recon(encoded)  # [B, N, C*patch*patch]
        decoded = decoded.view(B, -1, C, self.patch_size, self.patch_size)  # [B, N, C, pH, pW]

        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        decoded = decoded.permute(0, 2, 1, 3, 4)
        decoded = decoded.reshape(B, C, h_patches, w_patches, self.patch_size, self.patch_size)
        decoded = decoded.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)
        return decoded

# ✅ Train + Eval
@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    scores, labels = [], []
    for x, y in test_loader:
        x = x.to(device)
        recon = model(x)
        score = ((x - recon) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
        scores.extend(score)
        labels.extend(y.numpy())

    threshold = np.percentile(scores, 95)
    preds = (np.array(scores) > threshold).astype(int)
    print(classification_report(labels, preds, digits=4))
    print("ROC AUC:", roc_auc_score(labels, scores))


def train_and_evaluate(model, train_loader, test_loader, device, epochs=10):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            out = model(x)
            loss = criterion(out, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    evaluate(model, test_loader, device)
    return model

def train_start():
    PREPROCESSING_TYPE = 'hilbert'
    ROOT = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq'
    win = 16
    BATCH_SIZE = 1024 * 4

    def get_paths_and_labels(path, label):
        files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
        return [os.path.join(path, f) for f in files], [label]*len(files)

    benign_paths, benign_labels = get_paths_and_labels(os.path.join(ROOT, 'benign_train'), 0)
    attack_paths, attack_labels = get_paths_and_labels(os.path.join(ROOT, 'attack'), 1)

    train_p, train_l = benign_paths[:50000], benign_labels[:50000]
    val_p, val_l = benign_paths[:5000], benign_labels[:5000]
    val_attack_p, val_attack_l = attack_paths[:5000], attack_labels[:5000]

    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
    train_ds = StatisticalChannelDataset(train_p, train_l, transform, window_size=win)
    val_ds_b = StatisticalChannelDataset(val_p, val_l, transform, window_size=win)
    val_ds_a = StatisticalChannelDataset(val_attack_p, val_attack_l, transform, window_size=win)
    test_ds = ConcatDataset([val_ds_b, val_ds_a])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = CNNTransformerAE()
    model = train_and_evaluate(model, train_loader, test_loader, device, epochs=10)

    torch.save(model.state_dict(), f'./AI/Statistic/CNN_Transformer/Model/stat_cnn_transformer_ae_{win}.pth')

def evaluate_start():
    PREPROCESSING_TYPE = 'hilbert'
    ROOT = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq'
    win = 9
    BATCH_SIZE = 1024 * 4

    def get_paths_and_labels(path, label):
        files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
        return [os.path.join(path, f) for f in files], [label]*len(files)

    benign_paths, benign_labels = get_paths_and_labels(os.path.join(ROOT, 'benign_train'), 0)
    attack_paths, attack_labels = get_paths_and_labels(os.path.join(ROOT, 'attack'), 1)

    val_p, val_l = benign_paths[:5000], benign_labels[:5000]
    val_attack_p, val_attack_l = attack_paths[:5000], attack_labels[:5000]

    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])

    val_ds_b = StatisticalChannelDataset(val_p, val_l, transform, window_size=win)
    val_ds_a = StatisticalChannelDataset(val_attack_p, val_attack_l, transform, window_size=win)
    
    test_ds = ConcatDataset([val_ds_b, val_ds_a])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = CNNTransformerAE()
    model.load_state_dict(
        torch.load(
            f'./AI/Statistic/CNN_Transformer/Model/stat_cnn_transformer_ae_{win}.pth', 
            weights_only=True
    ))

    model = model.to(device)
    evaluate(model, test_loader, device)


# ✅ Run
if __name__ == '__main__':
    # train_start()
    evaluate_start()  # Uncomment to run evaluation
