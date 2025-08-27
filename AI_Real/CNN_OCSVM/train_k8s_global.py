import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import random

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_k8s/Session_Windows_15"
H, W = 34, 44
BATCH_SIZE = 2**11
EPOCHS = 10
FEAT_DIM = 128
TEMPERATURE = 0.1
SEED = 42

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# === Dataset for Contrastive ===
class ContrastiveDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.HW = H * W

    def __len__(self): 
        return len(self.paths)

    def augment(self, vec):
        noise = np.random.normal(0, 0.01, size=vec.shape)
        return np.clip(vec + noise, 0, 1)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])[0]
        x = np.nan_to_num(x)
        x = np.clip(x, 0, 255).astype(np.float32) / 255.0
        if x.shape[0] < self.HW:
            x = np.pad(x, (0, self.HW - x.shape[0]))
        vec = x[:self.HW]
        img1 = self.augment(vec).reshape(1, H, W)
        img2 = self.augment(vec).reshape(1, H, W)
        return torch.tensor(img1, dtype=torch.float32), torch.tensor(img2, dtype=torch.float32)

# === Encoder ===
class Encoder(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class Encoder_4x128(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )
    def forward(self, x): return self.net(x)

# === NT-Xent Loss ===
def nt_xent(z1, z2, temperature=TEMPERATURE):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # [2N, D]
    N = z1.shape[0]

    sim = torch.matmul(z, z.T) / temperature  # [2N, 2N]
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)  # remove self-similarity

    # labels[i] = positive index for row i
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels])  # [N..2N-1, 0..N-1]

    return F.cross_entropy(sim, labels)

# === Utilities ===
def list_benign_folders(data_dir, exclude=("attack",)):
    """data_dir 바로 아래의 서브폴더 중 공격 폴더 제외하고 반환"""
    subs = []
    for name in os.listdir(data_dir):
        p = os.path.join(data_dir, name)
        if os.path.isdir(p) and name not in exclude and not name.startswith('.'):
            subs.append(name)
    subs = sorted(subs)
    if not subs:
        raise RuntimeError(f"No benign folders found in {data_dir}.")
    print(f"[INFO] Benign folders: {subs}")
    return subs

def list_npy_in_folder(data_dir, folder):
    """폴더 바로 아래의 .npy 파일만 (정렬)"""
    return sorted(glob(os.path.join(data_dir, folder, "*.npy")))

def take_train_test_from_folder(paths, n_train=2000, n_test=500):
    """
    경로 리스트를 받아 앞쪽 n_train, 그 다음 n_test를 반환.
    파일 수가 부족하면 가능한 만큼만 사용.
    """
    if len(paths) < n_train + n_test:
        folder_name = os.path.basename(os.path.dirname(paths[0])) if paths else "folder"
        print(f"[WARN] {folder_name}: requested {n_train+n_test}, but found {len(paths)} files.")
    train = paths[:min(n_train, len(paths))]
    remain = paths[len(train):]
    test  = remain[:min(n_test, len(remain))]
    return train, test

def load_attack_paths(data_dir, limit):
    """attack/*/*.npy 구조 가정"""
    pattern = os.path.join(data_dir, "attack", "*", "*.npy")
    paths = sorted(glob(pattern))[:limit]
    if len(paths) < limit:
        print(f"[WARN] attack: requested {limit} but found {len(paths)} files.")
    return paths

def extract_features(paths, model):
    model.eval()
    feats = []
    HW = H * W
    with torch.no_grad():
        for f in paths:
            x = np.load(f)[0]
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            if x.shape[0] < HW:
                x = np.pad(x, (0, HW - x.shape[0]))
            img = x[:HW].reshape(1, H, W)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z = model(img).squeeze(0).cpu().numpy()
            feats.append(z)
    return np.array(feats)

# === Training Contrastive Encoder ===
def train_encoder(train_paths):
    print("Training contrastive encoder...")
    dataset = ContrastiveDataset(train_paths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # model = Encoder().to(DEVICE)
    model = Encoder_4x128().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x1, x2 in dataloader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            z1, z2 = model(x1), model(x2)
            loss = nt_xent(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        denom = max(1, len(dataloader))
        print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss / denom:.4f}")

    os.makedirs('./AI_Real/CNN_OCSVM/Model/global', exist_ok=True)
    torch.save(model.state_dict(), f'./AI_Real/CNN_OCSVM/Model/global/cnn_contrastive_encoder_k8s_{TEMPERATURE}_{EPOCHS}_4x128.pth')
    return model

# === Main Pipeline (generalized) ===
def main():
    # 1) Benign 폴더 자동 탐색 (attack 제외)
    benign_folders = list_benign_folders(DATA_DIR, exclude=("attack",))

    # 2) 각 폴더에서 학습 2000, 테스트 500씩 수집
    per_folder_train = 2000
    per_folder_test  = 500
    benign_train_paths, benign_test_paths = [], []

    for folder in benign_folders:
        all_paths = list_npy_in_folder(DATA_DIR, folder)
        if len(all_paths) == 0:
            print(f"[WARN] {folder}: no .npy files found. Skipped.")
            continue
        tr, te = take_train_test_from_folder(
            all_paths, n_train=per_folder_train, n_test=per_folder_test
        )
        print(f"[INFO] {folder}: train {len(tr)}, test {len(te)}")
        benign_train_paths.extend(tr)
        benign_test_paths.extend(te)

    total_train = len(benign_train_paths)
    total_test  = len(benign_test_paths)
    print(f"[INFO] Total benign -> train: {total_train}, test: {total_test}")

    if total_train == 0 or total_test == 0:
        raise RuntimeError("Not enough benign data to proceed. Check your folder structure and files.")

    # 3) Encoder 학습 (benign train 전부 사용)
    model = train_encoder(benign_train_paths)

    # 4) OCSVM 학습용 피처
    benign_train_feats = extract_features(benign_train_paths, model)
    ocsvm = OneClassSVM(kernel='rbf', gamma=10.0, nu=0.1)
    ocsvm.fit(benign_train_feats)

    # 5) 공격 테스트는 benign 테스트 총 개수에 맞춤
    attack_test_paths = load_attack_paths(DATA_DIR, limit=total_test)

    # 6) 테스트 피처 추출 및 평가
    benign_test_feats = extract_features(benign_test_paths, model)
    attack_test_feats = extract_features(attack_test_paths, model)

    X = np.vstack([benign_test_feats, attack_test_feats])
    y = np.array([0]*len(benign_test_feats) + [1]*len(attack_test_feats))

    scores = ocsvm.decision_function(X)
    preds = (scores < 0).astype(int)  # 음수 = 이상 탐지

    print("\n=== OCSVM Evaluation (Default Threshold: 0) ===")
    print(classification_report(y, preds, target_names=["Benign", "Attack"]))
    auc = roc_auc_score(y, -scores)
    print(f"ROC AUC Score: {auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y, -scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1_scores)) if len(f1_scores) > 0 else 0
    best_thresh = thresholds[best_idx] if len(thresholds) > 0 else 0.0
    print(f"Best Threshold (F1): {best_thresh:.4f}")

    preds_best = (scores < -best_thresh).astype(int)
    print("\n=== OCSVM Evaluation (Best F1 Threshold) ===")
    print(classification_report(y, preds_best, target_names=["Benign", "Attack"]))
    print(f"ROC AUC Score (unchanged by threshold): {roc_auc_score(y, -scores):.4f}")

if __name__ == "__main__":
    main()
