import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from ptflops import get_model_complexity_info

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_k8s/Session_Windows_15"
H, W = 34, 44
BATCH_SIZE = 2**11
EPOCHS = 20
FEAT_DIM = 128
TEMPERATURE = 0.1

# === Dataset ===
class ContrastiveDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.HW = H * W

    def __len__(self): return len(self.paths)

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

# === Student Encoder ===
class StudentEncoder_2x32(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128), nn.ReLU(),
            nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class StudentEncoder_2x16(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),  # ↓ 채널 수 감소
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),  # ↓ Feature map 크기도 축소
            nn.Linear(16 * 2 * 2, 64), nn.ReLU(),        # ↓ FC 계층 축소
            nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class StudentEncoder_2x8(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),         # 첫 conv: 4채널
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(),         # 두 번째 conv: 8채널
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(8 * 2 * 2, 32), nn.ReLU(),              # FC는 축소
            nn.BatchNorm1d(32), nn.Dropout(0.3),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class StudentEncoder_1x16(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),       # 더 넓은 채널 하나
            nn.AdaptiveAvgPool2d((2, 2)),                    # 약간의 공간 정보 유지
            nn.Flatten(),                                    # [batch, 16 * 2 * 2]
            nn.Linear(16 * 2 * 2, 64), nn.ReLU(),            # FC 계층
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class StudentEncoder_1x8(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),      # 하나의 얕은 conv
            nn.AdaptiveAvgPool2d((1, 1)),                  # 글로벌 평균 풀링
            nn.Flatten(),                                  # [batch, 8]
            nn.Linear(8, out_dim)                          # 매우 얇은 FC
        )

    def forward(self, x):
        return self.net(x)

# === Util ===
def load_paths(data_dir, is_attack=False, limit=10000):
    pattern = 'attack/*/*.npy' if is_attack else 'save_front/*.npy'
    return sorted(glob(os.path.join(data_dir, pattern)))[:limit]

def extract_features(paths, model):
    model.eval(); feats = []
    with torch.no_grad():
        for f in paths:
            x = np.load(f)[0]
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            if x.shape[0] < H * W:
                x = np.pad(x, (0, H * W - x.shape[0]))
            img = x[:H * W].reshape(1, H, W)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z = model(img).squeeze(0).cpu().numpy()
            feats.append(z)
    return np.array(feats)

# === Main ===
def main():
    student = StudentEncoder_1x8().to(DEVICE)
    student.eval()
    student.load_state_dict(torch.load(f'./AI_Real/KD_CNN_OCSVM/Model/student_encoder_kd_k8s_10_1x8_v2.pth'))

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            student, 
            (1, 34, 44), 
            as_strings=True, 
            print_per_layer_stat=False,
            verbose=True
        )
        print(f'FLOPs: {macs}, Params: {params}')

    benign_train = load_paths(DATA_DIR, False, 10000)
    benign_feats = extract_features(benign_train, student)

    gamma = 1000000
    nu = 0.1
    kernel = 'rbf'
    print(f"Epochs: {EPOCHS}")
    print(f"Training OCSVM with gamma={gamma}, nu={nu}, kernel={kernel}")
    ocsvm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    ocsvm.fit(benign_feats)

    benign_test = load_paths(DATA_DIR, False, 11214)
    benign_test = benign_test[-1214:] 
    attack_test = load_paths(DATA_DIR, True, 1214)
    benign_feats = extract_features(benign_test, student)
    attack_feats = extract_features(attack_test, student)

    X = np.vstack([benign_feats, attack_feats])
    y = np.array([0]*len(benign_feats) + [1]*len(attack_feats))
    scores = ocsvm.decision_function(X)

    print("\n=== Student OCSVM Evaluation ===")

    precision, recall, thresholds = precision_recall_curve(y, -scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"Best Threshold (F1): {best_thresh:.4f}")
    preds = (scores < -best_thresh).astype(int)
    print(classification_report(y, preds, target_names=["Benign", "Attack"], digits=4))
    print(f"ROC AUC Score with Best Threshold: {roc_auc_score(y, -scores):.4f}")

if __name__ == "__main__":
    main()
