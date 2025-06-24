import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_CIC/Session_Windows_15"
BATCH_SIZE = 2**9
EPOCHS = 20
FEAT_DIM = 128
TEMPERATURE = 0.1
H, W = 1479, 5  # Input shape

# === Contrastive Dataset ===
class ContrastiveDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self): return len(self.paths)

    def augment(self, x):
        noise = np.random.normal(0, 0.01, size=x.shape)
        return np.clip(x + noise, 0, 1)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])  # <-- 수정: [0] 제거
        x = x.T
        x = np.nan_to_num(x)
        x = np.clip(x, 0, 255).astype(np.float32) / 255.0
        x = x[:, :5]  # (1479, 5)

        img1 = self.augment(x).reshape(1, H, W)
        img2 = self.augment(x).reshape(1, H, W)
        return torch.tensor(img1, dtype=torch.float32), torch.tensor(img2, dtype=torch.float32)

class Encoder(nn.Module):
    def __init__(self, out_dim=128):  # FEAT_DIM과 동일하게 설정
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # ✅ 한 번만 사용 (34x44 → 17x22)

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            # ❌ MaxPool2 제거

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.AdaptiveAvgPool2d((4, 4)),  # ✅ 더 큰 feature map 보존
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.net(x)

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
    pattern = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
    return sorted(glob(os.path.join(data_dir, pattern)))[:limit]

def extract_features(paths, model):
    model.eval()
    feats = []
    with torch.no_grad():
        for path in paths:
            x = np.load(path)  # (1479, 15)
            x = x.T
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            x = x[:, :5]  # (1479, 5)

            img = torch.tensor(x.reshape(1, H, W), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z = model(img).squeeze(0).cpu().numpy()
            feats.append(z)
    return np.array(feats)

def print_model_flops(model, input_res=(1, 34, 44)):
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model, 
            input_res,
            as_strings=False,
            print_per_layer_stat=True,
            verbose=True
        )
        print(f"\n[Total] FLOPs: {macs / 1e6:.5f} MFLOPs, Params: {params / 1e6:.5f} M")


# === Main Pipeline ===
def main():
    # model = Encoder().to(DEVICE)
    # model.load_state_dict(
    #     torch.load(f'./AI_Real_v2/Model/cnn_deep_contrastive_encoder_{EPOCHS}.pth') 
    # )
    # model.eval()
    model = StudentEncoder_1x8().to(DEVICE)
    model.load_state_dict(
        torch.load(f'./AI_Real_v2/Model/student_encoder_kd_20_1x8.pth') 
    )
    model.eval()

    # print_model_flops(model)

    benign_train = load_paths(DATA_DIR, is_attack=False, limit=10000)
    benign_feats = extract_features(benign_train, model)

    gamma = 1000
    nu = 0.01
    kernel = 'rbf'
    print(f"Epochs: {EPOCHS}")
    print(f"Training OCSVM with gamma={gamma}, nu={nu}, kernel={kernel}")
    ocsvm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    ocsvm.fit(benign_feats)

    benign_test = load_paths(DATA_DIR, is_attack=False, limit=12500)
    attack_test = load_paths(DATA_DIR, is_attack=True, limit=12500)
    benign_feats = extract_features(benign_test, model)
    attack_feats = extract_features(attack_test, model)

    X = np.vstack([benign_feats, attack_feats])
    y = np.array([0]*len(benign_feats) + [1]*len(attack_feats))
    scores = ocsvm.decision_function(X)
    preds = (scores < 0).astype(int)  # 음수 = 이상 탐지

    print("\n=== OCSVM Evaluation ===")
    
    precision, recall, thresholds = precision_recall_curve(y, -scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"Best Threshold (F1): {best_thresh:.4f}")
    preds = (scores < -best_thresh).astype(int)
    print(classification_report(y, preds, target_names=["Benign", "Attack"], digits=4))
    print(f"ROC AUC Score with Best Threshold: {roc_auc_score(y, -scores):.4f}")

if __name__ == "__main__":
    main()
