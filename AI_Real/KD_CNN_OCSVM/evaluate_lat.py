import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import time

import joblib

# os.sched_setaffinity(0, {0, 1, 2})
# os.environ["OMP_NUM_THREADS"] = "3"
# os.environ["MKL_NUM_THREADS"] = "3"
# torch.set_num_threads(3)
os.sched_setaffinity(0, {0})
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# === Config ===
DEVICE = torch.device("cpu")  # 강제로 CPU
DATA_DIR = "./Data_CIC/Session_Windows_15"
H, W = 34, 44
FEAT_DIM = 128

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


class Encoder_4x128(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 34x44 → 17x22

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

    def forward(self, x):
        return self.net(x)

class StudentEncoder_2x64(nn.Module):
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

# === Load paths ===
def load_paths(is_attack=False, limit=100):
    pattern = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
    return sorted(glob(os.path.join(DATA_DIR, pattern)))[:limit]

# === Feature + OCSVM inference 시간 측정 ===
def evaluate_with_timing(paths, model, ocsvm):
    model.eval()
    total_time = 0.0
    inference_time_list = []
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad(): 
        for p in paths:
            x = np.load(p)[0]
            start = time.time()

            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            if x.shape[0] < H * W:
                x = np.pad(x, (0, H * W - x.shape[0]))
            img = x[:H * W].reshape(1, H, W)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            feat = model(img).detach().cpu().numpy()
            score = ocsvm.decision_function(feat)[0]
            pred = 1 if score < 0 else 0
            elapsed = time.time() - start

            inference_time_list.append(elapsed)
            total_time += elapsed

    avg_time = total_time / len(paths)
    std_time = np.std(inference_time_list)

    print(f"\n[Analysis Timing]")
    print(f"Total samples: {len(paths)}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Avg time per image: {avg_time * 1000:.3f} ms")
    print(f"Std of time per image: {std_time * 1000:.3f} ms")

# === Main ===
def main():
    # 1. Load student encoder
    model = Encoder().to(DEVICE)
    # model.load_state_dict(torch.load("./AI_Real/KD_CNN_OCSVM/Model/student_encoder_kd_20_e20_2x8.pth", map_location=DEVICE))
    model.load_state_dict(torch.load(f'./AI_Real/CNN_OCSVM/Model/cnn_deep_contrastive_encoder_0.1_1_v2.pth'))
    model.eval()

    # 2. Fit OCSVM
    ocsvm = OneClassSVM(kernel='rbf', gamma=1e6, nu=0.1)
    ocsvm = joblib.load('./AI_Real/KD_CNN_OCSVM/Model/ocsvm.pkl')


    # train_paths = load_paths(False, limit=10000)
    # feats = []
    # with torch.no_grad():
    #     for p in train_paths:
    #         x = np.load(p)[0]
    #         x = np.nan_to_num(x)
    #         x = np.clip(x, 0, 255).astype(np.float32) / 255.0
    #         if x.shape[0] < H * W:
    #             x = np.pad(x, (0, H * W - x.shape[0]))
    #         img = x[:H * W].reshape(1, H, W)
    #         img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    #         feat = model(img).detach().cpu().numpy()
    #         feats.append(feat[0])
    # ocsvm.fit(np.array(feats))

    # 3. Evaluate with timing
    test_paths = load_paths(False, 5000) + load_paths(True, 5000)
    evaluate_with_timing(test_paths, model, ocsvm)

if __name__ == "__main__":
    main()
