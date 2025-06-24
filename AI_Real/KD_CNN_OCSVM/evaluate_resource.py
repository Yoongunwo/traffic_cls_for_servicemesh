import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import time
import psutil

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

# === Resource Usage Function ===
def get_resource_usage(pid=None):
    if pid is None:
        pid = os.getpid()
    p = psutil.Process(pid)
    cpu = p.cpu_percent(interval=0.0)  # 즉시 반환
    mem = p.memory_info().rss / (1024 ** 2)
    return cpu, mem

# === Load paths ===
def load_paths(is_attack=False, limit=100):
    pattern = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
    return sorted(glob(os.path.join(DATA_DIR, pattern)))[:limit]

def evaluate_with_timing(paths, model, ocsvm):
    model.eval()
    inference_time_list = []

    # 현재 프로세스
    p = psutil.Process(os.getpid())
    
    # 초기화 및 시작 측정
    _ = p.cpu_percent(interval=None)  # CPU 사용률 초기화
    mem_before = p.memory_info().rss / (1024 ** 2)  # MB
    start_time = time.time()

    with torch.no_grad():
        for pth in paths:
            x = np.load(pth)[0]
            start = time.time()

            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            if x.shape[0] < H * W:
                x = np.pad(x, (0, H * W - x.shape[0]))
            img = x[:H * W].reshape(1, H, W)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            feat = model(img).detach().cpu().numpy()
            score = ocsvm.decision_function(feat)[0]
            _ = 1 if score < 0 else 0

            inference_time_list.append(time.time() - start)

    cpu_percent = p.cpu_percent(interval=None)
    mem_after = p.memory_info().rss / (1024 ** 2)

    # 전체 측정 종료
    total_time = sum(inference_time_list)
    avg_latency = np.mean(inference_time_list)
    avg_mem_usage = (mem_before + mem_after) / 2
    total_cpu_time = (cpu_percent / 100.0) * total_time

    # 출력
    print(f"[Analysis Timing]")
    print(f"Total samples: {len(paths)}")
    print(f"Total elapsed time: {total_time:.3f} s")
    print(f"Avg of Latency per image: {avg_latency * 1000:.3f} ms")
    print(f"Std of Latency per image: {np.std(inference_time_list) * 1000:.3f} ms\n")

    print(f"Avg CPU usage: {cpu_percent:.3f}%")
    print(f"→ Total CPU time: {total_cpu_time:.3f} s")
    print(f"Avg memory usage: {avg_mem_usage:.3f} MB\n")

model_paths = [
    "./AI_Real/CNN_OCSVM/Model/cnn_deep_contrastive_encoder_0.1_10_4x128.pth",
    "./AI_Real/KD_CNN_OCSVM/Model/student_encoder_kd_20_e20_2x32.pth",
    "./AI_Real/KD_CNN_OCSVM/Model/student_encoder_kd_20_e20_2x16.pth",
    "./AI_Real/KD_CNN_OCSVM/Model/student_encoder_kd_20_e20_2x8.pth",
    "./AI_Real/KD_CNN_OCSVM/Model/student_encoder_kd_20_e20_1x16.pth",
    "./AI_Real/KD_CNN_OCSVM/Model/student_encoder_kd_20_e20_1x8.pth"
]

model_classes = [
    Encoder_4x128,
    StudentEncoder_2x32,
    StudentEncoder_2x16,
    StudentEncoder_2x8,
    StudentEncoder_1x16,
    StudentEncoder_1x8
]

# === Main ===
def main():
    test_paths = load_paths(False, 5000) + load_paths(True, 5000)
    ocsvm = joblib.load('./AI_Real/KD_CNN_OCSVM/Model/ocsvm.pkl')

    for path, model_cls in zip(model_paths, model_classes):
        print(f"Loading model from {path}...")
        model_ins = model_cls().to(DEVICE)
        model_ins.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model_ins.eval()

        evaluate_with_timing(test_paths, model_ins, ocsvm)

if __name__ == "__main__":
    main()
