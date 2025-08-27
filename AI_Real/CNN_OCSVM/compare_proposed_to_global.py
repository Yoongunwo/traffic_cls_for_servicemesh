import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import random

# =====================
# Config
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_k8s/Session_Windows_15"
H, W = 34, 44
FEAT_DIM = 128
BATCH_SIZE = 2**11
SEED = 42

# 폴더별 샘플 수
PER_FOLDER_TRAIN = 2000
PER_FOLDER_TEST  = 1000

# =====================
# Reproducibility
# =====================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =====================
# Encoders (예시 아키텍처)
# =====================
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
    def forward(self, x): return self.net(x)

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

# =====================
# Utils: 데이터 로딩 (멀티 폴더)
# =====================
def list_benign_folders(data_dir, exclude=("attack",)):
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
    return sorted(glob(os.path.join(data_dir, folder, "*.npy")))

def take_train_test_from_folder(paths, n_train=2000, n_test=500):
    if len(paths) < n_train + n_test:
        folder_name = os.path.basename(os.path.dirname(paths[0])) if paths else "folder"
        print(f"[WARN] {folder_name}: requested {n_train+n_test}, but found {len(paths)} files.")
    train = paths[:min(n_train, len(paths))]
    remain = paths[len(train):]
    test  = remain[:min(n_test, len(remain))]
    return train, test

def load_attack_paths(data_dir, limit):
    pattern = os.path.join(data_dir, "attack", "*", "*.npy")
    paths = sorted(glob(pattern))[:limit]
    if len(paths) < limit:
        print(f"[WARN] attack: requested {limit} but found {len(paths)} files.")
    return paths

def collect_paths_for_folders(data_dir, folders, per_train, per_test):
    """특정 benign 폴더 리스트에서 학습/테스트 경로 모으기"""
    train_paths, test_paths = [], []
    for folder in folders:
        all_paths = list_npy_in_folder(data_dir, folder)
        if len(all_paths) == 0:
            print(f"[WARN] {folder}: no .npy files. Skipped.")
            continue
        tr, te = take_train_test_from_folder(all_paths, per_train, per_test)
        print(f"[INFO] {folder}: train {len(tr)}, test {len(te)}")
        train_paths.extend(tr)
        test_paths.extend(te)
    return train_paths, test_paths

# =====================
# Feature 추출
# =====================
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

# =====================
# OCSVM 하이퍼파라미터 자동 추정 (옵션)
# =====================
def estimate_gamma_rbf(X, max_samples=2000, seed=SEED):
    """
    RBF gamma ~ 1 / (2 * median_distance^2)
    과한 메모리 방지를 위해 일부 샘플만 사용
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    if n == 0:
        return 1.0
    idx = rng.choice(n, size=min(n, max_samples), replace=False)
    S = X[idx]
    # 중심 대비 거리의 중앙값 (pairwise보다 가벼움)
    mu = S.mean(axis=0, keepdims=True)
    d = np.linalg.norm(S - mu, axis=1)
    med = np.median(d)
    if med <= 1e-12:
        return 1.0
    gamma = 1.0 / (2.0 * (med ** 2))
    return float(gamma)

# =====================
# PR 곡선 (모델별 y/scores 각각 받기)
# =====================
def plot_pr_multi_models(results, title="Precision–Recall Curves (vs. Global Model)",
                         save_path=None, downsample=1, baseline_name="Global", palette=None):
    """
    results: [{"name": str, "y": array, "scores": array}, ...]
    baseline_name: 베이스(예: 'Global')로 강조할 모델 이름
    palette: None이면 Tableau 10(색맹 친화) 기본 팔레트 사용
    """
    # 색맹 친화 + 논문 친화 팔레트 (Tableau 10)
    if palette is None:
        palette = [
            "#E15759", "#4E79A7", "#F28E2B", "#76B7B2", "#59A14F",
            "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
        ]
    base_color = "#4D4D4D"  # Global용 베이스 컬러(짙은 그레이)

    # Global 먼저 그리도록 정렬
    # results_sorted = sorted(results, key=lambda r: (r["name"] != baseline_name, r["name"]))
    fig, ax = plt.subplots(figsize=(7, 3.5))

    color_idx = 0
    for r in results:
        name, y, scores = r["name"], r["y"], r["scores"]
        precision, recall, _ = precision_recall_curve(y, -scores)  # 점수 클수록 정상 → -scores
        # ap = average_precision_score(y, -scores)  # 범례에 AUPRC 표시하고 싶으면 사용

        if downsample > 1:
            precision = precision[::downsample]; recall = recall[::downsample]

        if name == baseline_name:
            ax.plot(recall, precision, label=f"{name}",
                    color=base_color, linewidth=3, zorder=3)
        else:
            c = palette[color_idx % len(palette)]
            color_idx += 1
            ax.plot(recall, precision, label=f"{name}",
                    color=c, linewidth=2, alpha=0.95, zorder=2)

    # ax.set_xlabel("Recall", fontsize=12)
    # ax.set_ylabel("Precision", fontsize=12)
    # ax.set_title(title, fontsize=14)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.4)

    ax.legend(loc="lower left")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()

# =====================
# 모델 등록: (라벨, 생성자, 체크포인트, 사용할 benign 폴더, OCSVM 설정(옵션))
# =====================
# ★ folders 값:
#   - "ALL"  : attack 제외한 모든 benign 폴더 사용
#   - ["folderA", "folderB"] : 지정 폴더만 사용
MODEL_SPECS = [
    {
        "name": "Global",
        "builder": lambda: Encoder(out_dim=128),
        "ckpt": "./AI_Real/CNN_OCSVM/Model/global/cnn_contrastive_encoder_k8s_0.1_20.pth",
        "folders": "ALL",
        "ocsvm": None,  # None이면 gamma 자동추정, nu=0.01, kernel='rbf'
    },
    {
        "name": "React+Nginx",
        "builder": lambda: Encoder(out_dim=128),
        "ckpt": "./AI_Real/CNN_OCSVM/Model/specific/cnn_contrastive_encoder_k8s_0.1_10.pth",
        "folders": ["save_front"],   # ← 실제 benign 폴더명으로 바꿔주세요
        # "ocsvm": {"kernel": "rbf", "nu": 0.01, "gamma": 10**4},  # 예시: 모델별 별도 설정
        "ocsvm": None,
    },
    {
        "name": "FastAPI",
        "builder": lambda: Encoder(out_dim=128),
        "ckpt": "./AI_Real/CNN_OCSVM/Model/others/cnn_contrastive_encoder_k8s_0.1_20_back.pth",
        "folders": ["save_back"],          # ← 실제 benign 폴더명
        "ocsvm": None,
    },
    {
        "name": "PostgreSQL",
        "builder": lambda: Encoder(out_dim=128),
        "ckpt": "./AI_Real/CNN_OCSVM/Model/others/cnn_contrastive_encoder_k8s_0.1_20_postgres.pth",
        "folders": ["postgres"],         # ← 실제 benign 폴더명
        "ocsvm": {"kernel": "rbf", "nu": 0.02},  # gamma 자동추정 사용 가능(명시 안하면 자동)
    },
    {
        "name": "pgAdmin",
        "builder": lambda: Encoder(out_dim=128),
        "ckpt": "./AI_Real/CNN_OCSVM/Model/others/cnn_contrastive_encoder_k8s_0.1_10_pgadmin.pth",
        "folders": ["pgadmin"],          # ← 실제 benign 폴더명
        "ocsvm": None,
    },
]

# =====================
# Main
# =====================
def main():
    # 0) benign 폴더 전체 리스트(ALL 처리를 위해)
    all_benign_folders = list_benign_folders(DATA_DIR, exclude=("attack",))

    # 1) 모델별로 전용 benign 데이터만 수집 → OCSVM 학습 → 전용 테스트 + 공격 테스트 평가
    all_results = []
    for spec in MODEL_SPECS:
        name, builder, ckpt = spec["name"], spec["builder"], spec["ckpt"]
        folders = spec["folders"]
        ocsvm_cfg = spec.get("ocsvm", None)

        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"[{name}] checkpoint not found: {ckpt}")

        # 폴더 결정
        if folders == "ALL":
            use_folders = all_benign_folders
        elif isinstance(folders, (list, tuple)):
            use_folders = folders
        else:
            raise ValueError(f"[{name}] 'folders' must be 'ALL' or list of folder names.")

        print(f"\n[MODEL] {name}")
        print(f"[INFO] Using benign folders: {use_folders}")

        # 해당 모델 전용 benign train/test 경로
        benign_train_paths, benign_test_paths = collect_paths_for_folders(
            DATA_DIR, use_folders, PER_FOLDER_TRAIN, PER_FOLDER_TEST
        )
        if len(benign_train_paths) == 0 or len(benign_test_paths) == 0:
            raise RuntimeError(f"[{name}] Not enough benign data in specified folders: {use_folders}")

        # 공격 테스트는 해당 모델 benign 테스트 개수에 맞춤
        attack_test_paths = load_attack_paths(DATA_DIR, limit=len(benign_test_paths))

        # 모델 로드
        model = builder().to(DEVICE)
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        # feature 추출
        print("[INFO] Extract features (train, benign) ...")
        benign_train_feats = extract_features(benign_train_paths, model)

        # OCSVM 설정 (모델별)
        if ocsvm_cfg is None:
            # 자동 추정: gamma
            gamma_auto = estimate_gamma_rbf(benign_train_feats, max_samples=2000, seed=SEED)
            ocsvm_kernel = "rbf"
            ocsvm_nu = 0.01
            ocsvm_gamma = gamma_auto
            print(f"[INFO] OCSVM(auto): kernel={ocsvm_kernel}, nu={ocsvm_nu}, gamma≈{ocsvm_gamma:.4g}")
        else:
            ocsvm_kernel = ocsvm_cfg.get("kernel", "rbf")
            ocsvm_nu = ocsvm_cfg.get("nu", 0.01)
            # gamma 미지정이면 자동 추정
            if "gamma" in ocsvm_cfg and ocsvm_cfg["gamma"] is not None:
                ocsvm_gamma = ocsvm_cfg["gamma"]
                print(f"[INFO] OCSVM(cfg): kernel={ocsvm_kernel}, nu={ocsvm_nu}, gamma={ocsvm_gamma}")
            else:
                ocsvm_gamma = estimate_gamma_rbf(benign_train_feats, max_samples=2000, seed=SEED)
                print(f"[INFO] OCSVM(cfg + auto gamma): kernel={ocsvm_kernel}, nu={ocsvm_nu}, gamma≈{ocsvm_gamma:.4g}")

        ocsvm = OneClassSVM(kernel=ocsvm_kernel, gamma=ocsvm_gamma, nu=ocsvm_nu)
        ocsvm.fit(benign_train_feats)

        print("[INFO] Extract features (test, benign/attack) ...")
        benign_test_feats = extract_features(benign_test_paths, model)
        attack_test_feats = extract_features(attack_test_paths, model)

        X = np.vstack([benign_test_feats, attack_test_feats])
        y = np.array([0]*len(benign_test_feats) + [1]*len(attack_test_feats))

        scores = ocsvm.decision_function(X)  # 값이 클수록 정상

        # 참고용 출력
        preds0 = (scores < 0).astype(int)
        print("[Eval @ thr=0]")
        print(classification_report(y, preds0, target_names=["Benign","Attack"], digits=4))
        print("AUROC:", roc_auc_score(y, -scores))

        all_results.append({"name": name, "y": y, "scores": scores})

    # 2) PR 곡선 한 장에 그리기 (모델별 y/scores 각각 사용)
    plot_pr_multi_models(all_results, title="Precision–Recall Curves (vs. Global Model)", save_path="pr_5models.png")

if __name__ == "__main__":
    main()
