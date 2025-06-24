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

# === Teacher Encoder ===
class Encoder(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d((8, 2)),  # Reduce spatial size
            nn.Flatten(),
            nn.Linear(32 * 8 * 2, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class Encoderv2(nn.Module):
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
class StudentEncoder1x8(nn.Module):
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

# === KD Loss ===
def kd_loss(student_feats, teacher_feats):
    return F.mse_loss(student_feats, teacher_feats)

# === Util ===
def load_paths(data_dir, is_attack=False, limit=10000):
    pattern = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
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

# === Train Student ===
def train_student_encoder(teacher_model, train_paths):
    student = StudentEncoder1x8().to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    dataset = ContrastiveDataset(train_paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    teacher_model.eval()
    for epoch in range(EPOCHS):
        student.train(); total_loss = 0
        for x1, _ in loader:
            x1 = x1.to(DEVICE)
            with torch.no_grad():
                teacher_feat = teacher_model(x1)
            student_feat = student(x1)
            loss = kd_loss(student_feat, teacher_feat)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"[Student {epoch+1}/{EPOCHS}] KD Loss: {total_loss / len(loader):.4f}")

    torch.save(student.state_dict(), f'./AI_Real_v2/Model/student_encoder_kd_{EPOCHS}_1x8.pth')
    return student

# === Main ===
def main():
    teacher = Encoder().to(DEVICE)
    teacher.load_state_dict(torch.load(f'./AI_Real_v2/Model/cnn_contrastive_encoder_20.pth'))

    train_paths = load_paths(DATA_DIR, is_attack=False, limit=10000)
    student = train_student_encoder(teacher, train_paths)
    student.eval()

    benign_train = load_paths(DATA_DIR, False, 10000)
    benign_feats = extract_features(benign_train, student)

    gamma = 0.1
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
    preds = (scores < 0).astype(int)

    print("\n=== Student OCSVM Evaluation ===")
    precision, recall, thresholds = precision_recall_curve(y, -scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"Best Threshold (F1): {best_thresh:.4f}")
    preds = (scores < -best_thresh).astype(int)
    print(classification_report(y, preds, target_names=["Benign", "Attack"]))
    print(f"ROC AUC Score with Best Threshold: {roc_auc_score(y, -scores):.4f}")

if __name__ == "__main__":
    main()
