import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_CIC/Session_Windows_15"
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

# === Teacher Encoder ===
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

# === Student Encoder ===
class StudentEncoder(nn.Module):
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
    student = StudentEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    dataset = ContrastiveDataset(train_paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    teacher_model.eval()
    for epoch in range(20):
        student.train(); total_loss = 0
        for x1, _ in loader:
            x1 = x1.to(DEVICE)
            with torch.no_grad():
                teacher_feat = teacher_model(x1)
            student_feat = student(x1)
            loss = kd_loss(student_feat, teacher_feat)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"[Student {epoch+1}/{20}] KD Loss: {total_loss / len(loader):.4f}")

    torch.save(student.state_dict(), f'./AI_Real/KD_CNN_OCSVM/Model/student_encoder_kd_{EPOCHS}_e20_2x16.pth')
    return student

# === Main ===
def main():
    teacher = Encoder().to(DEVICE)
    teacher.load_state_dict(torch.load(f'./AI_Real/CNN_OCSVM/Model/cnn_contrastive_encoder_0.1_{EPOCHS}.pth'))

    train_paths = load_paths(DATA_DIR, is_attack=False, limit=10000)
    student = train_student_encoder(teacher, train_paths)
    student.eval()
    benign_train = load_paths(DATA_DIR, False, 10000)
    benign_feats = extract_features(benign_train, student)


    gamma = 10.0
    nu = 0.1
    kernel = 'rbf'
    print(f"Epochs: {EPOCHS}")
    print(f"Training OCSVM with gamma={gamma}, nu={nu}, kernel={kernel}")
    ocsvm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    ocsvm.fit(benign_feats)
    
    benign_test = load_paths(DATA_DIR, False, 12500)
    attack_test = load_paths(DATA_DIR, True, 12500)

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
