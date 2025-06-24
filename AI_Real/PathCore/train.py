import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import NearestNeighbors
import faiss  # Facebook AI Similarity Search

# === Config ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './Data_CIC/Session_Windows_15'
H, W = 34, 44
PATCH_SIZE = 3
STRIDE = 1
FEAT_DIM = 128
K = 5  # for KNN
BATCH_SIZE = 2**15

# === Dataset ===
class SingleImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.HW = H * W

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])
        x = np.nan_to_num(x)
        x = np.clip(x, 0, 255).astype(np.float32) / 255.0
        vec = x[0]
        if vec.shape[0] < self.HW:
            vec = np.pad(vec, (0, self.HW - vec.shape[0]))
        img = vec[:self.HW].reshape(1, H, W)
        return torch.tensor(img, dtype=torch.float32)

# === Backbone Encoder ===
class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_dim, 3, padding=1), nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)  # [B, C, H, W]

# === Extract Patch Embeddings ===
def extract_patches(feat_map):
    B, C, Hf, Wf = feat_map.shape
    patches = F.unfold(feat_map, kernel_size=PATCH_SIZE, stride=STRIDE)  # [B, C*PS*PS, L]
    patches = patches.transpose(1, 2).contiguous()  # [B, L, C']
    return patches.view(-1, patches.size(-1))  # [B*L, C'] 

# === Load Paths ===
def load_paths(is_attack=False, limit=1000):
    pattern = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
    return sorted(glob(os.path.join(DATA_DIR, pattern)))[:limit]

# === Train PatchCore ===
def train_patchcore(model, paths, batch_size=64):
    model.eval()
    dataset = SingleImageDataset(paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    patch_bank = []
    with torch.no_grad():
        for x in loader:
            x = x.to(DEVICE)
            feat = model(x)  # [B, C, Hf, Wf]
            patches = extract_patches(feat)  # [B*L, D]
            patch_bank.append(patches.cpu())
    return torch.cat(patch_bank, dim=0).numpy()

# === Anomaly Scoring ===
def compute_scores(model, patch_bank, paths, batch_size=32):
    index = faiss.IndexFlatL2(patch_bank.shape[1])
    index.add(patch_bank)
    dataset = SingleImageDataset(paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    scores = []
    with torch.no_grad():
        for x in loader:
            x = x.to(DEVICE)
            feat = model(x)
            patches = extract_patches(feat).cpu().numpy()
            D, _ = index.search(patches, K)
            # 평균 score 계산 (배치 단위로)
            patch_per_sample = patches.shape[0] // x.size(0)
            for i in range(x.size(0)):
                d = D[i * patch_per_sample: (i+1) * patch_per_sample, 0]
                scores.append(np.mean(d))
    return np.array(scores)

# === Main ===
def main():
    model = FeatureExtractor().to(DEVICE)
    model.eval()

    benign_train = load_paths(is_attack=False, limit=1000)
    patch_bank = train_patchcore(model, benign_train)
    # Save the patch bank for later use
    np.save('./AI_Real/PatchCore/Model/patch_bank.npy', patch_bank)


    benign_test = load_paths(is_attack=False, limit=500)
    attack_test = load_paths(is_attack=True, limit=500)

    test_paths = benign_test + attack_test
    test_labels = np.array([0] * len(benign_test) + [1] * len(attack_test))
    scores = compute_scores(model, patch_bank, test_paths)

    # Threshold tuning
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(test_labels, scores)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]

    preds = (scores > best_threshold).astype(int)
    print("\n=== PatchCore Evaluation ===")
    print(classification_report(test_labels, preds, target_names=["Benign", "Attack"]))
    print(f"ROC AUC: {roc_auc_score(test_labels, scores):.4f}")
    print(f"Best F1 Score: {f1s[best_idx]:.4f} | Threshold: {best_threshold:.4f}")

if __name__ == "__main__":
    main()
