import os, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from glob import glob

# === Config ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './Data_CIC/Session_Windows_15'  # should contain [T, 1, H, W] .npy files
T, H, W = 15, 34, 44
BATCH_SIZE = 2**6
EPOCHS = 15

# === Dataset ===
class SequenceDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.HW = H * W

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])  # [T, F]
        if x.ndim != 2:
            raise ValueError(f"Expected [T, F], got {x.shape}.")
        x = np.nan_to_num(x)
        x = np.clip(x, 0, 255).astype(np.float32) / 255.0  # normalize
        padded = np.zeros((T, self.HW), dtype=np.float32)
        padded[:, :min(self.HW, x.shape[1])] = x[:, :min(self.HW, x.shape[1])]
        padded = padded.reshape(T, 1, H, W)
        return torch.tensor(padded, dtype=torch.float32)

# === ConvLSTM Cell ===
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        i, f, o, g = torch.chunk(conv_out, 4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# === ConvLSTM AutoEncoder ===
class ConvLSTMAE(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.encoder_cell = ConvLSTMCell(input_dim, hidden_dim)
        self.decoder_cell = ConvLSTMCell(input_dim, hidden_dim)
        self.reconstruct = nn.Conv2d(hidden_dim, input_dim, 3, padding=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        h, c = torch.zeros(B, 32, H, W).to(x.device), torch.zeros(B, 32, H, W).to(x.device)

        # Encoder: last hidden state summarizes sequence
        for t in range(T):
            h, c = self.encoder_cell(x[:, t], h, c)

        # Decoder: reconstruct sequence
        outputs = []
        dec_input = torch.zeros_like(x[:, 0])
        for _ in range(T):
            h, c = self.decoder_cell(dec_input, h, c)
            out = self.reconstruct(h)
            outputs.append(out.unsqueeze(1))
            dec_input = out.detach()
        return torch.cat(outputs, dim=1)

# === Utility ===
def load_paths(is_attack=False, limit=1000):
    pattern = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
    return sorted(glob(os.path.join(DATA_DIR, pattern)))[:limit]

def reconstruction_errors(model, paths):
    model.eval()
    errors = []
    with torch.no_grad():
        for f in paths:
            x = np.load(f)  # shape: [T, F]
            if x.ndim != 2:
                raise ValueError(f"Expected [T, F], got {x.shape} at {f}")
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0

            # Padding and reshape to [1, T, 1, H, W]
            padded = np.zeros((T, H * W), dtype=np.float32)
            padded[:, :min(H * W, x.shape[1])] = x[:, :min(H * W, x.shape[1])]
            padded = padded.reshape(1, T, 1, H, W)

            x_tensor = torch.tensor(padded, dtype=torch.float32).to(DEVICE)
            recon = model(x_tensor)
            loss = nn.functional.mse_loss(recon, x_tensor).item()
            errors.append(loss)
    return np.array(errors)


# === Train & Evaluate ===
def train_and_evaluate():
    train_paths = load_paths(is_attack=False, limit=2000)
    test_benign_paths = load_paths(is_attack=False, limit=500)
    test_attack_paths = load_paths(is_attack=True, limit=500)

    train_loader = DataLoader(SequenceDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True)
    model = ConvLSTMAE().to(DEVICE)
    # model.load_state_dict(torch.load('./APCC/Model/convlstm_ae.pth', map_location=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x in train_loader:
            x = x.to(DEVICE)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), './APCC/Model/convlstm_ae.pth')

    # Inference
    benign_errors = reconstruction_errors(model, test_benign_paths)
    attack_errors = reconstruction_errors(model, test_attack_paths)

    all_errors = np.concatenate([benign_errors, attack_errors])
    labels = np.array([0]*len(benign_errors) + [1]*len(attack_errors))

    # Optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(labels, all_errors)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    preds = (all_errors > best_threshold).astype(int)

    print("\n=== ConvLSTM AE Anomaly Detection ===")
    print(f"Best Threshold: {best_threshold:.6f}")
    print(classification_report(labels, preds, target_names=["Benign", "Attack"], digits=4))
    print(f"ROC AUC: {roc_auc_score(labels, all_errors):.4f}")
    print(f"Best F1 Score: {f1_scores[best_idx]:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
