import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score

current_dir = os.getcwd() 
sys.path.append(current_dir)

from AI.Model.CNN import train_v2 as cnn_train
from AI.Model.CAE import model as cae_model
from AI.Model.CAE import evaluate as cae_evaluate
from AI import train as com_train


@torch.no_grad()
def calculate_threshold(model, loader, device='cpu', percentile=95):
    model.eval()
    reconstruction_errors = []

    for x, _ in loader:
        x = x.to(device)
        outputs = model(x)
        errors = F.mse_loss(outputs, x, reduction='none')
        errors = errors.view(errors.size(0), -1).mean(dim=1)
        reconstruction_errors.extend(errors.cpu().numpy())

    threshold = np.percentile(reconstruction_errors, percentile)
    print(f"📌 Calculated threshold (P{percentile}): {threshold:.6f}")
    return threshold

@torch.no_grad()
def evaluate_model(model, loader, device, threshold):
    model.eval()
    all_errors = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        outputs = model(x)
        errors = F.mse_loss(outputs, x, reduction='none')
        errors = errors.view(errors.size(0), -1).mean(dim=1)
        all_errors.extend(errors.cpu().numpy())
        all_labels.extend(y.numpy())

    errors = np.array(all_errors)
    labels = np.array(all_labels)
    preds = (errors > threshold).astype(int)

    print("\n📊 Classification Report:")
    print(classification_report(labels, preds, digits=4))
    print(f"ROC AUC: {roc_auc_score(labels, errors):.4f}")

def visualize_reconstruction_distribution(model, loader, device):
    model.eval()
    errors = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            loss = F.mse_loss(out, x, reduction='none')
            loss = loss.view(loss.size(0), -1).mean(dim=1)
            errors.extend(loss.cpu().numpy())
            labels.extend(y.numpy())

    errors = np.array(errors)
    labels = np.array(labels)

    plt.figure(figsize=(10, 6))
    plt.hist(errors[labels == 0], bins=100, alpha=0.6, label='Benign')
    plt.hist(errors[labels == 1], bins=100, alpha=0.6, label='Attack')
    plt.legend()
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

TEST = 'cic'
PREPROCESSING_TYPE = 'row'
METHOD = '_window4'
BATCH_SIZE = 4096 * 4
SIZE = 32

TEST_DATASET = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq/benign_train'
ATTACK_DATASET = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq/attack'

CAE_MODEL_PATH = f'./AI/Model/CAE/Model/{TEST}_{PREPROCESSING_TYPE}_autoencoder{METHOD}.pth'
CAE_THRESHOLD_PATH = f'./AI/Model/CAE/Model/{TEST}_{PREPROCESSING_TYPE}_autoencoder_threshold{METHOD}.npy'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor()
])

normal_test = cnn_train.PacketImageDataset(TEST_DATASET, transform, is_flat_structure=True, label=0)
attack_test = cnn_train.PacketImageDataset(ATTACK_DATASET, transform, is_flat_structure=True, label=1)

if SIZE == 32:
    # 원래 Dataset 객체에서 경로와 라벨 추출
    original_dataset = normal_test.dataset if isinstance(normal_test, Subset) else normal_test
    selected_indices = normal_test.indices if isinstance(normal_test, Subset) else list(range(len(normal_test)))
    image_paths = [original_dataset.images[i] for i in selected_indices]
    labels = [original_dataset.labels[i] for i in selected_indices]
    normal_test = com_train.SlidingConcatDataset(image_paths, labels, transform=transform)

    original_dataset = attack_test.dataset if isinstance(attack_test, Subset) else attack_test
    selected_indices = attack_test.indices if isinstance(attack_test, Subset) else list(range(len(attack_test)))
    image_paths = [original_dataset.images[i] for i in selected_indices]
    labels = [original_dataset.labels[i] for i in selected_indices]
    attack_test = com_train.SlidingConcatDataset(image_paths, labels, transform=transform)

min_len = min(len(normal_test), len(attack_test))
test_dataset = torch.utils.data.ConcatDataset([
    Subset(normal_test, list(range(min_len))),
    Subset(attack_test, list(range(min_len)))
])

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

cae = cae_model.ConvAutoencoder32() if SIZE == 32 else cae_model.ConvAutoencoder()
cae.load_state_dict(torch.load(CAE_MODEL_PATH, weights_only=True))
cae.eval()
cae.to(device)

visualize_reconstruction_distribution(cae, test_loader, device=device)