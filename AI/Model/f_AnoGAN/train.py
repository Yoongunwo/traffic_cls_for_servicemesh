# f_AnoGAN 전체 학습 파이프라인
# - Generator (G): 정상 이미지 생성
# - Discriminator (D): 진짜 vs 가짜 이미지 판별
# - Encoder (E): 이미지 → latent vector 추정 (inference 속도 개선)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision.datasets.folder import default_loader
from sklearn.metrics import classification_report, roc_auc_score
import os
import sys
import numpy as np

current_dir = os.getcwd()
sys.path.append(current_dir)

from AI.Model.CNN import train_v2 as cnn_train

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 2 * 2),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, latent_dim=100):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, latent_dim)
        )

    def forward(self, x):
        return self.model(x)

def compute_anomaly_score(x, x_gen, z, z_hat, lambda_=0.1):
    recon_loss = F.l1_loss(x, x_gen)
    latent_loss = F.mse_loss(z, z_hat)
    return recon_loss + lambda_ * latent_loss

def train_gan(generator, discriminator, dataloader, device, gen_path, disc_path, latent_dim=100, num_epochs=50):
    G, D = generator.to(device), discriminator.to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            bs = imgs.size(0)
            real = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)

            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = G(z)

            d_loss = criterion(D(imgs), real) + criterion(D(fake_imgs.detach()), fake)
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            g_loss = criterion(D(fake_imgs), real)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    torch.save(G.state_dict(), gen_path)
    torch.save(D.state_dict(), disc_path)

def train_encoder(encoder, generator, dataloader, device, encoder_path, latent_dim=100, num_epochs=50):
    E, G = encoder.to(device), generator.to(device)
    opt_E = torch.optim.Adam(E.parameters(), lr=0.0002)
    criterion = nn.MSELoss()
    G.eval()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            z_hat = E(imgs)
            recons = G(z_hat)
            loss = criterion(recons, imgs)
            opt_E.zero_grad(); loss.backward(); opt_E.step()
            total_loss += loss.item()

        print(f"[Encoder] Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.6f}")

    torch.save(E.state_dict(), encoder_path)

def evaluate(encoder, generator, dataloader, device):
    encoder.eval()
    generator.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            z_hat = encoder(imgs)
            x_gen = generator(z_hat)
            scores = F.l1_loss(imgs, x_gen, reduction='none')
            scores = scores.view(scores.size(0), -1).mean(dim=1)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())

    preds = [1 if s > np.percentile(all_scores, 95) else 0 for s in all_scores]
    print(classification_report(all_labels, preds, digits=4))
    print(f"ROC AUC: {roc_auc_score(all_labels, all_scores):.4f}")

def calculate_threshold(generator, encoder, dataloader, device, lambda_=0.1):
    scores = []
    generator.eval()
    encoder.eval()

    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            z_hat = encoder(imgs)
            x_gen = generator(z_hat)
            score = F.l1_loss(imgs, x_gen, reduction='none')
            score = score.view(score.size(0), -1).mean(dim=1)
            scores.extend(score.cpu().numpy())

    scores = np.array(scores)
    threshold = scores.mean() + 3 * scores.std()  # 3-sigma 기준
    print(f"📌 Threshold (mean + 3std): {threshold:.6f}")
    return threshold

def save_threshold(threshold, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.array([threshold]))

def load_threshold(path):
    return float(np.load(path)[0])

def train_model(device, train_loader, epoches, model_dir, gen_path, disc_path, encoder_path, threshold_path):
    
    G = Generator()
    D = Discriminator()
    E = Encoder()

    train_gan(G, D, train_loader, device, num_epochs=epoches, 
              gen_path=os.path.join(model_dir, gen_path), disc_path=os.path.join(model_dir, disc_path))
    train_encoder(E, G, train_loader, device, num_epochs=epoches, encoder_path=os.path.join(model_dir, encoder_path))

    threshold = calculate_threshold(G, E, train_loader, device)
    save_threshold(threshold, path=os.path.join(model_dir, threshold_path))


TRAIN_DATASET = './Data/cic_data/Wednesday-workingHours/benign_train'
TEST_DATASET = './Data/cic_data/Wednesday-workingHours/benign_test'
ATTACK_DATASET = './Data/cic_data/Wednesday-workingHours/attack'

MODEL_DIR = './AI/Model/f_AnoGAN/Model'
ENCODER_MODEL_PATH =  './AI/Model/f_AnoGAN/Model/cic_encoder_epoch50.pth'
GENERATOR_MODEL_PATH = './AI/Model/f_AnoGAN/cic_generator_epoch50.pth'
DISCRIMINATOR_MODEL_PATH = './AI/Model/f_AnoGAN/cic_discriminator_epoch50.pth'
THRESHOLD_PATH = './AI/Model/f_AnoGAN/Model/cic_threshold.npy'

BATCH_SIZE = 4096 * 32

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    normal_train = cnn_train.PacketImageDataset(TRAIN_DATASET, transform, is_flat_structure=True, label=0)

    train_loader = DataLoader(normal_train, batch_size=BATCH_SIZE, shuffle=False)

    G = Generator()
    D = Discriminator()
    E = Encoder()

    train_gan(G, D, train_loader, device, num_epochs=50, gen_path=GENERATOR_MODEL_PATH, disc_path=DISCRIMINATOR_MODEL_PATH)
    train_encoder(E, G, train_loader, device, num_epochs=50, encoder_path=ENCODER_MODEL_PATH)
    
    normal_test = cnn_train.PacketImageDataset(TEST_DATASET, transform, is_flat_structure=True, label=0)
    attack_test = cnn_train.PacketImageDataset(ATTACK_DATASET, transform, is_flat_structure=True, label=1)

    min_len = min(len(normal_test), len(attack_test))

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    evaluate(E, G, test_loader, device)
    threshold = calculate_threshold(G, E, train_loader, device)
    save_threshold(threshold, path=THRESHOLD_PATH)

if __name__ == '__main__':
    main()
