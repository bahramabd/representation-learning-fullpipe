import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict
import os
import scipy.io as sio



def extract_subject_id(filename):
    parts = filename.split("_")
    return parts[1] + "_" + parts[2] + "_" + parts[3]  


def compute_global_min_max(data_folder):
    classes = ["CN", "EMCI", "LMCI", "AD"]
    global_min = float('inf')
    global_max = float('-inf')

    for class_name in classes:
        class_path = os.path.join(data_folder, class_name)
        for file in os.listdir(class_path):
            if file.endswith(".mat"):
                file_path = os.path.join(class_path, file)
                mat_data = sio.loadmat(file_path)
                kl_matrix = mat_data[list(mat_data.keys())[-1]]
                if kl_matrix.shape == (90, 90):
                    global_min = min(global_min, kl_matrix.min())
                    global_max = max(global_max, kl_matrix.max())

    print(f"Global min: {global_min:.4f}, Global max: {global_max:.4f}")
    return global_min, global_max



class FullKLDataset(Dataset):
    def __init__(self, data_folder, global_min=None, global_max=None):
        self.data = []
        self.labels = []
        self.subject_ids = []
        self.classes = ["CN", "EMCI", "LMCI", "AD"]
        self.global_min = global_min
        self.global_max = global_max
        self.load_all_data(data_folder)

    def load_all_data(self, data_folder):
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_folder, class_name)
            for file in os.listdir(class_path):
                if file.endswith(".mat"):
                    file_path = os.path.join(class_path, file)
                    mat_data = sio.loadmat(file_path)
                    kl_matrix = mat_data[list(mat_data.keys())[-1]]

                    if kl_matrix.shape == (90, 90):
                        flat_kl = kl_matrix.flatten().astype(np.float32)
                        if self.global_min is not None and self.global_max is not None:
                            flat_kl = (flat_kl - self.global_min) / (self.global_max - self.global_min + 1e-8)
                        subject_id = extract_subject_id(file)
                        self.data.append(flat_kl)
                        self.labels.append(class_idx)
                        self.subject_ids.append(subject_id)

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx]), self.subject_ids[idx]



class SemiSupervisedVAE(nn.Module):
    def __init__(self, input_dim=8100, latent_dim=180, num_classes=4):
        super(SemiSupervisedVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

        # Classifier head (training signal to shape latent space)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        class_logits = self.classifier(z)
        return recon, mu, logvar, class_logits


def semi_supervised_vae_loss(recon_x, x, mu, logvar, class_logits, y, beta=1.0, alpha=0.5):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    class_loss = F.cross_entropy(class_logits, y)
    total_loss = recon_loss + beta * kl_div + alpha * class_loss
    return total_loss, recon_loss, kl_div, class_loss


def train_vae(model, dataloader, epochs, lr, beta=18.0, alpha=8.0):
    """Train VAE on entire dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_losses, recon_losses, kl_losses, class_losses = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_epoch_loss, recon_epoch_loss, kl_epoch_loss, class_epoch_loss = 0, 0, 0, 0

        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            recon, mu, logvar, class_logits = model(x)
            loss, recon_loss, kl_div, class_loss = semi_supervised_vae_loss(
                recon, x, mu, logvar, class_logits, y, beta, alpha
            )
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            recon_epoch_loss += recon_loss.item()
            kl_epoch_loss += kl_div.item()
            class_epoch_loss += class_loss.item()

        total_losses.append(total_epoch_loss / len(dataloader))
        recon_losses.append(recon_epoch_loss / len(dataloader))
        kl_losses.append(kl_epoch_loss / len(dataloader))
        class_losses.append(class_epoch_loss / len(dataloader))

        if (epoch + 1) % 50 == 0:
            print(f"  [VAE Epoch {epoch+1}/{epochs}] Total: {total_losses[-1]:.4f} | "
                  f"Recon: {recon_losses[-1]:.4f} | KL: {kl_losses[-1]:.4f} | Cls: {class_losses[-1]:.4f}")

    # Plot VAE training
    plt.figure(figsize=(10, 5))
    plt.plot(total_losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kl_losses, label="KL Divergence")
    plt.plot(class_losses, label="Classification Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Stage 1: Supervised VAE Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model


def extract_latents(model, dataloader):
    """Extract latent vectors using mu (deterministic, no sampling noise)."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    z_list, labels, subject_ids = [], [], []

    with torch.no_grad():
        for x, y, s_id in dataloader:
            x = x.to(device)
            mu, _ = model.encode(x)  # Use mu, NOT reparameterize
            z_list.append(mu.cpu().numpy())
            labels.append(y.numpy())
            subject_ids.extend(s_id)

    return np.concatenate(z_list), np.concatenate(labels), np.array(subject_ids)



class MLPClassifier(nn.Module):
    def __init__(self, input_dim=180, num_classes=4):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Sigmoid(),

            

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def normalize_latents(X):
    global_min = np.min(X)
    global_max = np.max(X)
    print(f"  Latents range: [{global_min:.4f}, {global_max:.4f}]")
    X_norm = (X - global_min) / (global_max - global_min + 1e-8)
    return X_norm


def subject_stratified_kfold(subject_ids, labels, k=5, seed=42):
    """Subject-wise stratified split for MLP evaluation."""
    import random
    random.seed(seed)
    subject_to_indices = defaultdict(list)
    subject_to_label = {}

    for idx, sid in enumerate(subject_ids):
        subject_to_indices[sid].append(idx)
        if sid not in subject_to_label:
            subject_to_label[sid] = labels[idx]

    subjects = list(subject_to_label.keys())
    subject_labels = [subject_to_label[sid] for sid in subjects]

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    folds = []
    for train_sub_idx, test_sub_idx in skf.split(subjects, subject_labels):
        train_ids = [subjects[i] for i in train_sub_idx]
        test_ids = [subjects[i] for i in test_sub_idx]

        train_idx = [i for sid in train_ids for i in subject_to_indices[sid]]
        test_idx = [i for sid in test_ids for i in subject_to_indices[sid]]

        folds.append((train_idx, test_idx))
    return folds


def train_mlp_kfold(X, y, subject_ids, k=5, epochs=100, batch_size=16, lr=1e-4):
    """Train and evaluate MLP with subject-based K-Fold on extracted latents."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = subject_stratified_kfold(subject_ids, y, k)

    fold_accs = []
    fold_f1s = []

    for fold, (train_idx, test_idx) in enumerate(folds):
        print(f"\n  --- MLP Fold {fold+1}/{k} ---")

        # Verify no leakage
        train_subjects = set(subject_ids[i] for i in train_idx)
        test_subjects = set(subject_ids[i] for i in test_idx)
        assert len(train_subjects & test_subjects) == 0, "Subject leakage in MLP folds!"

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).long()

        model = MLPClassifier(input_dim=X.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(Subset(TensorDataset(X_t, y_t), train_idx),
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(TensorDataset(X_t, y_t), test_idx),
                                 batch_size=batch_size, shuffle=False)

        train_accs, test_accs = [], []
        train_losses, test_losses = [], []

        for epoch in range(epochs):
            # Train
            model.train()
            total_loss, correct, total = 0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (preds.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
            train_accs.append(100 * correct / total)
            train_losses.append(total_loss / len(train_loader))

            # Test
            model.eval()
            total_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    total_loss += loss.item()
                    correct += (preds.argmax(dim=1) == yb).sum().item()
                    total += yb.size(0)
            test_accs.append(100 * correct / total)
            test_losses.append(total_loss / len(test_loader))

        # Final evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model(xb)
                all_preds.extend(preds.argmax(dim=1).cpu().numpy())
                all_labels.extend(yb.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)

        fold_accs.append(acc * 100)
        fold_f1s.append(f1 * 100)

        print(f"  Accuracy: {acc*100:.2f}%, Macro F1: {f1*100:.2f}%")
        print(f"  Confusion Matrix:\n{cm}")

        # Plot fold
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(test_accs, label='Test Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Fold {fold+1} Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold+1} Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # Summary
    print(f"\n  {'='*50}")
    print(f"  MLP RESULTS")
    print(f"  {'='*50}")
    for i in range(k):
        print(f"  Fold {i+1}: Acc={fold_accs[i]:.2f}%  F1={fold_f1s[i]:.2f}%")
    print(f"  {'─'*40}")
    print(f"  Mean Accuracy: {np.mean(fold_accs):.2f}% +/- {np.std(fold_accs):.2f}%")
    print(f"  Mean Macro F1: {np.mean(fold_f1s):.2f}% +/- {np.std(fold_f1s):.2f}%")
    print(f"  {'='*50}")

    return fold_accs, fold_f1s



def main():
    data_path = "/kaggle/input/adnida/"  # Update this path

    
    print("=" * 60)
    print("  STAGE 1: SUPERVISED VAE")
    print("=" * 60)

    dataset = FullKLDataset(data_path, global_min=0.0, global_max=26.8768)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = 90 * 90
    latent_dim = 180
    vae_lr = 1e-4
    vae_epochs = 100
    beta = 18.0
    alpha = 8.0

    print(f"  Loaded {len(dataset)} samples.")
    print(f"  Hyperparams: latent_dim={latent_dim}, beta={beta}, alpha={alpha}\n")

    vae_model = SemiSupervisedVAE(input_dim=input_dim, latent_dim=latent_dim, num_classes=4)
    vae_model = train_vae(vae_model, dataloader, vae_epochs, vae_lr, beta, alpha)

    # Extract latents (using mu, not reparameterize)
    print("\n  Extracting latent vectors...")
    z_latents, labels, subject_ids = extract_latents(vae_model, dataloader)
    print(f"  Extracted latents shape: {z_latents.shape}")

    # Normalize latents
    z_latents = normalize_latents(z_latents)


    print(f"\n{'='*60}")
    print("  STAGE 2: MLP CLASSIFIER ON EXTRACTED LATENTS")
    print(f"{'='*60}")
    print("  NOTE: VAE was trained on ALL data — test latents are NOT unseen by VAE.")
    print("  This means results may be optimistically biased.\n")

    fold_accs, fold_f1s = train_mlp_kfold(
        z_latents, labels, subject_ids,
        k=5, epochs=100, batch_size=16, lr=1e-4
    )

    print("\nDone!")


if __name__ == "__main__":
    main()