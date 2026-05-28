import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
import scipy.io as sio


def extract_subject_id(filename):
    parts = filename.split("_")
    return parts[1] + "_" + parts[2] + "_" + parts[3]


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
            for file in sorted(os.listdir(class_path)):
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
        self.subject_ids = np.array(self.subject_ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])


class SupervisedVAE(nn.Module):
    """
    VAE with an internal classifier head.
    The classifier is a training signal to shape the latent space —
    it will be DISCARDED after training. Only the encoder is kept.
    """
    def __init__(self, input_dim=8100, latent_dim=180, num_classes=4):
        super(SupervisedVAE, self).__init__()

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

        # Internal classifier (training signal only — discarded after stage 1)
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


def vae_loss(recon_x, x, mu, logvar, class_logits, y, beta=18.0, alpha=8.0):
    """L_total = L_MSE + beta * D_KL + alpha * L_CE"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    class_loss = F.cross_entropy(class_logits, y)
    total_loss = recon_loss + beta * kl_div + alpha * class_loss
    return total_loss, recon_loss, kl_div, class_loss


def train_vae(model, train_loader, epochs, lr, beta, alpha, patience=30, verbose=True):
    """Train the supervised VAE on training data only."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            recon, mu, logvar, class_logits = model(xb)
            loss, _, _, _ = vae_loss(recon, xb, mu, logvar, class_logits, yb, beta, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    [VAE Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.2f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"    VAE early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def extract_latents(model, data_tensor):
    """
    Extract latent vectors using mu (mean) — NOT reparameterized samples.
    Using mu gives deterministic, noise-free representations for downstream use.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = data_tensor.to(device)
        mu, _ = model.encode(x)
        return mu.cpu().numpy()


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


def train_mlp(model, train_loader, test_loader, epochs, lr, patience=20, verbose=True):
    """Train the MLP classifier on extracted latent vectors."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_accs, test_accs = [], []
    train_losses, test_losses = [], []
    best_test_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # --- Train ---
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

        # --- Test ---
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

        test_acc = 100 * correct / total
        test_accs.append(test_acc)
        test_losses.append(total_loss / len(test_loader))

        # Early stopping on test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if verbose:
                print(f"    MLP early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {
        'train_accs': train_accs, 'test_accs': test_accs,
        'train_losses': train_losses, 'test_losses': test_losses
    }
    return model, history, best_test_acc


def get_subject_folds(dataset, n_folds=5, random_state=42):
    """Stratified Group K-Fold: no subject in both train and test."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []

    for train_idx, test_idx in sgkf.split(
        X=np.zeros(len(dataset.labels)),
        y=dataset.labels,
        groups=dataset.subject_ids
    ):
        # Safety check
        train_subjects = set(dataset.subject_ids[train_idx])
        test_subjects = set(dataset.subject_ids[test_idx])
        assert len(train_subjects & test_subjects) == 0, "Subject leakage!"
        folds.append((train_idx, test_idx))

    return folds


def run_two_stage_cv(dataset, n_folds=5, latent_dim=180, beta=18.0, alpha=8.0,
                     vae_epochs=300, vae_lr=1e-4,
                     mlp_epochs=200, mlp_lr=1e-3,
                     batch_size=16):
    """
    For EACH fold:
      1. Train a fresh supervised VAE on TRAIN subjects only
      2. Freeze encoder, extract latents for train AND test
      3. Train a fresh MLP on train latents
      4. Evaluate MLP on test latents (never seen by VAE)
    """

    input_dim = 8100
    num_classes = 4
    folds = get_subject_folds(dataset, n_folds=n_folds)

    fold_accs = []
    fold_f1s = []
    all_histories = []

    print("=" * 60)
    print(f"  TWO-STAGE PIPELINE: Supervised VAE → Separate MLP")
    print(f"  {n_folds}-Fold Subject-wise Cross-Validation")
    print(f"  Samples: {len(dataset)} | Subjects: {len(set(dataset.subject_ids))}")
    print(f"  VAE: latent={latent_dim}, beta={beta}, alpha={alpha}, lr={vae_lr}")
    print(f"  MLP: lr={mlp_lr}, epochs={mlp_epochs}")
    print("=" * 60)

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_subjects = set(dataset.subject_ids[train_idx])
        test_subjects = set(dataset.subject_ids[test_idx])

        print(f"\n{'─'*50}")
        print(f"  FOLD {fold_i + 1}/{n_folds}")
        print(f"  Train: {len(train_idx)} samples ({len(train_subjects)} subjects)")
        print(f"  Test:  {len(test_idx)} samples ({len(test_subjects)} subjects)")
        print(f"  Test subjects: {sorted(test_subjects)}")
        print(f"{'─'*50}")

        # Get data tensors
        X_train = torch.tensor(dataset.data[train_idx])
        y_train = torch.tensor(dataset.labels[train_idx])
        X_test = torch.tensor(dataset.data[test_idx])
        y_test = torch.tensor(dataset.labels[test_idx])

        print("  [Stage 1] Training Supervised VAE...")

        vae_train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )

        vae_model = SupervisedVAE(
            input_dim=input_dim, latent_dim=latent_dim, num_classes=num_classes
        )
        vae_model = train_vae(
            vae_model, vae_train_loader,
            epochs=vae_epochs, lr=vae_lr, beta=beta, alpha=alpha
        )

        print("  [Stage 1] Extracting latent vectors...")
        z_train = extract_latents(vae_model, X_train)
        z_test = extract_latents(vae_model, X_test)

        # Normalize latents (fit on train, apply to both)
        z_min = z_train.min()
        z_max = z_train.max()
        z_train = (z_train - z_min) / (z_max - z_min + 1e-8)
        z_test = (z_test - z_min) / (z_max - z_min + 1e-8)

        z_train_t = torch.tensor(z_train, dtype=torch.float32)
        z_test_t = torch.tensor(z_test, dtype=torch.float32)

        print("  [Stage 2] Training MLP Classifier...")

        mlp_train_loader = DataLoader(
            TensorDataset(z_train_t, y_train), batch_size=batch_size, shuffle=True
        )
        mlp_test_loader = DataLoader(
            TensorDataset(z_test_t, y_test), batch_size=batch_size, shuffle=False
        )

        mlp_model = MLPClassifier(input_dim=latent_dim, num_classes=num_classes)
        mlp_model, history, best_acc = train_mlp(
            mlp_model, mlp_train_loader, mlp_test_loader,
            epochs=mlp_epochs, lr=mlp_lr
        )

        # Final evaluation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in mlp_test_loader:
                xb = xb.to(device)
                preds = mlp_model(xb)
                all_preds.extend(preds.argmax(dim=1).cpu().numpy())
                all_labels.extend(yb.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)

        fold_accs.append(best_acc)
        fold_f1s.append(f1 * 100)
        all_histories.append(history)

        print(f"  >> Fold {fold_i+1} Best Accuracy: {best_acc:.2f}%")
        print(f"  >> Fold {fold_i+1} Macro F1: {f1*100:.2f}%")
        print(f"  >> Confusion Matrix:\n{cm}")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for i in range(n_folds):
        print(f"  Fold {i+1}: Acc={fold_accs[i]:.2f}%  F1={fold_f1s[i]:.2f}%")
    print(f"  {'─'*40}")
    print(f"  Mean Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"  Mean Macro F1: {mean_f1:.2f}% +/- {std_f1:.2f}%")
    print(f"{'='*60}")

    return fold_accs, fold_f1s, all_histories


def plot_results(all_histories, fold_accs):
    n_folds = len(all_histories)

    # Per-fold accuracy curves
    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 4), sharey=True)
    if n_folds == 1:
        axes = [axes]

    for i, history in enumerate(all_histories):
        axes[i].plot(history['train_accs'], label='Train Acc', color='blue')
        axes[i].plot(history['test_accs'], label='Test Acc', color='orange')
        axes[i].set_title(f'Fold {i+1} MLP (best: {fold_accs[i]:.1f}%)')
        axes[i].set_xlabel('Epoch')
        if i == 0:
            axes[i].set_ylabel('Accuracy (%)')
        axes[i].legend()
        axes[i].grid(True)

    plt.suptitle('Stage 2: MLP Accuracy per Fold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # Summary bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, n_folds + 1), fold_accs, color='steelblue', edgecolor='black')
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    plt.axhline(y=mean_acc, color='red', linestyle='--',
                label=f'Mean: {mean_acc:.1f}% +/- {std_acc:.1f}%')
    plt.xlabel('Fold')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Two-Stage Pipeline: Subject-wise 5-Fold CV')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    data_path = "/kaggle/input/adnida/"  # Update this path

    print("Loading dataset...")
    dataset = FullKLDataset(data_path, global_min=0.0, global_max=26.8768)
    print(f"Loaded {len(dataset)} samples from {len(set(dataset.subject_ids))} subjects.\n")

    fold_accs, fold_f1s, all_histories = run_two_stage_cv(
        dataset,
        n_folds=5,
        latent_dim=180,
        beta=18.0,
        alpha=8.0,
        vae_epochs=300,
        vae_lr=1e-4,
        mlp_epochs=200,
        mlp_lr=1e-3,
        batch_size=16
    )

    plot_results(all_histories, fold_accs)
    print("Done!")


if __name__ == "__main__":
    main()