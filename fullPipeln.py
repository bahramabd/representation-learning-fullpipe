import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
import os
import scipy.io as sio


def extract_subject_id(filename):
    parts = filename.split("_")
    return parts[1] + "_" + parts[2] + "_" + parts[3]  # Example: "018_S_4733"


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


class JointVAEMLP(nn.Module):
    def __init__(self, input_dim=8100, latent_dim=180, num_classes=4):
        super(JointVAEMLP, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

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

        # VAE reconstruction
        recon = self.decoder(z)

        class_logits = self.classifier(z)

        return recon, mu, logvar, class_logits, z


def joint_loss(recon_x, x, mu, logvar, class_logits, y, beta=1.0, alpha=1.0):
    
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    class_loss = F.cross_entropy(class_logits, y)

    # Combined loss
    total_loss = recon_loss + beta * kl_div + alpha * class_loss

    return total_loss, recon_loss, kl_div, class_loss

def get_subject_folds(dataset, n_folds=5, random_state=42):
    
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []

    for train_idx, test_idx in sgkf.split(
        X=np.zeros(len(dataset.labels)),
        y=dataset.labels,
        groups=dataset.subject_ids
    ):
        # Safety check: no subject leakage
        train_subjects = set(dataset.subject_ids[train_idx])
        test_subjects = set(dataset.subject_ids[test_idx])
        assert len(train_subjects & test_subjects) == 0, "error"
        folds.append((train_idx, test_idx))

    return folds


def create_fold_dataloaders(dataset, train_idx, test_idx, batch_size=32):
    X_train = torch.tensor(dataset.data[train_idx])
    y_train = torch.tensor(dataset.labels[train_idx])
    X_test = torch.tensor(dataset.data[test_idx])
    y_test = torch.tensor(dataset.labels[test_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_joint_model(model, train_loader, test_loader, epochs=100, lr=1e-4, beta=18.0, alpha=8.0):
    """Joint training of VAE + MLP for one fold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Tracking metrics
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    recon_losses, kl_losses, class_losses = [], [], []

    best_test_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # ===== Training Phase =====
        model.train()
        total_train_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_class_loss = 0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            recon, mu, logvar, class_logits, _ = model(xb)

            loss, recon_loss, kl_div, class_loss = joint_loss(
                recon, xb, mu, logvar, class_logits, yb, beta, alpha
            )

            loss.backward()
            optimizer.step()

            # Track losses
            total_train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_div.item()
            total_class_loss += class_loss.item()

            # Track accuracy
            train_correct += (class_logits.argmax(dim=1) == yb).sum().item()
            train_total += yb.size(0)

        # Average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        train_losses.append(avg_train_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        class_losses.append(avg_class_loss)
        train_accs.append(train_acc)

        # ===== Testing Phase =====
        model.eval()
        total_test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon, mu, logvar, class_logits, _ = model(xb)

                loss, _, _, _ = joint_loss(recon, xb, mu, logvar, class_logits, yb, beta, alpha)

                total_test_loss += loss.item()
                test_correct += (class_logits.argmax(dim=1) == yb).sum().item()
                test_total += yb.size(0)

        avg_test_loss = total_test_loss / len(test_loader)
        test_acc = 100 * test_correct / test_total

        test_losses.append(avg_test_loss)
        test_accs.append(test_acc)

        # Track best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"  [Epoch {epoch + 1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_recon_loss:.4f}, "
                  f"KL: {avg_kl_loss:.4f}, Class: {avg_class_loss:.4f}) | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'recon_losses': recon_losses,
        'kl_losses': kl_losses,
        'class_losses': class_losses
    }

    return history, best_test_acc


def plot_training_results(history, fold_num=None):
    """Plot training results for a single fold."""
    title_suffix = f" (Fold {fold_num})" if fold_num else ""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy plot
    axes[0, 0].plot(history['train_accs'], label='Train Accuracy', color='blue')
    axes[0, 0].plot(history['test_accs'], label='Test Accuracy', color='red')
    axes[0, 0].set_title(f'Accuracy{title_suffix}')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Total loss plot
    axes[0, 1].plot(history['train_losses'], label='Train Loss', color='blue')
    axes[0, 1].plot(history['test_losses'], label='Test Loss', color='red')
    axes[0, 1].set_title(f'Total Loss{title_suffix}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Component losses plot
    axes[1, 0].plot(history['recon_losses'], label='Reconstruction Loss', color='green')
    axes[1, 0].plot(history['kl_losses'], label='KL Divergence', color='orange')
    axes[1, 0].plot(history['class_losses'], label='Classification Loss', color='purple')
    axes[1, 0].set_title(f'Component Losses{title_suffix}')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Loss ratios
    train_arr = np.array(history['train_losses'])
    train_arr[train_arr == 0] = 1e-8  # avoid division by zero
    axes[1, 1].plot(np.array(history['recon_losses']) / train_arr,
                    label='Recon/Total', color='green')
    axes[1, 1].plot(np.array(history['kl_losses']) / train_arr,
                    label='KL/Total', color='orange')
    axes[1, 1].plot(np.array(history['class_losses']) / train_arr,
                    label='Class/Total', color='purple')
    axes[1, 1].set_title(f'Loss Component Ratios{title_suffix}')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_cv_summary(fold_accs):
    """Plot summary bar chart of all folds."""
    n_folds = len(fold_accs)
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    plt.figure(figsize=(6, 4))
    plt.bar(range(1, n_folds + 1), fold_accs, color='steelblue', edgecolor='black')
    plt.axhline(y=mean_acc, color='red', linestyle='--',
                label=f'Mean: {mean_acc:.1f}% ± {std_acc:.1f}%')
    plt.xlabel('Fold')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Subject-wise 5-Fold Cross-Validation Results')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
def main():
    # Data path
    data_path = "/kaggle/input/adnida/"  # Update this path

    # Load dataset
    print("Loading dataset...")
    dataset = FullKLDataset(data_path, global_min=0.0, global_max=26.8768)
    print(f"Loaded {len(dataset)} samples from {len(set(dataset.subject_ids))} unique subjects.")

    # Model parameters (unchanged)
    input_dim = 90 * 90  # 8100
    latent_dim = 180
    num_classes = 4
    lr = 1e-4
    epochs = 300
    beta = 18.0   # Weight for KL divergence
    alpha = 8.0   # Weight for classification loss
    batch_size = 32
    n_folds = 5

    # Get subject-wise folds
    folds = get_subject_folds(dataset, n_folds=n_folds)

    fold_accs = []
    all_histories = []

    print(f"\n{'='*60}")
    print(f"  SUBJECT-WISE {n_folds}-FOLD CROSS-VALIDATION")
    print(f"  latent_dim={latent_dim}, beta={beta}, alpha={alpha}, lr={lr}")
    print(f"{'='*60}")

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_subjects = set(dataset.subject_ids[train_idx])
        test_subjects = set(dataset.subject_ids[test_idx])

        print(f"\n--- Fold {fold_i + 1}/{n_folds} ---")
        print(f"  Train: {len(train_idx)} samples ({len(train_subjects)} subjects)")
        print(f"  Test:  {len(test_idx)} samples ({len(test_subjects)} subjects)")
        print(f"  Test subjects: {sorted(test_subjects)}")

        # Create dataloaders for this fold
        train_loader, test_loader = create_fold_dataloaders(
            dataset, train_idx, test_idx, batch_size=batch_size
        )

        # Fresh model each fold
        model = JointVAEMLP(input_dim=input_dim, latent_dim=latent_dim, num_classes=num_classes)

        # Train
        history, best_acc = train_joint_model(
            model, train_loader, test_loader,
            epochs=epochs, lr=lr, beta=beta, alpha=alpha
        )

        fold_accs.append(best_acc)
        all_histories.append(history)

        print(f"  >> Fold {fold_i + 1} Best Test Accuracy: {best_acc:.2f}%")

        # Plot this fold
        plot_training_results(history, fold_num=fold_i + 1)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for i, acc in enumerate(fold_accs):
        print(f"  Fold {i + 1}: {acc:.2f}%")
    print(f"  --------------------------")
    print(f"  Mean Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"{'='*60}")

    # Summary plot
    plot_cv_summary(fold_accs)

    print("Training completed!")


if __name__ == "__main__":
    main()