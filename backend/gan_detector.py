"""
GAN-Based Cyber Attack Detection for Enterprise Client Platforms
================================================================
Architecture:
  - Generator  : Synthesizes realistic attack traffic to augment training data
  - Discriminator: Classifies network events as normal or attack
  - Anomaly Detector: Uses the discriminator's learned representation for
                      unsupervised / semi-supervised detection at inference

Supported attack categories (NSL-KDD / CICIDS inspired):
  DoS, Probe, R2L, U2R, Brute Force, Port Scan, Web Attack, Botnet, Infiltration

Dependencies:
  pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn tqdm
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score,
)

# ─────────────────────────────────────────────────────────────
# 0.  REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ─────────────────────────────────────────────────────────────
# 1.  SYNTHETIC DATASET  (replace with real PCAP/log features)
# ─────────────────────────────────────────────────────────────
FEATURE_DIM   = 41      # NSL-KDD has 41 network features
NOISE_DIM     = 64      # GAN latent dimension
N_SAMPLES     = 20_000
ATTACK_RATIO  = 0.35    # 35 % of samples are attacks

ATTACK_TYPES = [
    "DoS", "Probe", "R2L", "U2R",
    "BruteForce", "PortScan", "WebAttack", "Botnet",
]

def generate_synthetic_dataset(n_samples: int = N_SAMPLES,
                                attack_ratio: float = ATTACK_RATIO,
                                feature_dim: int = FEATURE_DIM) -> pd.DataFrame:
    """
    Simulates realistic network-flow feature vectors.
    Normal traffic:  low port numbers, small byte counts, stable jitter.
    Attack traffic:  high connection rates, unusual byte distributions, flag anomalies.
    """
    n_attack = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attack

    rng = np.random.RandomState(SEED)

    # ── Normal traffic ──────────────────────────────────────
    normal = rng.randn(n_normal, feature_dim) * 0.5
    normal[:, 0]  = rng.uniform(0, 1024, n_normal)      # src_port (benign)
    normal[:, 1]  = rng.uniform(80, 443, n_normal)       # dst_port
    normal[:, 2]  = rng.exponential(500, n_normal)       # bytes_sent
    normal[:, 3]  = rng.exponential(300, n_normal)       # bytes_recv
    normal[:, 4]  = rng.uniform(0.001, 0.5, n_normal)    # duration_s
    normal[:, 5]  = rng.randint(1, 5, n_normal)          # tcp_flags

    # ── Attack traffic ──────────────────────────────────────
    attacks = rng.randn(n_attack, feature_dim) * 1.5 + 2.0
    attacks[:, 0] = rng.uniform(1024, 65535, n_attack)   # high ephemeral ports
    attacks[:, 1] = rng.choice([22, 23, 3389, 445, 80, 8080], n_attack)
    attacks[:, 2] = rng.exponential(5000, n_attack)      # large payload
    attacks[:, 3] = rng.exponential(100, n_attack)
    attacks[:, 4] = rng.uniform(0.0, 0.01, n_attack)     # very short / flooding
    attacks[:, 5] = rng.randint(0, 32, n_attack)         # arbitrary flags

    attack_labels = rng.choice(ATTACK_TYPES, n_attack)

    X = np.vstack([normal, attacks])
    y_binary = np.array([0] * n_normal + [1] * n_attack)
    y_type   = np.array(["Normal"] * n_normal + list(attack_labels))

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(feature_dim)])
    df["label"]        = y_binary
    df["attack_type"]  = y_type

    # ── Add extra columns for dashboard display ────────────────
    countries = ["USA", "China", "Germany", "France", "Japan", "UK", "Russia", "Brazil", "India", "South Korea"]
    protocols = ["TCP", "UDP", "ICMP"]
    ml_models = ["K-Nearest", "Neural Network", "Support Vector", "Logistic Regression", "Random Forest"]
    affected_systems = ["Workstation", "Web Server", "Database", "Firewall", "Application Server", "Network Router"]

    df["source_ip"] = [f"{rng.randint(1,255)}.{rng.randint(1,255)}.{rng.randint(1,255)}.{rng.randint(1,255)}" for _ in range(n_samples)]
    df["destination_ip"] = [f"{rng.randint(1,255)}.{rng.randint(1,255)}.{rng.randint(1,255)}.{rng.randint(1,255)}" for _ in range(n_samples)]
    df["source_country"] = rng.choice(countries, n_samples)
    df["destination_country"] = rng.choice(countries, n_samples)
    df["protocol"] = rng.choice(protocols, n_samples)
    df["ml_model"] = rng.choice(ml_models, n_samples)
    df["affected_system"] = rng.choice(affected_systems, n_samples)
    df["port_type"] = "Other"

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 2.  DATA PIPELINE
# ─────────────────────────────────────────────────────────────

class NetworkFlowDataset:
    """Preprocesses raw network-flow DataFrame for GAN + classifier training."""

    def __init__(self, df: pd.DataFrame):
        self.scaler = StandardScaler()
        self.le     = LabelEncoder()

        feature_cols = [c for c in df.columns if c.startswith("feature_")]
        X = df[feature_cols].values.astype(np.float32)
        y = df["label"].values.astype(np.float32)

        X_scaled = self.scaler.fit_transform(X)

        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(
            X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
        )

        # Separate normal samples for GAN generator training
        normal_mask = self.y_train == 0
        self.X_normal = self.X_train[normal_mask]
        self.X_attack = self.X_train[~normal_mask]

    def get_loaders(self, batch_size: int = 256):
        train_ds = TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.FloatTensor(self.y_train),
        )
        test_ds  = TensorDataset(
            torch.FloatTensor(self.X_test),
            torch.FloatTensor(self.y_test),
        )
        normal_ds = TensorDataset(torch.FloatTensor(self.X_normal))

        return (
            DataLoader(train_ds,  batch_size=batch_size, shuffle=True,  drop_last=True),
            DataLoader(test_ds,   batch_size=batch_size, shuffle=False),
            DataLoader(normal_ds, batch_size=batch_size, shuffle=True,  drop_last=True),
        )


# ─────────────────────────────────────────────────────────────
# 3.  GAN ARCHITECTURE
# ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Lightweight residual block for tabular data."""
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class Generator(nn.Module):
    """
    Maps Gaussian noise → synthetic normal network-flow vectors.
    Trained to fool the Discriminator into labelling its outputs as 'normal'.
    """
    def __init__(self, noise_dim: int = NOISE_DIM, out_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(512),
            nn.Linear(512, out_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    Binary classifier: normal (0) vs attack (1).
    Also serves as the anomaly-scoring backbone at inference.
    """
    def __init__(self, in_dim: int = FEATURE_DIM, dropout: float = 0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        logit    = self.classifier(features)
        return logit, features  # return both for anomaly scoring


# ─────────────────────────────────────────────────────────────
# 4.  GAN TRAINER
# ─────────────────────────────────────────────────────────────

class GANCyberDetector:
    """
    Trains a Wasserstein-GAN with gradient penalty (WGAN-GP) to:
      1. Generate synthetic normal traffic (data augmentation).
      2. Produce a Discriminator that separates attack from normal traffic.
    """
    def __init__(self,
                 feature_dim: int = FEATURE_DIM,
                 noise_dim:   int = NOISE_DIM,
                 lr_g: float = 1e-4,
                 lr_d: float = 1e-4,
                 n_critic: int = 5,
                 gp_lambda: float = 10.0):

        self.noise_dim = noise_dim
        self.n_critic  = n_critic
        self.gp_lambda = gp_lambda

        self.G = Generator(noise_dim, feature_dim).to(DEVICE)
        self.D = Discriminator(feature_dim).to(DEVICE)

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.0, 0.9))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.0, 0.9))

        # Supervised loss for labelled samples
        self.bce = nn.BCEWithLogitsLoss()

        self.history = {"d_loss": [], "g_loss": [], "d_acc": []}

    # ── Gradient Penalty ────────────────────────────────────
    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        b = real.size(0)
        alpha = torch.rand(b, 1, device=DEVICE).expand_as(real)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        logit, _ = self.D(interp)
        grads = torch.autograd.grad(
            outputs=logit, inputs=interp,
            grad_outputs=torch.ones_like(logit),
            create_graph=True, retain_graph=True,
        )[0]
        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    # ── One Training Epoch ──────────────────────────────────
    def train_epoch(self,
                    train_loader,
                    normal_loader,
                    supervised: bool = True) -> dict:
        self.G.train(); self.D.train()
        d_losses, g_losses, accs = [], [], []

        normal_iter = iter(normal_loader)

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE).unsqueeze(1)
            b = batch_X.size(0)

            # ── Discriminator update (n_critic times) ──────
            for _ in range(self.n_critic):
                try:
                    (real_normal,) = next(normal_iter)
                except StopIteration:
                    normal_iter = iter(normal_loader)
                    (real_normal,) = next(normal_iter)

                real_normal = real_normal.to(DEVICE)
                z = torch.randn(b, self.noise_dim, device=DEVICE)
                fake = self.G(z).detach()

                d_real, _ = self.D(real_normal[:b])
                d_fake, _ = self.D(fake)
                gp = self._gradient_penalty(real_normal[:b], fake)

                # WGAN-GP critic loss
                d_loss_wgan = d_fake.mean() - d_real.mean() + self.gp_lambda * gp

                # Optional supervised cross-entropy on labelled batch
                if supervised:
                    d_sup_logit, _ = self.D(batch_X)
                    d_loss_sup = self.bce(d_sup_logit, batch_y)
                    d_loss = d_loss_wgan + d_loss_sup
                else:
                    d_loss = d_loss_wgan

                self.opt_D.zero_grad()
                d_loss.backward()
                self.opt_D.step()

            d_losses.append(d_loss.item())

            # ── Generator update ───────────────────────────
            z = torch.randn(b, self.noise_dim, device=DEVICE)
            fake = self.G(z)
            d_fake, _ = self.D(fake)
            g_loss = -d_fake.mean()

            self.opt_G.zero_grad()
            g_loss.backward()
            self.opt_G.step()

            g_losses.append(g_loss.item())

            # Accuracy on labelled batch
            with torch.no_grad():
                logits, _ = self.D(batch_X)
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == batch_y).float().mean().item()
                accs.append(acc)

        return {
            "d_loss": np.mean(d_losses),
            "g_loss": np.mean(g_losses),
            "d_acc":  np.mean(accs),
        }

    # ── Full Training Loop ──────────────────────────────────
    def fit(self, train_loader, normal_loader, epochs: int = 50):
        print("\n" + "═" * 60)
        print("  GAN TRAINING — Enterprise Cyber Attack Detector")
        print("═" * 60)
        for epoch in tqdm(range(1, epochs + 1), desc="Training"):
            stats = self.train_epoch(train_loader, normal_loader)
            for k, v in stats.items():
                self.history[k].append(v)
            if epoch % 10 == 0:
                tqdm.write(
                    f"  Epoch {epoch:03d} | "
                    f"D-loss: {stats['d_loss']:+.4f} | "
                    f"G-loss: {stats['g_loss']:+.4f} | "
                    f"D-acc: {stats['d_acc']:.3f}"
                )
        print("═" * 60)

    # ── Evaluation ──────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, test_loader) -> dict:
        self.D.eval()
        all_probs, all_labels = [], []

        for batch_X, batch_y in test_loader:
            logits, _ = self.D(batch_X.to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(batch_y.numpy())

        probs  = np.array(all_probs)
        labels = np.array(all_labels, dtype=int)
        preds  = (probs > 0.5).astype(int)

        return {
            "probs":  probs,
            "preds":  preds,
            "labels": labels,
        }

    # ── Anomaly Score (unsupervised inference) ───────────────
    @torch.no_grad()
    def anomaly_score(self, X: np.ndarray,
                      n_samples: int = 64,
                      threshold: float = 0.5) -> np.ndarray:
        """
        Reconstruction-based anomaly score:
          score = ||D_features(x) – mean(D_features(G(z)))||₂
        High score ⇒ likely attack.
        """
        self.G.eval(); self.D.eval()
        X_t = torch.FloatTensor(X).to(DEVICE)

        # Reference distribution from generator
        z   = torch.randn(n_samples, self.noise_dim, device=DEVICE)
        ref = self.G(z)
        _, ref_feat = self.D(ref)
        ref_mean = ref_feat.mean(0, keepdim=True)  # (1, 128)

        _, x_feat = self.D(X_t)                    # (N, 128)
        scores = (x_feat - ref_mean).norm(dim=1).cpu().numpy()
        return scores


# ─────────────────────────────────────────────────────────────
# 5.  VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str = "gan_training_history.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#58a6ff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    epochs = range(1, len(history["d_loss"]) + 1)
    axes[0].plot(epochs, history["d_loss"], color="#ff6b6b", lw=2)
    axes[0].set_title("Discriminator Loss (WGAN-GP)")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, history["g_loss"], color="#51cf66", lw=2)
    axes[1].set_title("Generator Loss")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, history["d_acc"], color="#74c0fc", lw=2)
    axes[2].axhline(0.5, color="#888", ls="--", lw=1)
    axes[2].set_title("Discriminator Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)

    plt.suptitle("GAN Training — Enterprise Cyber Attack Detector",
                 color="#e6edf3", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {save_path}")


def plot_roc_pr(results: dict, save_path: str = "gan_roc_pr.png"):
    labels, probs = results["labels"], results["probs"]
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_roc = roc_auc_score(labels, probs)
    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        for k in ("tick_params", ):
            ax.tick_params(colors="#c9d1d9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#58a6ff")

    ax1.plot(fpr, tpr, color="#ff6b6b", lw=2, label=f"AUC = {auc_roc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve"); ax1.legend(facecolor="#21262d", labelcolor="#e6edf3")

    ax2.plot(rec, prec, color="#51cf66", lw=2, label=f"AP = {ap:.4f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(facecolor="#21262d", labelcolor="#e6edf3")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {save_path}")


def plot_confusion_matrix(results: dict, save_path: str = "gan_confusion_matrix.png"):
    cm = confusion_matrix(results["labels"], results["preds"])
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                cmap="YlOrRd",
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"],
                linewidths=0.5, linecolor="#30363d")
    ax.set_xlabel("Predicted", color="#c9d1d9")
    ax.set_ylabel("Actual",    color="#c9d1d9")
    ax.set_title("Confusion Matrix", color="#58a6ff")
    ax.tick_params(colors="#c9d1d9")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {save_path}")


def plot_anomaly_scores(scores: np.ndarray, labels: np.ndarray,
                        threshold: float,
                        save_path: str = "gan_anomaly_scores.png"):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    normal_idx = labels == 0
    attack_idx = labels == 1

    ax.scatter(np.where(normal_idx)[0], scores[normal_idx],
               s=4, alpha=0.4, color="#74c0fc", label="Normal")
    ax.scatter(np.where(attack_idx)[0], scores[attack_idx],
               s=4, alpha=0.6, color="#ff6b6b", label="Attack")
    ax.axhline(threshold, color="#ffd43b", lw=1.5, ls="--", label=f"Threshold={threshold:.2f}")
    ax.set_xlabel("Sample Index", color="#c9d1d9")
    ax.set_ylabel("Anomaly Score", color="#c9d1d9")
    ax.set_title("GAN Anomaly Scores — Enterprise Traffic",color="#58a6ff")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#21262d", labelcolor="#e6edf3")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {save_path}")


def plot_generated_vs_real(model, X_real: np.ndarray,
                           save_path: str = "gan_generated_vs_real.png"):
    """t-SNE comparison: real normal traffic vs GAN-generated."""
    from sklearn.manifold import TSNE

    model.G.eval()
    with torch.no_grad():
        z = torch.randn(min(500, len(X_real)), model.noise_dim, device=DEVICE)
        fake = model.G(z).cpu().numpy()

    real_sub  = X_real[:500]
    combined  = np.vstack([real_sub, fake])
    tsne_out  = TSNE(n_components=2, random_state=SEED,
                     perplexity=30).fit_transform(combined)

    n = len(real_sub)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.scatter(tsne_out[:n, 0], tsne_out[:n, 1],
               s=8, alpha=0.5, color="#74c0fc", label="Real Normal")
    ax.scatter(tsne_out[n:, 0], tsne_out[n:, 1],
               s=8, alpha=0.5, color="#51cf66", label="GAN Generated")
    ax.set_title("t-SNE: Real Normal vs GAN-Generated Traffic", color="#58a6ff")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#21262d", labelcolor="#e6edf3")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {save_path}")


# ─────────────────────────────────────────────────────────────
# 6.  REAL-TIME INFERENCE SIMULATOR
# ─────────────────────────────────────────────────────────────

class EnterpriseDetector:
    """
    Wraps the trained GAN Discriminator for streaming inference.
    Suitable for integration with SIEM / EDR pipelines.
    """
    SEVERITY = {
        "CRITICAL": ("🔴", ">= 0.90"),
        "HIGH":     ("🟠", "0.75 – 0.90"),
        "MEDIUM":   ("🟡", "0.50 – 0.75"),
        "LOW":      ("🟢", "< 0.50"),
    }

    def __init__(self, model: GANCyberDetector, scaler: StandardScaler,
                 threshold: float = 0.5):
        self.model     = model
        self.scaler    = scaler
        self.threshold = threshold

    def classify(self, raw_features: np.ndarray) -> list:
        X = self.scaler.transform(raw_features)
        X_t = torch.FloatTensor(X).to(DEVICE)
        with torch.no_grad():
            logits, _ = self.model.D(X_t)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        results = []
        for i, p in enumerate(probs):
            if p >= 0.90:
                sev = "CRITICAL"
            elif p >= 0.75:
                sev = "HIGH"
            elif p >= 0.50:
                sev = "MEDIUM"
            else:
                sev = "LOW"
            results.append({"sample": i, "attack_prob": round(float(p), 4),
                             "is_attack": bool(p >= self.threshold),
                             "severity": sev})
        return results

    def stream_demo(self, df: pd.DataFrame, n: int = 10):
        print("\n" + "═" * 55)
        print("  ENTERPRISE REAL-TIME THREAT DETECTION — LIVE FEED")
        print("═" * 55)
        feature_cols = [c for c in df.columns if c.startswith("feature_")]
        samples = df.sample(n, random_state=SEED)

        for idx, row in enumerate(samples.itertuples()):
            raw = np.array([[getattr(row, c) for c in feature_cols]])
            res = self.classify(raw)[0]
            icon = self.SEVERITY[res["severity"]][0]
            true_lbl = "ATTACK" if row.label == 1 else "NORMAL"
            atk_type = row.attack_type
            print(
                f"  {icon} [{res['severity']:8s}] "
                f"prob={res['attack_prob']:.4f}  "
                f"detected={'YES' if res['is_attack'] else 'NO ':3}  "
                f"true={true_lbl:6}  type={atk_type}"
            )
            time.sleep(0.05)
        print("═" * 55)


# ─────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 60)
    print("  GAN-BASED CYBER ATTACK DETECTION — ENTERPRISE PLATFORM")
    print("█" * 60)

    # ── 7.1  Data ────────────────────────────────────────────
    print("\n[1/5] Generating synthetic enterprise network dataset …")
    df = generate_synthetic_dataset()
    print(f"      Total samples : {len(df):,}")
    print(f"      Normal        : {(df.label==0).sum():,}")
    print(f"      Attack        : {(df.label==1).sum():,}")
    print(f"      Attack types  : {', '.join(ATTACK_TYPES)}")

    dataset = NetworkFlowDataset(df)
    train_loader, test_loader, normal_loader = dataset.get_loaders(batch_size=256)

    # ── 7.2  Train GAN ───────────────────────────────────────
    print("\n[2/5] Training GAN (Generator + Discriminator) …")
    model = GANCyberDetector(
        feature_dim=FEATURE_DIM,
        noise_dim=NOISE_DIM,
        lr_g=1e-4, lr_d=1e-4,
        n_critic=5, gp_lambda=10.0,
    )
    model.fit(train_loader, normal_loader, epochs=60)

    # ── 7.3  Evaluate ────────────────────────────────────────
    print("\n[3/5] Evaluating on held-out test set …")
    results = model.evaluate(test_loader)
    print("\n  Classification Report:")
    print(classification_report(results["labels"], results["preds"],
                                 target_names=["Normal", "Attack"]))
    auc = roc_auc_score(results["labels"], results["probs"])
    ap  = average_precision_score(results["labels"], results["probs"])
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Avg-Prec: {ap:.4f}")

    # ── 7.4  Anomaly scoring ─────────────────────────────────
    print("\n[4/5] Computing anomaly scores …")
    X_test_np = dataset.X_test
    y_test_np = dataset.y_test.astype(int)
    scores    = model.anomaly_score(X_test_np)
    # Threshold = mean + 1 std of normal scores
    normal_scores = scores[y_test_np == 0]
    threshold = float(normal_scores.mean() + normal_scores.std())
    print(f"      Anomaly threshold: {threshold:.4f}")

    # ── 7.5  Visualisations ──────────────────────────────────
    print("\n[5/5] Generating visualisation plots …")
    plot_training_history(model.history)
    plot_roc_pr(results)
    plot_confusion_matrix(results)
    plot_anomaly_scores(scores, y_test_np, threshold)
    plot_generated_vs_real(model, dataset.X_normal)

    # ── Live demo ────────────────────────────────────────────
    detector = EnterpriseDetector(model, dataset.scaler, threshold=0.5)
    detector.stream_demo(df, n=15)

    print("\nDone!  All outputs saved to working directory.\n")


if __name__ == "__main__":
    main()
