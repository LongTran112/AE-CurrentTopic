import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available. Use --device cpu or --device auto.")
    return torch.device(device_arg)


def get_mnist_train_and_test(data_dir: str) -> Tuple[datasets.MNIST, datasets.MNIST]:
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return train_ds, test_ds


def subset_by_labels(dataset: datasets.MNIST, allowed_labels: List[int], sample_count: int = None, seed: int = 0) -> Subset:
    allowed = set(allowed_labels)
    indices = [idx for idx, label in enumerate(dataset.targets.tolist()) if int(label) in allowed]
    if sample_count is not None:
        if sample_count > len(indices):
            raise ValueError(
                f"Requested {sample_count} samples but only {len(indices)} available for labels {allowed_labels}."
            )
        rng = random.Random(seed)
        indices = rng.sample(indices, sample_count)
    return Subset(dataset, indices)


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(z)
        return decoded.view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        recon = self.decode(z)
        return recon


def save_weights(model: nn.Module, path: str) -> None:
    ensure_parent_dir(path)
    torch.save(model.state_dict(), path)
    print(f"Saved weights to: {path}")


def load_weights(model: nn.Module, path: str, device: torch.device) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights file not found: {path}. Run --mode train first.")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded weights from: {path}")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def train_ae(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    set_seed(args.seed)
    train_ds, _ = get_mnist_train_and_test(args.data_dir)
    train_subset = subset_by_labels(train_ds, [0])
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)

    model = AutoEncoder(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    epoch_losses: List[float] = []
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        sample_count = 0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()

            batch = images.size(0)
            running_loss += loss.item() * batch
            sample_count += batch

        epoch_loss = running_loss / max(sample_count, 1)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1:03d}/{args.epochs:03d} - train_mse: {epoch_loss:.6f}")

    save_weights(model, args.weights_path)
    plot_training_curve(epoch_losses, args.train_curve_path)


def plot_training_curve(epoch_losses: List[float], out_path: str) -> None:
    ensure_parent_dir(out_path)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.title("AE Training Loss Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved training curve to: {out_path}")


def latent_to_square_image(latent: np.ndarray) -> np.ndarray:
    latent_dim = latent.shape[0]
    side = int(math.ceil(math.sqrt(latent_dim)))
    target = side * side
    if target != latent_dim:
        padded = np.zeros(target, dtype=np.float32)
        padded[:latent_dim] = latent
        latent = padded
    return latent.reshape(side, side)


def run_latent_viz(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    set_seed(args.seed)
    _, test_ds = get_mnist_train_and_test(args.data_dir)
    eval_subset = subset_by_labels(test_ds, [0, 1], sample_count=100, seed=args.seed)
    eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False)

    model = AutoEncoder(latent_dim=args.latent_dim).to(device)
    load_weights(model, args.weights_path, device)
    model.eval()

    latent_list: List[np.ndarray] = []
    label_list: List[int] = []
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            latents = model.encode(images).cpu().numpy()
            latent_list.extend([lat for lat in latents])
            label_list.extend(labels.numpy().tolist())

    total = len(latent_list)
    cols = 10
    rows = int(math.ceil(total / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < total:
            latent_img = latent_to_square_image(latent_list[i])
            ax.imshow(latent_img, cmap="viridis")
            ax.set_title(f"y={label_list[i]}", fontsize=8)

    plt.tight_layout()
    ensure_parent_dir(args.latent_plot_path)
    plt.savefig(args.latent_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved latent visualization for {total} samples to: {args.latent_plot_path}")


@dataclass
class ThresholdResult:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    tp: int
    tn: int
    fp: int
    fn: int


def compute_confusion(labels: np.ndarray, preds: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    return tp, tn, fp, fn


def evaluate_threshold(labels: np.ndarray, errors: np.ndarray, threshold: float) -> ThresholdResult:
    preds = (errors >= threshold).astype(np.int64)
    tp, tn, fp, fn = compute_confusion(labels, preds)
    total = max(len(labels), 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return ThresholdResult(threshold, accuracy, precision, recall, tp, tn, fp, fn)


def plot_error_histogram(errors: np.ndarray, labels: np.ndarray, out_path: str) -> None:
    cls0 = errors[labels == 0]
    cls1 = errors[labels == 1]
    plt.figure(figsize=(8, 5))
    plt.hist(cls0, bins=40, alpha=0.6, label="Class 0", color="tab:blue")
    plt.hist(cls1, bins=40, alpha=0.6, label="Class 1", color="tab:orange")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Count")
    plt.title("Reconstruction Error Distribution by Class")
    plt.legend()
    plt.tight_layout()
    ensure_parent_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved MSE histogram to: {out_path}")


def run_threshold_search(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    set_seed(args.seed)
    _, test_ds = get_mnist_train_and_test(args.data_dir)
    eval_subset = subset_by_labels(test_ds, [0, 1], sample_count=1000, seed=args.seed)
    eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False)

    model = AutoEncoder(latent_dim=args.latent_dim).to(device)
    load_weights(model, args.weights_path, device)
    model.eval()

    sample_errors: List[float] = []
    sample_labels: List[int] = []
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            recon = model(images)
            per_sample_mse = torch.mean((recon - images) ** 2, dim=(1, 2, 3))
            sample_errors.extend(per_sample_mse.cpu().numpy().tolist())
            sample_labels.extend(labels.numpy().tolist())

    errors = np.array(sample_errors, dtype=np.float32)
    labels = np.array(sample_labels, dtype=np.int64)

    thresholds = np.unique(errors)
    best = None
    for t in thresholds:
        result = evaluate_threshold(labels, errors, float(t))
        if best is None or result.accuracy > best.accuracy:
            best = result

    assert best is not None
    print("Best threshold search result")
    print(f"- samples: {len(labels)}")
    print(f"- best_threshold: {best.threshold:.8f}")
    print(f"- best_accuracy: {best.accuracy:.4f}")
    print(f"- precision_class1: {best.precision:.4f}")
    print(f"- recall_class1: {best.recall:.4f}")
    print(f"- confusion_matrix: TP={best.tp}, TN={best.tn}, FP={best.fp}, FN={best.fn}")

    plot_error_histogram(errors, labels, args.hist_plot_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNIST class-0 AutoEncoder assignment script.")
    parser.add_argument("--mode", required=True, choices=["train", "latent_viz", "threshold_search"])
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--weights_path", default="ae_mnist0.pt")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--latent_plot_path", default="outputs/ae_latent_vectors_0_1.png")
    parser.add_argument("--hist_plot_path", default="outputs/ae_mse_hist_0_1.png")
    parser.add_argument("--train_curve_path", default="outputs/ae_train_loss_curve.png")
    parser.add_argument("--save_hist", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        train_ae(args)
    elif args.mode == "latent_viz":
        run_latent_viz(args)
    elif args.mode == "threshold_search":
        run_threshold_search(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
