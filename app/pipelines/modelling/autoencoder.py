"""Convolutional Autoencoder baseline model and evaluation pipeline."""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from app.core.logger import logger
from app.domain import MVTecImageDataset, build_mvtec_manifest


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for reconstructed image anomaly detection."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 128) -> None:
        """Initialize encoder and decoder convolutional modules.

        Args:
            in_channels: Number of input image channels (default 3 for RGB).
            latent_channels: Number of bottleneck feature map channels.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass encoding to spatial bottleneck and reconstructing.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Reconstructed image tensor of shape (B, C, H, W).
        """
        decoded: torch.Tensor = self.decoder(self.encoder(x))
        return decoded


def evaluate_autoencoder(
    model: nn.Module,
    test_loader: DataLoader[Any],
    class_names: list[str] | None = None,  # Ignore Parameters  # noqa: ARG001
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], dict[str, Any]]:
    """Evaluate autoencoder reconstruction error on test set.

    Args:
        model: Trained autoencoder model.
        test_loader: Test dataloader yielding (x, y_cls, path/c_name).
        class_names: Optional class names list.

    Returns:
        Tuple of (reconstruction scores, true labels, metrics dictionary).
    """
    model.eval()
    scores: list[float] = []
    true_labels: list[int] = []
    true_class_names: list[str] = []

    with torch.no_grad():
        for x, y_cls, c_name in test_loader:
            rec = model(x)
            mse_per_sample = torch.mean((x - rec) ** 2, dim=(1, 2, 3)).cpu().numpy()
            scores.extend(mse_per_sample.tolist())
            true_labels.extend(y_cls.tolist() if isinstance(y_cls, torch.Tensor) else list(y_cls))
            if isinstance(c_name, tuple | list):
                true_class_names.extend(c_name)
            else:
                true_class_names.append(str(c_name))

    binary_gt = [0 if label == 0 else 1 for label in true_labels]
    normal_scores = [s for s, b in zip(scores, binary_gt, strict=False) if b == 0]
    threshold = float(np.percentile(normal_scores, 95)) if normal_scores else 0.0

    preds = [0 if s <= threshold else 1 for s in scores]
    logger.info("--- Multi-Class Anomaly Evaluation (Autoencoder) ---")
    logger.info("Decision Threshold (95th percentile normal): %.6f", threshold)

    report_str = classification_report(binary_gt, preds, target_names=["good", "defective"], zero_division=0)

    auroc = 0.0
    if len(set(binary_gt)) > 1:
        auroc = float(roc_auc_score(binary_gt, scores))

    acc = float(accuracy_score(binary_gt, preds))
    prec = float(precision_score(binary_gt, preds, zero_division=0))
    rec_score = float(recall_score(binary_gt, preds, zero_division=0))

    metrics = {
        "threshold": threshold,
        "auroc": auroc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec_score,
        "report": report_str,
        "total_samples": len(scores),
        "anomalous_samples": sum(binary_gt),
    }

    return np.array(scores), np.array(true_labels), metrics


def prepare_autoencoder_dataloaders(
    data_root: str,
    category: str,
    img_size: int = 64,
    batch_size: int = 16,
) -> tuple[DataLoader[Any], DataLoader[Any], int]:
    """Build and return train/test dataloaders for the specified category.

    Args:
        data_root: Root directory of MVTec dataset.
        category: Object category to train on (e.g. 'bottle').
        img_size: Image resize dimension (width and height).
        batch_size: Batch size for training and evaluation.

    Returns:
        Tuple of (train_loader, test_loader, num_train_samples).
    """
    manifest = build_mvtec_manifest(data_root)
    cat_manifest = manifest[manifest["product"] == category].copy()

    if cat_manifest.empty:
        raise ValueError(f"No samples found for category '{category}' in {data_root}")

    train_df = cat_manifest[(cat_manifest["split"] == "train") & (~cat_manifest["is_anomaly"])].copy()
    test_df = cat_manifest[cat_manifest["split"] == "test"].copy()

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = MVTecImageDataset(train_df, transform=transform)
    test_dataset = MVTecImageDataset(test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(train_dataset)


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader[Any],
    num_train_samples: int,
    epochs: int = 5,
    lr: float = 1e-3,
) -> float:
    """Train the autoencoder model and return the final epoch loss.

    Args:
        model: Autoencoder neural network model to train.
        train_loader: DataLoader providing training image batches.
        num_train_samples: Total number of training samples for loss normalization.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.

    Returns:
        Final average epoch loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, _, _ in train_loader:
            optimizer.zero_grad()
            rec = model(x)
            loss = criterion(rec, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x)
        final_loss = epoch_loss / max(num_train_samples, 1)
        logger.info("Epoch %d/%d - Loss: %.6f", epoch + 1, epochs, final_loss)

    return final_loss


def run_autoencoder_pipeline(
    data_root: str = "data/raw/mvtec_ad",
    category: str = "bottle",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    latent_dim: int = 128,
    img_size: int = 64,
) -> dict[str, Any]:
    """Train and evaluate the ConvAutoencoder on a specified category.

    Args:
        data_root: Root directory of MVTec dataset.
        category: Object category to train on (e.g. 'bottle').
        epochs: Number of training epochs.
        batch_size: Batch size for training and evaluation.
        lr: Learning rate for Adam optimizer.
        latent_dim: Latent representation channels.
        img_size: Image resize dimension (width and height).

    Returns:
        Dictionary containing training and evaluation results.
    """
    logger.info("Starting ConvAutoencoder pipeline for category: %s", category)

    train_loader, test_loader, num_train_samples = prepare_autoencoder_dataloaders(
        data_root=data_root,
        category=category,
        img_size=img_size,
        batch_size=batch_size,
    )

    model = ConvAutoencoder(in_channels=3, latent_channels=latent_dim)
    final_loss = train_autoencoder(
        model=model,
        train_loader=train_loader,
        num_train_samples=num_train_samples,
        epochs=epochs,
        lr=lr,
    )

    _, _, metrics = evaluate_autoencoder(model, test_loader)
    metrics["final_train_loss"] = final_loss
    metrics["category"] = category
    metrics["epochs"] = epochs
    return metrics
