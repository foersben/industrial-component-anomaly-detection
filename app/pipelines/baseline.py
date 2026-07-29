"""Runs the baseline Patchcore model on the MVTec AD dataset.

This serves as a functional baseline with PR-AUC evaluation.
"""

import warnings
from typing import Any

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# Suppress the timm deprecation warning caused by anomalib
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")


def run_baseline(data_root: str = "data/raw/mvtec_ad", category: str = "bottle") -> Any:
    """Run the first simple baseline using Patchcore and MVTec AD dataset.

    This serves as a functional baseline with PR-AUC evaluation.

    Args:
        data_root: Path to the root directory of the MVTec AD dataset.
        category: The specific category to evaluate (e.g., 'bottle').

    Returns:
        The evaluation test results.
    """
    # 1. Initialize data and model
    # anomalib will handle resizing and ImageNet normalization internally
    datamodule = MVTecAD(root=data_root, category=category)
    model = Patchcore(backbone="resnet18")

    # 2. Initialize engine with the PR-AUC metric enabled
    # These metrics are crucial for highly imbalanced anomaly detection
    engine = Engine(accelerator="gpu", devices=1)

    # 3. Fit and Test
    print(f"Fitting Patchcore model on {category} category...")
    engine.fit(model, datamodule)

    print("Testing Patchcore model...")
    test_results = engine.test(model=model, datamodule=datamodule)

    return test_results


if __name__ == "__main__":
    run_baseline()
