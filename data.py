"""Dataset loading and target-encoding helpers."""

from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris


def load_iris_dataset() -> tuple[np.ndarray, np.ndarray]:
    dataset = load_iris()
    return dataset.data.astype(float), dataset.target.astype(int)


def load_banknote_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load UCI Banknote Authentication data.

    The file has four feature columns followed by a binary class column and
    normally has no header.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Banknote dataset not found: {path}. "
            "Place data_banknote_authentication.txt in the project folder."
        )
    values = np.loadtxt(path, delimiter=",", dtype=float)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("Expected five columns: four features followed by the label.")
    return values[:, :4], values[:, 4].astype(int)


def load_dataset(path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load Banknote data when a path is supplied; otherwise use Iris demo data."""
    return load_banknote_dataset(path) if path is not None else load_iris_dataset()


def one_hot(labels: np.ndarray, class_count: int) -> np.ndarray:
    encoded = np.zeros((labels.size, class_count), dtype=float)
    encoded[np.arange(labels.size), labels.astype(int)] = 1.0
    return encoded
