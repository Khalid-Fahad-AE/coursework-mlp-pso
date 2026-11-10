"""Dataset loading and one-hot encoding helpers."""

import numpy as np
from sklearn.datasets import load_iris


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    dataset = load_iris()
    return dataset.data.astype(float), dataset.target.astype(int)


def one_hot(labels: np.ndarray, class_count: int) -> np.ndarray:
    encoded = np.zeros((labels.size, class_count), dtype=float)
    encoded[np.arange(labels.size), labels.astype(int)] = 1.0
    return encoded
