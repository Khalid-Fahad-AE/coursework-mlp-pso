"""Loss-function hierarchy used by the MLP adapter."""

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError


class MSE(Loss):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


class BinaryCrossEntropy(Loss):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


class CategoricalCrossEntropy(Loss):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-12, 1.0)
        return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))
