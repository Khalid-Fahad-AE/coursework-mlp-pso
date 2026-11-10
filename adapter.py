"""Adapter between the generic PSO fitness interface and the MLP."""

import numpy as np

from losses import Loss
from mlp import Network
from pso import FitnessFunction


class MLPFitnessAdapter(FitnessFunction):
    def __init__(self, network: Network, loss: Loss, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.network = network
        self.loss = loss
        self.x_train = x_train
        self.y_train = y_train

    def evaluate(self, position: np.ndarray) -> float:
        self.network.set_parameters(position)
        predictions = self.network.forward(self.x_train)
        return self.loss.compute(self.y_train, predictions)
