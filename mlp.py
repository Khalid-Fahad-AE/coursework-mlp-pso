"""MLP component: activations, layers, network, and Builder."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from losses import BinaryCrossEntropy, CategoricalCrossEntropy, Loss


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    def forward(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))


class Tanh(ActivationFunction):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)


class Softmax(ActivationFunction):
    def forward(self, z: np.ndarray) -> np.ndarray:
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_values = np.exp(np.clip(shifted, -60.0, 60.0))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)


class Linear(ActivationFunction):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return z


class Layer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: ActivationFunction,
        rng: np.random.Generator,
    ) -> None:
        if input_size < 1 or output_size < 1:
            raise ValueError("Layer sizes must be positive.")
        self.weights = rng.normal(0.0, np.sqrt(2.0 / input_size), (input_size, output_size))
        self.biases = np.zeros(output_size, dtype=float)
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.activation.forward(x @ self.weights + self.biases)

    def get_parameters(self) -> np.ndarray:
        return np.concatenate((self.weights.ravel(), self.biases.ravel()))

    def set_parameters(self, parameters: np.ndarray) -> None:
        weight_count = self.weights.size
        expected = weight_count + self.biases.size
        if parameters.size != expected:
            raise ValueError(f"Expected {expected} layer parameters, got {parameters.size}.")
        self.weights = parameters[:weight_count].reshape(self.weights.shape).copy()
        self.biases = parameters[weight_count:].copy()

    def parameter_count(self) -> int:
        return self.weights.size + self.biases.size


class Network:
    def __init__(self) -> None:
        self.layers: list[Layer] = []

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        output = np.asarray(x, dtype=float)
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def get_parameters(self) -> np.ndarray:
        if not self.layers:
            return np.array([], dtype=float)
        return np.concatenate([layer.get_parameters() for layer in self.layers])

    def set_parameters(self, parameters: np.ndarray) -> None:
        parameters = np.asarray(parameters, dtype=float)
        expected = self.count_parameters()
        if parameters.size != expected:
            raise ValueError(f"Expected {expected} network parameters, got {parameters.size}.")
        start = 0
        for layer in self.layers:
            end = start + layer.parameter_count()
            layer.set_parameters(parameters[start:end])
            start = end

    def count_parameters(self) -> int:
        return sum(layer.parameter_count() for layer in self.layers)


@dataclass
class MLPConfig:
    hidden_layers: list[int]
    activations: list[str]
    task: str
    seed: int = 42


class MLPBuilder:
    def __init__(self) -> None:
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.config: MLPConfig | None = None

    def set_dataset(self, x: np.ndarray, y: np.ndarray) -> "MLPBuilder":
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        return self

    def set_configuration(self, config: dict[str, Any]) -> "MLPBuilder":
        self.config = MLPConfig(**config)
        if len(self.config.hidden_layers) != len(self.config.activations):
            raise ValueError("One activation is required for every hidden layer.")
        if any(size < 1 for size in self.config.hidden_layers):
            raise ValueError("Hidden-layer sizes must be positive.")
        if self.config.task not in {"binary_classification", "multiclass_classification"}:
            raise ValueError("Task must be binary_classification or multiclass_classification.")
        return self

    def determine_input_size(self) -> int:
        if self.x is None or self.x.ndim != 2:
            raise ValueError("A two-dimensional dataset must be set before build().")
        return self.x.shape[1]

    def determine_output_size(self) -> int:
        if self.y is None:
            raise ValueError("A target array must be set before build().")
        if self.config is None:
            raise ValueError("A configuration must be set before build().")
        if self.config.task == "binary_classification":
            return 1
        if self.y.ndim == 2:
            return self.y.shape[1]
        return int(np.unique(self.y).size)

    def _activation(self, name: str) -> ActivationFunction:
        activations = {"sigmoid": Sigmoid, "tanh": Tanh, "linear": Linear}
        try:
            return activations[name.lower()]()
        except KeyError as exc:
            raise ValueError(f"Unsupported hidden activation: {name}") from exc

    def build(self) -> tuple[Network, Loss]:
        if self.config is None or self.x is None or self.y is None:
            raise ValueError("Set dataset and configuration before build().")
        rng = np.random.default_rng(self.config.seed)
        network = Network()
        sizes = [self.determine_input_size(), *self.config.hidden_layers, self.determine_output_size()]
        for index in range(len(sizes) - 1):
            is_output = index == len(sizes) - 2
            activation = (
                Sigmoid()
                if self.config.task == "binary_classification" and is_output
                else Softmax()
                if is_output
                else self._activation(self.config.activations[index])
            )
            network.add_layer(Layer(sizes[index], sizes[index + 1], activation, rng))
        loss: Loss = (
            BinaryCrossEntropy()
            if self.config.task == "binary_classification"
            else CategoricalCrossEntropy()
        )
        return network, loss
