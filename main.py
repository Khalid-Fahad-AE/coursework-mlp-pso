"""Run the coursework experiment and write metrics/convergence artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adapter import MLPFitnessAdapter
from data import load_dataset, one_hot
from mlp import MLPBuilder
from pso import NearestNeighbourInformantSelection, Population, RandomInformantSelection


def run_experiment(strategy_name: str, seed: int, x_train: np.ndarray, y_train: np.ndarray,
                   x_test: np.ndarray, y_test_labels: np.ndarray, class_count: int) -> dict:
    y_train_one_hot = one_hot(y_train, class_count)
    builder = MLPBuilder().set_dataset(x_train, y_train_one_hot).set_configuration({
        "hidden_layers": [8, 5],
        "activations": ["tanh", "sigmoid"],
        "task": "multiclass_classification",
        "seed": seed,
    })
    network, loss = builder.build()
    fitness = MLPFitnessAdapter(network, loss, x_train, y_train_one_hot)
    strategy = (
        RandomInformantSelection(np.random.default_rng(seed + 1000))
        if strategy_name == "random"
        else NearestNeighbourInformantSelection()
    )
    population = Population(
        fitness_function=fitness,
        informant_strategy=strategy,
        num_particles=20,
        num_informants=3,
        num_iterations=60,
        dimension=network.count_parameters(),
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        step_size=0.05,
        seed=seed,
    )
    best_position, best_loss = population.optimise()
    network.set_parameters(best_position)
    predicted_labels = np.argmax(network.predict(x_test), axis=1)
    return {
        "strategy": strategy_name,
        "best_training_loss": best_loss,
        "test_accuracy": accuracy_score(y_test_labels, predicted_labels),
        "test_precision": precision_score(y_test_labels, predicted_labels, average="macro", zero_division=0),
        "test_recall": recall_score(y_test_labels, predicted_labels, average="macro", zero_division=0),
        "test_f1": f1_score(y_test_labels, predicted_labels, average="macro", zero_division=0),
        "history": population.history,
    }


def main() -> None:
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    results = [
        run_experiment(name, 42, x_train, y_train, x_test, y_test, int(np.unique(y).size))
        for name in ("random", "nearest")
    ]
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    fieldnames = ["strategy", "best_training_loss", "test_accuracy", "test_precision", "test_recall", "test_f1"]
    with (output_dir / "experiment_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: result[key] for key in fieldnames} for result in results)
    plt.figure(figsize=(8, 5))
    for result in results:
        plt.plot(result["history"], label=result["strategy"])
    plt.xlabel("Iteration")
    plt.ylabel("Best training loss")
    plt.title("PSO convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "convergence.png", dpi=150)
    for result in results:
        print(
            f"{result['strategy']:>7} | loss={result['best_training_loss']:.5f} | "
            f"accuracy={result['test_accuracy']:.3f} | precision={result['test_precision']:.3f} | "
            f"recall={result['test_recall']:.3f} | f1={result['test_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
