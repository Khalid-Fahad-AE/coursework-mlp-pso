"""Run the coursework experiment and write metrics/convergence artifacts."""

from __future__ import annotations

import argparse
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


def run_experiment(
    strategy_name: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    class_count: int,
) -> dict:
    y_train_for_network = (
        y_train.reshape(-1, 1)
        if task == "binary_classification"
        else one_hot(y_train, class_count)
    )
    builder = MLPBuilder().set_dataset(x_train, y_train_for_network).set_configuration({
        "hidden_layers": [8, 5],
        "activations": ["tanh", "sigmoid"],
        "task": task,
        "seed": seed,
    })
    network, loss = builder.build()
    fitness = MLPFitnessAdapter(network, loss, x_train, y_train_for_network)
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
    train_predictions = network.predict(x_train)
    val_predictions = network.predict(x_val)
    test_predictions = network.predict(x_test)
    if task == "binary_classification":
        predicted_labels = (test_predictions.ravel() >= 0.5).astype(int)
        val_labels = (val_predictions.ravel() >= 0.5).astype(int)
        train_labels = (train_predictions.ravel() >= 0.5).astype(int)
    else:
        predicted_labels = np.argmax(test_predictions, axis=1)
        val_labels = np.argmax(val_predictions, axis=1)
        train_labels = np.argmax(train_predictions, axis=1)
    return {
        "strategy": strategy_name,
        "best_training_loss": best_loss,
        "train_accuracy": accuracy_score(y_train, train_labels),
        "val_accuracy": accuracy_score(y_val, val_labels),
        "test_accuracy": accuracy_score(y_test, predicted_labels),
        "test_precision": precision_score(y_test, predicted_labels, average="macro", zero_division=0),
        "test_recall": recall_score(y_test, predicted_labels, average="macro", zero_division=0),
        "test_f1": f1_score(y_test, predicted_labels, average="macro", zero_division=0),
        "history": population.history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to data_banknote_authentication.txt; omit for the Iris demonstration.",
    )
    args = parser.parse_args()
    x, y = load_dataset(args.data)
    task = "binary_classification" if np.unique(y).size == 2 else "multiclass_classification"
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    results = [
        run_experiment(
            name, 42, x_train, y_train, x_val, y_val, x_test, y_test,
            task, int(np.unique(y).size)
        )
        for name in ("random", "nearest")
    ]
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    fieldnames = [
        "strategy", "best_training_loss", "train_accuracy", "val_accuracy",
        "test_accuracy", "test_precision", "test_recall", "test_f1",
    ]
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
            f"train={result['train_accuracy']:.3f} | val={result['val_accuracy']:.3f} | "
            f"accuracy={result['test_accuracy']:.3f} | precision={result['test_precision']:.3f} | "
            f"recall={result['test_recall']:.3f} | f1={result['test_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
