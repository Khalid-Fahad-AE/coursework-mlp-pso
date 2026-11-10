"""Compare selected MLP and PSO hyperparameter configurations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import load_dataset
from main import run_experiment


CONFIGURATIONS = (
    {"name": "baseline", "hidden_layers": [8, 5], "num_particles": 20, "num_iterations": 60},
    {"name": "smaller_swarm", "hidden_layers": [8, 5], "num_particles": 10, "num_iterations": 60},
    {"name": "more_iterations", "hidden_layers": [8, 5], "num_particles": 20, "num_iterations": 100},
    {"name": "wider_mlp", "hidden_layers": [12, 8], "num_particles": 20, "num_iterations": 60},
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    x, y = load_dataset(args.data)
    class_count = int(np.unique(y).size)
    task = "binary_classification" if class_count == 2 else "multiclass_classification"
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

    rows = []
    for config in CONFIGURATIONS:
        for strategy in ("random", "nearest"):
            for seed in range(args.seeds):
                result = run_experiment(
                    strategy, seed, x_train, y_train, x_val, y_val, x_test, y_test,
                    task, class_count, config["hidden_layers"],
                    config["num_particles"], config["num_iterations"],
                )
                rows.append({
                    "configuration": config["name"],
                    "strategy": strategy,
                    "seed": seed,
                    "best_training_loss": result["best_training_loss"],
                    "val_accuracy": result["val_accuracy"],
                    "test_accuracy": result["test_accuracy"],
                    "test_precision": result["test_precision"],
                    "test_recall": result["test_recall"],
                    "test_f1": result["test_f1"],
                })

    args.output.mkdir(exist_ok=True)
    fields = list(rows[0])
    with (args.output / "hyperparameter_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_fields = ["configuration", "strategy", "runs"]
    metrics = [
        "best_training_loss", "val_accuracy", "test_accuracy",
        "test_precision", "test_recall", "test_f1",
    ]
    for metric in metrics:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    summaries = []
    for config in CONFIGURATIONS:
        for strategy in ("random", "nearest"):
            selected = [
                row for row in rows
                if row["configuration"] == config["name"] and row["strategy"] == strategy
            ]
            summary = {"configuration": config["name"], "strategy": strategy, "runs": len(selected)}
            for metric in metrics:
                values = np.array([float(row[metric]) for row in selected])
                summary[f"{metric}_mean"] = values.mean()
                summary[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            summaries.append(summary)
            print(
                f"{config['name']:>16} | {strategy:>7} | "
                f"accuracy={summary['test_accuracy_mean']:.3f} +/- "
                f"{summary['test_accuracy_std']:.3f} | "
                f"f1={summary['test_f1_mean']:.3f} +/- {summary['test_f1_std']:.3f}"
            )

    with (args.output / "hyperparameter_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
