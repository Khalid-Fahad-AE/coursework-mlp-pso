"""Run the required 10-seed comparison for both PSO informant strategies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import load_dataset
from main import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
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

    results: list[dict[str, float | int | str]] = []
    histories: dict[str, list[list[float]]] = {"random": [], "nearest": []}
    for strategy in ("random", "nearest"):
        for seed in range(10):
            result = run_experiment(
                strategy, seed, x_train, y_train, x_val, y_val, x_test, y_test,
                task, class_count
            )
            histories[strategy].append(result["history"])
            results.append({
                "strategy": strategy,
                "seed": seed,
                **{key: result[key] for key in (
                    "best_training_loss", "train_accuracy", "val_accuracy",
                    "test_accuracy", "test_precision", "test_recall", "test_f1",
                )},
            })

    args.output.mkdir(exist_ok=True)
    metric_names = [
        "best_training_loss", "train_accuracy", "val_accuracy",
        "test_accuracy", "test_precision", "test_recall", "test_f1",
    ]
    with (args.output / "multi_seed_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["strategy", "seed", *metric_names])
        writer.writeheader()
        writer.writerows(results)

    summary_rows = []
    for strategy in ("random", "nearest"):
        subset = [row for row in results if row["strategy"] == strategy]
        summary = {"strategy": strategy}
        for metric in metric_names:
            values = np.array([float(row[metric]) for row in subset])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
        summary_rows.append(summary)
        print(
            f"{strategy:>7} | "
            f"loss={summary['best_training_loss_mean']:.5f} +/- {summary['best_training_loss_std']:.5f} | "
            f"accuracy={summary['test_accuracy_mean']:.3f} +/- {summary['test_accuracy_std']:.3f} | "
            f"f1={summary['test_f1_mean']:.3f} +/- {summary['test_f1_std']:.3f}"
        )

    summary_fields = ["strategy"] + [
        field for metric in metric_names for field in (f"{metric}_mean", f"{metric}_std")
    ]
    with (args.output / "multi_seed_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    plt.figure(figsize=(8, 5))
    iterations = np.arange(len(histories["random"][0]))
    for strategy, colour in (("random", "tab:blue"), ("nearest", "tab:orange")):
        values = np.asarray(histories[strategy])
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        plt.plot(iterations, mean, label=strategy, color=colour)
        plt.fill_between(iterations, mean - std, mean + std, color=colour, alpha=0.15)
    plt.xlabel("Iteration")
    plt.ylabel("Best training loss")
    plt.title("10-seed PSO convergence (mean +/- standard deviation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / "multi_seed_convergence.png", dpi=150)


if __name__ == "__main__":
    main()
