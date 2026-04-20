"""Visualization and results reporting utilities."""

import matplotlib.pyplot as plt
import torch


def plot_gate_histograms(models_dict: dict, save_path: str = "gate_histograms.png"):
    """
    Plot histogram of final gate values for each lambda.
    models_dict: {lambda_value: trained_model}
    """
    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (lam, model) in zip(axes, models_dict.items()):
        gates = model.all_gate_values().numpy()
        ax.hist(gates, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(x=1e-2, color="red", linestyle="--", linewidth=1.2, label="threshold")
        ax.set_title(f"λ = {lam:.0e}", fontsize=13)
        ax.set_xlabel("Gate Value", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("Distribution of Gate Values After Training", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Histogram saved → {save_path}")


def print_results_table(results: list[dict]):
    """Print a formatted results table."""
    header = f"\n{'Lambda':<12} {'Accuracy':>10} {'Sparsity':>10}"
    sep    = "─" * 34
    print(f"\n{'═'*34}")
    print("  EXPERIMENT RESULTS")
    print(f"{'═'*34}")
    print(header)
    print(sep)
    for r in results:
        print(f"  {r['lambda']:<10.0e}  {r['accuracy']:>8.2%}  {r['sparsity']:>8.2%}")
    print(f"{'═'*34}\n")
