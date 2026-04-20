"""
Self-Pruning Neural Network — Entry Point
Trains SelfPruningNet on CIFAR-10 with multiple λ values.
"""

import torch
import config
from models   import SelfPruningNet
from training import Trainer, get_dataloaders
from utils    import plot_gate_histograms, print_results_table


def main():
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n  Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(
        config.DATA_DIR, config.BATCH_SIZE, config.NUM_WORKERS
    )

    # ── Experiments ───────────────────────────────────────────────────────────
    results      = []
    trained_models = {}

    for lam in config.LAMBDA_VALUES:
        model = SelfPruningNet(
            input_dim   = 32 * 32 * 3,
            hidden1     = config.HIDDEN_1,
            hidden2     = config.HIDDEN_2,
            num_classes = config.NUM_CLASSES,
        )

        trainer = Trainer(model, device, lam=lam, lr=config.LEARNING_RATE)
        metrics = trainer.fit(train_loader, test_loader, epochs=config.EPOCHS)

        results.append(metrics)
        trained_models[lam] = model

    # ── Report ────────────────────────────────────────────────────────────────
    print_results_table(results)

    # ── Visualize ─────────────────────────────────────────────────────────────
    plot_gate_histograms(trained_models, save_path="gate_histograms.png")


if __name__ == "__main__":
    main()
