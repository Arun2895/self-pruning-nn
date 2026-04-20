"""Custom prunable linear layer and feedforward network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    Linear layer with learnable gate scores.
    Gates suppress unnecessary weights during training itself.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard learnable parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores — same shape as weight, learned independently
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        # Init gate_scores near 0 → sigmoid ≈ 0.5 (neutral start)
        nn.init.zeros_(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Soft binary mask in [0, 1]; gradients flow through sigmoid
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise mask: suppresses weights the network finds useless
        pruned_weights = self.weight * gates

        return F.linear(x, pruned_weights, self.bias)

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below threshold (effectively pruned)."""
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            return (gates < threshold).float().mean().item()

    def gate_values(self) -> torch.Tensor:
        """Return detached gate values for analysis/visualization."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).cpu().flatten()


class SelfPruningNet(nn.Module):
    """
    Feedforward network built entirely from PrunableLinear layers.
    Input (3072) → FC1 (512) → ReLU → FC2 (256) → ReLU → FC3 (10)
    """

    def __init__(self, input_dim: int = 3072, hidden1: int = 512,
                 hidden2: int = 256, num_classes: int = 10):
        super().__init__()

        self.fc1 = PrunableLinear(input_dim, hidden1)
        self.fc2 = PrunableLinear(hidden1, hidden2)
        self.fc3 = PrunableLinear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # flatten CIFAR-10 images
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)                  # logits; loss handles softmax

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of all gate values across all layers (sum, not mean)."""
        total = sum(
            torch.sigmoid(layer.gate_scores).sum()
            for layer in [self.fc1, self.fc2, self.fc3]
        )
        return total

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Average sparsity across all PrunableLinear layers."""
        values = [layer.sparsity(threshold) for layer in [self.fc1, self.fc2, self.fc3]]
        return sum(values) / len(values)

    def all_gate_values(self) -> torch.Tensor:
        """Concatenated gate values from all layers for histogram plotting."""
        return torch.cat([layer.gate_values() for layer in [self.fc1, self.fc2, self.fc3]])
