"""
Self-Pruning Neural Network — Unified Implementation
Created for the Tredence Analytics Case Study.

This script contains:
1. PrunableLinear Layer (Custom gated weights)
2. SelfPruningNet (Model architecture)
3. Training & Evaluation Logic
4. Main Experiment Loop (CIFAR-10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────

BATCH_SIZE = 128
NUM_WORKERS = 2  # Increased for efficiency, set to 0 if Windows errors occur
EPOCHS = 35      # Increased for better accuracy
LEARNING_RATE = 1e-3
LAMBDA_VALUES = [1e-5, 1e-4, 1e-3]
GATE_THRESHOLD = 1e-2
DATA_DIR = "./data"

# ── 2. PRUNABLE LINEAR LAYER ──────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    Linear layer with learnable gate scores.
    Each weight is multiplied by a sigmoid gate in [0, 1].
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores — same shape as weight
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.zeros_(self.gate_scores) # Start near sigmoid(0) = 0.5 (neutral)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform scores to gates [0, 1]
        gates = torch.sigmoid(self.gate_scores)
        # Apply gating to weights
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below activation threshold."""
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            return (gates < threshold).float().mean().item()

    def gate_values(self) -> torch.Tensor:
        """Flattened gate values for plotting."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).cpu().flatten()

# ── 3. SELF-PRUNING NETWORK ───────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feedforward network using only PrunableLinear layers.
    Architecture: [3072] -> 512 -> 256 -> [10]
    """
    def __init__(self, input_dim: int = 32*32*3, hidden1: int = 512, 
                 hidden2: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc1 = PrunableLinear(input_dim, hidden1)
        self.fc2 = PrunableLinear(hidden1, hidden2)
        self.fc3 = PrunableLinear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def sparsity_loss(self) -> torch.Tensor:
        """Sum of all gate values (L1 norm of sigmoids)."""
        return sum(torch.sigmoid(l.gate_scores).sum() 
                   for l in [self.fc1, self.fc2, self.fc3])

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Mean sparsity across all layers."""
        return sum(l.sparsity(threshold) for l in [self.fc1, self.fc2, self.fc3]) / 3

    def all_gate_values(self) -> torch.Tensor:
        """Concatenated gates for visualization."""
        return torch.cat([l.gate_values() for l in [self.fc1, self.fc2, self.fc3]])

# ── 4. TRAINING LOGIC ─────────────────────────────────────────────────────────

def get_dataloaders():
    # Standard CIFAR-10 Augmentation
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_ds = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    
    train_l = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_l  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_l, test_l

def train_and_eval(lam, device):
    print(f"\n--- Running Experiment: Lambda = {lam} ---")
    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Learning Rate Scheduler for smoother convergence
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_dataloaders()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits = model(images)
            ce_loss = criterion(logits, labels)
            sparse_loss = model.sparsity_loss()
            
            loss = ce_loss + lam * sparse_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
        
        scheduler.step()
        # Progress Print
        print(f"  Epoch {epoch:2d}/{EPOCHS} | Loss: {total_loss/total:.4f} | Train Acc: {correct/total:.2%}")

        # Final evaluation
        if epoch == EPOCHS:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    val_correct += (model(images).argmax(1) == labels).sum().item()
                    val_total += labels.size(0)
            
            acc = val_correct / val_total
            sparsity = model.overall_sparsity(GATE_THRESHOLD)
            print(f"  >> Final Result -> Accuracy: {acc:.2%} | Sparsity: {sparsity:.2%}")
            return acc, sparsity, model

    return 0, 0, model

# ── 5. MAIN EXECUTION ─────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []
    trained_models = {}

    for lam in LAMBDA_VALUES:
        acc, spar, model = train_and_eval(lam, device)
        results.append({"lambda": lam, "accuracy": acc, "sparsity": spar})
        trained_models[lam] = model

    # Print Table
    print("\n" + "="*40)
    print(f"{'Lambda':<10} | {'Accuracy':>10} | {'Sparsity':>10}")
    print("-" * 40)
    for res in results:
        print(f"{res['lambda']:<10.0e} | {res['accuracy']:>10.2%} | {res['sparsity']:>10.2%}")
    print("="*40)

    # Plot Best Model (Highest Lambda for clear sparsity visual or Medium)
    best_lam = LAMBDA_VALUES[-1]
    gates = trained_models[best_lam].all_gate_values().numpy()
    
    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=GATE_THRESHOLD, color='red', linestyle='--', label='Pruning Threshold')
    plt.title(f"Gate Value Distribution (Lambda = {best_lam})")
    plt.xlabel("Gate Value (sigmoid)")
    plt.ylabel("Weight Count")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("gate_distribution.png")
    print("\nVisualization saved to 'gate_distribution.png'")
    # plt.show() # Uncomment if running locally

if __name__ == "__main__":
    main()
