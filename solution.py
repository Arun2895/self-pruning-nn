"""
Self-Pruning Neural Network — Production Implementation
Tredence Analytics Case Study - AI Engineer

This script implements a dynamic pruning mechanism where the network learns
to remove its own connections during training via gated weights and L1 regularization.
"""

import os
import random
import logging
import argparse
import warnings
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# --- 1. GLOBAL CONFIGURATION & REPRODUCIBILITY ---

# Suppress the NumPy 2.4 / Torchvision visible deprecation warning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*dtype.*align.*")

def seed_everything(seed: int = 42):
    """Ensure fully reproducible results."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Setup professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 2. CORE MODULES ---

class PrunableLinear(nn.Module):
    """
    Custom Linear layer with learnable sigmoid gates for dynamic pruning.
    
    Formula: output = (Weight * Sigmoid(Gate_Scores)) @ Input + Bias
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard Weight and Bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores (initialized to 0 so sigmoid(0) = 0.5)
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.zeros_(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform scores to gates [0, 1]
        gates = torch.sigmoid(self.gate_scores)
        
        # Apply gating to weights (element-wise multiplication)
        # Gradient flows through both self.weight and self.gate_scores
        pruned_weights = self.weight * gates
        
        return F.linear(x, pruned_weights, self.bias)

    def get_sparsity(self, threshold: float = 1e-2) -> float:
        """Calculates percentage of weights pruned in this layer."""
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            return (gates < threshold).float().mean().item()

    def get_gate_values(self) -> torch.Tensor:
        """Returns flattened gate values (sigmoid outputs)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).cpu().flatten()


class SelfPruningNet(nn.Module):
    """
    Deep Feed-Forward Network composed of PrunableLinear layers.
    Architecture: [3072] -> 1024 -> 512 -> 256 -> [10]
    Includes BatchNorm and Dropout for enhanced stability and accuracy.
    """
    def __init__(self, input_dim: int = 3072, num_classes: int = 10):
        super().__init__()
        
        # Layers
        self.fc1 = PrunableLinear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = PrunableLinear(256, num_classes)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        
        return self.fc4(x)

    def sparsity_loss(self) -> torch.Tensor:
        """L1 Norm of all gate values across the network."""
        layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        return sum(torch.sigmoid(l.gate_scores).sum() for l in layers)

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Mean sparsity across all prunable layers."""
        layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        return sum(l.get_sparsity(threshold) for l in layers) / len(layers)

    def all_gates(self) -> torch.Tensor:
        """Concatenates all gate values for global distribution analysis."""
        layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        return torch.cat([l.get_gate_values() for l in layers])

# --- 3. DATA & TRAINING PIPELINE ---

def get_loaders(data_dir: str, batch_size: int):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader

def run_experiment(lam: float, epochs: int, device: torch.device, loaders: Tuple[DataLoader, DataLoader]):
    """Trains and evaluates the model for a specific lambda value."""
    logger.info(f"\n>>> STARTING EXPERIMENT: Lambda = {lam} <<<")
    
    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Cosine Annealing for smoother convergence and higher accuracy
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    train_loader, test_loader = loaders
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            
            # Hybrid Loss: Classification + Lambda * Sparsity
            ce_loss = criterion(logits, labels)
            sparse_loss = model.sparsity_loss()
            total_loss = ce_loss + lam * sparse_loss
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
            
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
            
        scheduler.step()
        train_acc = correct / total
        
        # Validation at end of epoch
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                val_correct += (model(images).argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        sparsity = model.overall_sparsity()
        
        logger.info(f"Epoch {epoch:2d} | Loss: {running_loss/total:.4f} | Train: {train_acc:.2%} | Val: {val_acc:.2%} | Sparsity: {sparsity:.2%}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_model_lam_{lam}.pth")

    final_sparsity = model.overall_sparsity()
    logger.info(f"DONE: Lambda {lam} | Best Val Acc: {best_acc:.2%} | Final Sparsity: {final_sparsity:.2%}")
    return best_acc, final_sparsity, model

# --- 4. MAIN ENTRY POINT ---

def main():
    parser = argparse.ArgumentParser(description="Self-Pruning NN Experiment")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")

    loaders = get_loaders(args.data_dir, args.batch_size)
    lambda_values = [1e-5, 1e-4, 1e-3]
    
    results = []
    best_model_overall = None
    
    for lam in lambda_values:
        acc, spar, model = run_experiment(lam, args.epochs, device, loaders)
        results.append({"lambda": lam, "accuracy": acc, "sparsity": spar})
        if lam == 1e-4: # Typically the best trade-off model for visualization
            best_model_overall = model

    # --- 5. FINAL REPORTING ---
    
    logger.info("\n" + "="*45)
    logger.info(f"{'Lambda':<12} | {'Test Accuracy':>12} | {'Sparsity (%)':>12}")
    logger.info("-" * 45)
    for res in results:
        logger.info(f"{res['lambda']:<12.0e} | {res['accuracy']:>12.2%} | {res['sparsity']:>12.2%}")
    logger.info("="*45)

    # Visualization of Gate Distribution
    if best_model_overall:
        gates = best_model_overall.all_gates().numpy()
        plt.figure(figsize=(10, 6))
        plt.hist(gates, bins=100, color='royalblue', edgecolor='black', alpha=0.8)
        plt.axvline(x=1e-2, color='crimson', linestyle='--', linewidth=2, label='Pruning Threshold (1e-2)')
        plt.title(f"Final Gate Score Distribution (Lambda = 1e-4)", fontsize=14)
        plt.xlabel("Gate Value (Sigmoid Output)", fontsize=12)
        plt.ylabel("Frequency (Number of Weights)", fontsize=12)
        plt.yscale('log') # Log scale helps see the distribution across magnitudes
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig("gate_distribution.png", dpi=300)
        logger.info("\nVisualization saved to 'gate_distribution.png'")

if __name__ == "__main__":
    main()
