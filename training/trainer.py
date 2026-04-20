"""Training and evaluation logic."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ── Data ──────────────────────────────────────────────────────────────────────

def get_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    """Return CIFAR-10 train and test DataLoaders."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model, device, lam: float, lr: float = 1e-3):
        self.model  = model.to(device)
        self.device = device
        self.lam    = lam                           # sparsity regularization weight
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── single epoch ──────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)

            ce_loss      = self.criterion(logits, labels)
            sparse_loss  = self.model.sparsity_loss()
            loss         = ce_loss + self.lam * sparse_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)

        return total_loss / total, correct / total

    # ── evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            preds   = self.model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)

        return correct / total

    # ── full training run ─────────────────────────────────────────────────────

    def fit(self, train_loader: DataLoader, test_loader: DataLoader,
            epochs: int) -> dict:
        """Train for `epochs` epochs; return final metrics."""
        print(f"\n{'─'*50}")
        print(f"  λ = {self.lam:.0e}  |  epochs = {epochs}")
        print(f"{'─'*50}")

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self.train_epoch(train_loader)
            if epoch % 5 == 0 or epoch == 1:
                val_acc = self.evaluate(test_loader)
                sparsity = self.model.overall_sparsity()
                print(f"  Ep {epoch:>2}/{epochs}  "
                      f"loss={tr_loss:.4f}  "
                      f"train_acc={tr_acc:.3f}  "
                      f"val_acc={val_acc:.3f}  "
                      f"sparsity={sparsity:.1%}")

        final_acc      = self.evaluate(test_loader)
        final_sparsity = self.model.overall_sparsity()
        print(f"\n  ✓ Final → acc={final_acc:.3f}  sparsity={final_sparsity:.1%}")
        return {"lambda": self.lam, "accuracy": final_acc, "sparsity": final_sparsity}
