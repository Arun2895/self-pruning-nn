"""Central configuration for the Self-Pruning Neural Network project."""

DEVICE = "cuda"  # falls back to cpu in main.py

# Data
BATCH_SIZE = 128
NUM_WORKERS = 2
NUM_CLASSES = 10

# Architecture
HIDDEN_1 = 512
HIDDEN_2 = 256

# Training
EPOCHS = 15
LEARNING_RATE = 1e-3

# Sparsity experiments
LAMBDA_VALUES = [1e-5, 1e-4, 1e-3]

# Pruning threshold for sparsity evaluation
GATE_THRESHOLD = 1e-2

# Data path
DATA_DIR = "./data"
