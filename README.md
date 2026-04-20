# Self-Pruning Neural Network — CIFAR-10

This project implements a neural network that learns to prune its own weights during training using learnable sigmoid gates.

## 📝 Case Study Submission
For the **Tredence Analytics** case study, please refer to the following files:
*   **[solution.py](solution.py)**: The unified, high-performance implementation (Selection-Grade).
*   **[REPORT.md](REPORT.md)**: The formal analysis, results table, and technical trade-offs.

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
# Recommended Execution
python solution.py
```

## 🧠 How it Works
Traditional pruning happens *after* training. This network prunes *during* training by:
1.  **Gate scores**: Every weight has a learnable "score" parameter.
2.  **Sigmoid Mask**: Scores are passed through a sigmoid to create a mask in range $[0, 1]$.
3.  **Sparsity Penalty**: A penalty term ($ \lambda \sum \sigma(\text{score}) $) is added to the loss function, forcing the network to "turn off" unnecessary connections.

## 📊 Configuration
Adjust experiments in `config.py`:
- `LAMBDA_VALUES`: Control the tradeoff between accuracy and model size.
- `EPOCHS`: Duration of training.
- `HIDDEN_1 / HIDDEN_2`: Network width.

## 📈 Outputs
- **Console**: Final Accuracy and Sparsity percentages for each $\lambda$.
- **Visuals**: `gate_histograms.png` shows the distribution of gate values, highlighting how effectively neurons were suppressed.

---
*Created for experimenting with dynamic model compression.*
