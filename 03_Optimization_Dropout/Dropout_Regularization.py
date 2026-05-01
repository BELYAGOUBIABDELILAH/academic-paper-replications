"""
Replication Study: Dropout — A Simple Way to Prevent Neural Networks from Overfitting
Srivastava et al., 2014  |  PyTorch implementation
"""

# Cell 2: Imports & Reproducibility
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("PyTorch:", torch.__version__)

# Cell 3: Dataset Generation
# ── Dataset ───────────────────────────────────────────────────────────────────
X, y = make_moons(n_samples=1000, noise=0.3, random_state=SEED)

# Force overfitting: only 100 train samples, 900 test samples
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.9, stratify=y, random_state=SEED
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
y_test  = torch.tensor(y_test.reshape(-1, 1),  dtype=torch.float32)

print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape,  y_test.shape)

# Cell 4 (Visualizing data) is skipped here because it's a notebook-only plot.

# Cell 5: Model Definitions
# ── Model definitions ─────────────────────────────────────────────────────────
class StandardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 1),
        )

    def forward(self, x):
        return self.net(x)


class DropoutMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 200), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(200, 200), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(200, 1),
        )

    def forward(self, x):
        return self.net(x)


model_A = StandardMLP()
model_B = DropoutMLP()


# Cell 6: Training Setup
# ── Training setup ────────────────────────────────────────────────────────────
criterion   = nn.BCEWithLogitsLoss()
optimizer_A = optim.Adam(model_A.parameters(), lr=0.01)
optimizer_B = optim.Adam(model_B.parameters(), lr=0.01)

EPOCHS = 500
train_loss_A, test_loss_A = [], []
train_loss_B, test_loss_B = [], []
train_acc_A,  test_acc_A  = [], []
train_acc_B,  test_acc_B  = [], []


def accuracy_from_logits(logits, y_true):
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == y_true).float().mean().item()



# Cell 7: Training Loop
# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    # Standard model
    model_A.train()
    optimizer_A.zero_grad()
    loss_A = criterion(model_A(X_train), y_train)
    loss_A.backward()
    optimizer_A.step()

    # Dropout model  (Dropout ON during training)
    model_B.train()
    optimizer_B.zero_grad()
    loss_B = criterion(model_B(X_train), y_train)
    loss_B.backward()
    optimizer_B.step()

    # Evaluation  (Dropout OFF)
    model_A.eval()
    model_B.eval()
    with torch.no_grad():
        tla = model_A(X_train);  tea = model_A(X_test)
        tlb = model_B(X_train);  teb = model_B(X_test)

        train_loss_A.append(criterion(tla, y_train).item())
        test_loss_A.append(criterion(tea, y_test).item())
        train_loss_B.append(criterion(tlb, y_train).item())
        test_loss_B.append(criterion(teb, y_test).item())

        train_acc_A.append(accuracy_from_logits(tla, y_train))
        test_acc_A.append(accuracy_from_logits(tea, y_test))
        train_acc_B.append(accuracy_from_logits(tlb, y_train))
        test_acc_B.append(accuracy_from_logits(teb, y_test))

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"A Train/Test Loss: {train_loss_A[-1]:.4f}/{test_loss_A[-1]:.4f} | "
            f"B Train/Test Loss: {train_loss_B[-1]:.4f}/{test_loss_B[-1]:.4f}"
        )

# Cell 8: Loss Comparison Plot
# ── Loss comparison plot ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

axes[0].plot(train_loss_A, label="Train Loss", color="#0f766e", linewidth=2)
axes[0].plot(test_loss_A,  label="Test Loss",  color="#be123c", linewidth=2)
axes[0].set_title("Without Regularization (Overfitting)", fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].legend(frameon=False)

axes[1].plot(train_loss_B, label="Train Loss", color="#0f766e", linewidth=2)
axes[1].plot(test_loss_B,  label="Test Loss",  color="#2563eb", linewidth=2)
axes[1].set_title("With Dropout (Generalized)", fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].legend(frameon=False)

fig.suptitle("Dropout Regularization on Noisy make_moons Data", fontsize=14, fontweight="bold")
plt.tight_layout()

# Guard: create assets/ directory if it doesn't exist (safe on Colab & locally)
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/loss_comparison.png", dpi=160, bbox_inches="tight")
plt.show()

# Cell 9: Final Metrics
# ── Final metrics ─────────────────────────────────────────────────────────────
print(f"Standard MLP  -> Final Train Loss: {train_loss_A[-1]:.4f}")
print(f"Standard MLP  -> Final Test  Loss: {test_loss_A[-1]:.4f}")
print(f"Standard MLP  -> Final Train Acc : {train_acc_A[-1]:.3f}")
print(f"Standard MLP  -> Final Test  Acc : {test_acc_A[-1]:.3f}")
print()
print(f"Dropout MLP   -> Final Train Loss: {train_loss_B[-1]:.4f}")
print(f"Dropout MLP   -> Final Test  Loss: {test_loss_B[-1]:.4f}")
print(f"Dropout MLP   -> Final Train Acc : {train_acc_B[-1]:.3f}")
print(f"Dropout MLP   -> Final Test  Acc : {test_acc_B[-1]:.3f}")

# Cell 10: Decision Boundary Plots
# ── Decision boundaries ───────────────────────────────────────────────────────
def plot_decision_boundary(model, X, y, title):
    model.eval()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        probs = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, probs, levels=30, cmap="coolwarm", alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=20, edgecolor="k")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    # Save the decision boundary graph
    filename = f"{title.lower().replace(' ', '_').replace('-', '')}.png"
    # Clean up multiple underscores
    import re
    filename = re.sub(r'_+', '_', filename)
    plt.savefig(os.path.join("assets", filename), dpi=160, bbox_inches="tight")
    
    plt.show()

plot_decision_boundary(model_A, X, y, "Decision Boundary - Standard MLP")
plot_decision_boundary(model_B, X, y, "Decision Boundary - Dropout MLP")

