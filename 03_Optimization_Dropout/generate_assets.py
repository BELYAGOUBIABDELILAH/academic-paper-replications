import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

os.makedirs("assets", exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

X, y = make_moons(n_samples=1000, noise=0.3, random_state=SEED)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.9, stratify=y, random_state=SEED
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
y_test  = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

class StandardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
    def forward(self, x):
        return self.net(x)

class DropoutMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 200),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(200, 1)
        )
    def forward(self, x):
        return self.net(x)

model_A = StandardMLP()
model_B = DropoutMLP()

criterion = nn.BCEWithLogitsLoss()
optimizer_A = optim.Adam(model_A.parameters(), lr=0.01)
optimizer_B = optim.Adam(model_B.parameters(), lr=0.01)

EPOCHS = 500
train_loss_A, test_loss_A = [], []
train_loss_B, test_loss_B = [], []

for epoch in range(EPOCHS):
    model_A.train()
    optimizer_A.zero_grad()
    loss_A = criterion(model_A(X_train), y_train)
    loss_A.backward()
    optimizer_A.step()

    model_B.train()
    optimizer_B.zero_grad()
    loss_B = criterion(model_B(X_train), y_train)
    loss_B.backward()
    optimizer_B.step()

    model_A.eval()
    model_B.eval()
    with torch.no_grad():
        train_loss_A.append(criterion(model_A(X_train), y_train).item())
        test_loss_A.append(criterion(model_A(X_test), y_test).item())
        train_loss_B.append(criterion(model_B(X_train), y_train).item())
        test_loss_B.append(criterion(model_B(X_test), y_test).item())

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

axes[0].plot(train_loss_A, label="Train Loss", color="#0f766e", linewidth=2)
axes[0].plot(test_loss_A, label="Test Loss", color="#be123c", linewidth=2)
axes[0].set_title("Without Regularization (Overfitting)", fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].legend(frameon=False)

axes[1].plot(train_loss_B, label="Train Loss", color="#0f766e", linewidth=2)
axes[1].plot(test_loss_B, label="Test Loss", color="#2563eb", linewidth=2)
axes[1].set_title("With Dropout (Generalized)", fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].legend(frameon=False)

fig.suptitle("Dropout Regularization on Noisy make_moons Data", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/loss_comparison.png", dpi=160, bbox_inches="tight")

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
    import re
    filename = re.sub(r'_+', '_', filename)
    plt.savefig(os.path.join("assets", filename), dpi=160, bbox_inches="tight")

plot_decision_boundary(model_A, X, y, "Decision Boundary - Standard MLP")
plot_decision_boundary(model_B, X, y, "Decision Boundary - Dropout MLP")

