"""
train.py
========
Training and Evaluation Routines for LeNet-5
Reference: LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998)

This module exposes two reusable functions that encapsulate the training loop and
evaluation logic, keeping the Jupyter Notebook a clean presentation layer.
"""

import torch


def train_model(model, train_loader, criterion, optimizer, epochs, device):
    """
    Execute the standard mini-batch Stochastic Gradient Descent (SGD) training loop.

    At each step, we:
      1. Forward pass: compute f(x; θ) — the model's current predictions.
      2. Compute loss: L(f(x;θ), y) via Cross-Entropy.
      3. Backward pass: compute ∂L/∂θ for all parameters via backpropagation.
      4. Parameter update: θ ← θ - η·∇L  (handled by the Adam optimizer).

    Args:
        model     : The LeNet-5 model instance (nn.Module).
        train_loader: DataLoader yielding (images, labels) batches.
        criterion : Loss function (e.g., nn.CrossEntropyLoss).
        optimizer : Optimizer instance (e.g., torch.optim.Adam).
        epochs    : Number of full passes over the training dataset.
        device    : torch.device — 'cuda' or 'cpu'.

    Returns:
        list[float]: Average cross-entropy loss per epoch, for plotting.
    """
    losses = []

    for epoch in range(epochs):
        # Set model to training mode — enables dropout, batch norm gradients, etc.
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            # Move batch tensors to the target compute device
            images, labels = images.to(device), labels.to(device)

            # Step 1 — Forward pass: compute logits
            outputs = model(images)

            # Step 2 — Compute Cross-Entropy Loss:
            # L = -1/N * Σ y_i * log(softmax(ŷ_i))
            # nn.CrossEntropyLoss applies log-softmax internally for numerical stability.
            loss = criterion(outputs, labels)

            # Step 3 — Zero gradients to prevent accumulation across batches
            optimizer.zero_grad()

            # Step 4 — Backward pass: compute gradient ∂L/∂θ for all parameters
            loss.backward()

            # Step 5 — Parameter update using the Adam adaptive learning rule
            optimizer.step()

            running_loss += loss.item()

        # Epoch-level average loss: measures convergence across the full dataset
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    return losses


def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model on a held-out test dataset.

    The model is set to evaluation mode, which disables gradient computation,
    dropout, and batch normalization updates. Classification is performed by
    selecting the class with the highest logit (argmax of the output).

    Args:
        model       : The trained LeNet-5 model instance (nn.Module).
        test_loader : DataLoader yielding (images, labels) for the test set.
        device      : torch.device — 'cuda' or 'cpu'.

    Returns:
        tuple:
            - accuracy (float): Top-1 classification accuracy in percent.
            - all_preds (list[int]): All predicted class indices.
            - all_labels (list[int]): All ground-truth class indices.
    """
    # Set model to inference mode — disables gradient tracking for efficiency
    model.eval()

    correct, total = 0, 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass — compute raw logits
            outputs = model(images)

            # Predicted class = argmax of logits (equivalent to argmax of softmax)
            _, predicted = torch.max(outputs, dim=1)

            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    return accuracy, all_preds, all_labels
