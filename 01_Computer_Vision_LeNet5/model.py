"""
model.py
========
LeNet-5 Architecture Definition
Reference: LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998)

This module defines the LeNet-5 CNN as a standalone, importable PyTorch class,
separating the architecture from the training and presentation layers.
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    PyTorch implementation of the classic LeNet-5 architecture.

    Originally proposed by Yann LeCun et al. (1998) for optical character
    recognition, it is widely regarded as the seminal Convolutional Neural
    Network (CNN) architecture.

    Architecture Summary (Table 1 of the original paper):
    -------------------------------------------------------
    Layer | Type          | Output Shape   | Parameters
    ------|---------------|----------------|------------------
    Input | -             | (1, 32, 32)    | -
    C1    | Conv2d        | (6, 28, 28)    | 156
    S2    | AvgPool2d     | (6, 14, 14)    | 0
    C3    | Conv2d        | (16, 10, 10)   | 2,416
    S4    | AvgPool2d     | (16, 5, 5)     | 0
    C5    | Linear        | (120,)         | 48,120
    F6    | Linear        | (84,)          | 10,164
    Out   | Linear        | (10,)          | 850
    """

    def __init__(self):
        super().__init__()

        # ------------------------------------------------------------------ #
        # C1: First Convolutional Layer                                        #
        # Input: (1, 32, 32) — 1-channel 32x32 greyscale image.               #
        # Filter: 6 filters of size 5x5, no padding.                          #
        # Mathematical intuition: Each filter performs a 2D cross-correlation  #
        # (S)(i,j) = (X * W)(i,j), learning basic spatial features such as    #
        # edges and corners. Output width = (32 - 5 + 1) = 28.                #
        # Output shape: (6, 28, 28)                                           #
        # ------------------------------------------------------------------ #
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)

        # ------------------------------------------------------------------ #
        # S2: First Subsampling (Average Pooling) Layer                       #
        # Kernel 2x2, stride 2 — downsamples each feature map by factor of 2. #
        # Mathematical intuition: Reduces spatial resolution while preserving  #
        # the most prominent features, providing local translation invariance. #
        # Output shape: (6, 14, 14)                                           #
        # ------------------------------------------------------------------ #
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # ------------------------------------------------------------------ #
        # C3: Second Convolutional Layer                                       #
        # Filter: 16 filters of size 5x5 applied to 6 input channels.         #
        # Mathematical intuition: Learns complex compositions of the lower-    #
        # level features (curves, loops, junctions) detected by C1.           #
        # Note: The original paper used a specific sparse connection table     #
        # between C1 and C3 to break symmetry, but modern PyTorch uses a full #
        # connection for simplicity without significant accuracy loss.         #
        # Output shape: (16, 10, 10)                                          #
        # ------------------------------------------------------------------ #
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # ------------------------------------------------------------------ #
        # S4: Second Subsampling (Average Pooling) Layer                      #
        # Output shape: (16, 5, 5)                                            #
        # ------------------------------------------------------------------ #
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # ------------------------------------------------------------------ #
        # C5 / FC1: First Fully Connected Layer                               #
        # Input: 16 × 5 × 5 = 400 units (flattened spatial features).         #
        # Mathematical intuition: Integrates all spatially distributed feature #
        # maps into a single 120-dimensional feature vector. Treated as a     #
        # convolutional layer in the original paper (with a 5x5 kernel that   #
        # exactly matches the S4 output size), but equivalent to nn.Linear.   #
        # Output: 120 units                                                   #
        # ------------------------------------------------------------------ #
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        # ------------------------------------------------------------------ #
        # F6: Second Fully Connected Layer                                    #
        # The value 84 was chosen in the original paper to correspond to a    #
        # 7×12 pixel bitmap representation of the ASCII character set.        #
        # Output: 84 units                                                    #
        # ------------------------------------------------------------------ #
        self.fc2 = nn.Linear(120, 84)

        # ------------------------------------------------------------------ #
        # Output Layer: Classification Head                                   #
        # The original paper used Euclidean RBF units here. This modern       #
        # implementation uses a standard linear layer. The Softmax activation #
        # is applied implicitly by PyTorch's nn.CrossEntropyLoss during       #
        # training, and explicitly via torch.softmax during inference.        #
        # Output: 10 raw logits (one per MNIST digit class)                  #
        # ------------------------------------------------------------------ #
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Forward pass: maps input image tensor to class logits.

        Activation function: Hyperbolic Tangent (Tanh) is used throughout, as
        specified in the original 1998 paper. It is a zero-centered symmetric
        squashing function: f(x) = (e^x - e^{-x}) / (e^x + e^{-x}), which was
        shown to improve convergence speed compared to standard Sigmoid.

        Args:
            x (Tensor): Input image batch of shape (N, 1, 32, 32).

        Returns:
            Tensor: Raw class logits of shape (N, 10).
        """
        x = torch.tanh(self.conv1(x))   # C1 → Tanh
        x = self.pool1(x)               # S2
        x = torch.tanh(self.conv2(x))   # C3 → Tanh
        x = self.pool2(x)               # S4
        x = x.view(-1, 16 * 5 * 5)     # Flatten: (N, 400)
        x = torch.tanh(self.fc1(x))     # C5 → Tanh
        x = torch.tanh(self.fc2(x))     # F6 → Tanh
        x = self.fc3(x)                 # Output: raw logits (no activation)
        return x

    def get_feature_maps(self, x):
        """
        Extract the 6 feature maps produced by the C1 convolutional layer.
        Useful for interpretability and visualization of learned filters.

        Args:
            x (Tensor): A single input image of shape (1, 1, 32, 32).

        Returns:
            Tensor: Feature maps of shape (1, 6, 28, 28).
        """
        return torch.tanh(self.conv1(x))
