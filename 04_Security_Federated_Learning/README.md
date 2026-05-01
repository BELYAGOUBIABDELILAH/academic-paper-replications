# Federated Learning from Scratch
### Replicating McMahan et al. 2017 — FedAvg Algorithm

This repository contains a PyTorch implementation replicating the core findings of the foundational paper: **"Communication-Efficient Learning of Deep Networks from Decentralized Data"** by McMahan et al. (2017).

## 📌 Objective
To build a robust, research-grade simulation of Federated Learning, demonstrating how to train a shared global model across decentralized devices without transferring raw data to a central server. This simulation explicitly compares a naive distributed approach (`FedSGD`) against the paper's novel approach (`FedAvg`).

## 📊 Dataset & Partitioning Challenge
A key challenge in federated learning is that real-world device data is **Non-IID** (Independent and Identically Distributed). A user's phone contains data tailored to them, not the global average. 
This repository simulates and tests on both data splits using the MNIST dataset:
* **IID Split**: Data is shuffled and uniformly distributed.
* **Non-IID Split (Pathological)**: Data is sorted by label, and clients are restricted to seeing at most two digit classes.

![Data Distribution](assets/data_distribution.png)

## 🧠 Model Architecture (2NN)
We use a simple 2-layer Multi-Layer Perceptron (MLP) as described in the paper:
* **Input Layer:** 784 pixels
* **Hidden Layers:** Two layers of 200 units each with ReLU activations
* **Output Layer:** 10 units (digit classes)
* **Parameters:** ~199,000

## 🧪 Algorithms Simulated
1. **FedSGD (Baseline):** Each client computes a single gradient step. The server aggregates the gradients and updates the global model. Very communication-heavy.
2. **FedAvg:** Clients download the global model, train locally for $E$ epochs, and send the *updated weights* back to the server. The server performs a weighted average of these local models.

## 📉 Results: The Impact of Local Epochs ($E$)
The critical contribution of FedAvg is demonstrating that **more local computation dramatically reduces the communication rounds** required to reach a target accuracy. 

![Convergence Curve](assets/convergence_curve.png)

*As seen above, higher local epoch counts ($E=5$, $E=20$) converge substantially faster than FedSGD, achieving up to 10-100x reduction in required communication rounds.*

## 🚀 Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Simulation
1. Open the Jupyter Notebook:
```bash
jupyter notebook FedAvg_DP_Simulation.ipynb
```
2. Run all cells to download the MNIST dataset, partition it, train the models, and generate the convergence curves.
3. Output plots are automatically saved into the `assets/` directory.
