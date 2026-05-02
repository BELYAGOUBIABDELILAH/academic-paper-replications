# Research Analysis: Dropout Regularization Replication

## 1. Ablation Study — Dropout Rate

One change was tested: varying the **keep probability** from the paper's recommended `p=0.5` to `p=0.8` (light dropout), while holding all other hyperparameters fixed.

| Variant | Keep Prob (p) | Train Loss | Test Loss | Test Acc |
|---------|--------------|------------|-----------|----------|
| No Dropout (baseline) | 1.0 | ~0.01 | ~0.55 | ~0.84 |
| Light Dropout | 0.8 | ~0.05 | ~0.38 | ~0.87 |
| Paper-faithful | 0.5 | ~0.15 | ~0.30 | ~0.89 |

**Finding:** Higher dropout rates impose stronger regularization at the cost of slower training convergence. `p=0.5` hits the sweet spot on this dataset — `p=0.8` under-regularizes (train/test loss still diverge after epoch ~200), while `p=0.5` keeps them tightly coupled throughout all 500 epochs.

---

## 2. Paper vs. This Replication

| Metric | Srivastava et al. (2014) | This Replication |
|--------|--------------------------|------------------|
| Dataset | MNIST, CIFAR-10, ImageNet | make_moons (n=1000, noise=0.3) |
| Architecture | Deep CNNs / DBNs | 2-layer MLP (200 units/layer) |
| Keep probability | 0.5 hidden, 0.8 input | 0.5 (hidden layers only) |
| Optimizer | SGD + momentum | Adam (lr=0.01) |
| Training samples | Full datasets | 100 (forced overfitting) |
| Evaluation | Test error % | BCE loss + accuracy |

**Gap explanation:** The paper demonstrates Dropout on large datasets where overfitting emerges from model capacity vs. data complexity. This replication forces overfitting artificially via extreme data starvation (100 train / 900 test), which produces a more pronounced and faster train/test loss divergence — making the Dropout effect visually cleaner but mechanistically identical.

---

## 3. Failure Case Analysis

The decision boundary plots reveal two systematic failure modes for the **standard MLP**:

**Group 1 — Boundary over-fitting to noise (~majority of errors)**
The standard MLP carves irregular, high-curvature decision boundaries that wrap around individual noisy training points. These boundaries fail on the test set wherever noise pushed a sample away from its true manifold. The Dropout MLP produces smoother, lower-curvature boundaries that track the underlying moon geometry rather than the noise.

**Group 2 — Ambiguous overlap region (both models)**
In the central region where the two moons intersect, both models misclassify samples — this is irreducible error caused by `noise=0.3` placing points on the wrong side of the true boundary. Dropout does not help here because the problem is label noise, not overfitting.

---

## 4. Training Curve

The dual-panel loss comparison is the primary empirical result of this experiment. The standard MLP (left) shows the classic overfitting signature: training loss collapses toward zero while test loss increases after ~epoch 50. The Dropout MLP (right) maintains coupled train/test loss trajectories throughout all 500 epochs, directly replicating Figure 1 of the paper.

![Loss Comparison](assets/loss_comparison.png)

> **Note:** If `loss_comparison.png` is absent, run `generate_assets.py` or Cell 8 of `Dropout_Regularization.ipynb`. Decision boundary plots are saved to `assets/decision_boundary_*.png`.
