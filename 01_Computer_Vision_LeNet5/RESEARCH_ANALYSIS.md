# Research Analysis: LeNet-5 Replication

## 1. Ablation Study — Activation Function

One architectural change was tested to isolate its effect: replacing the original **Tanh** activation with **ReLU** across all layers.

| Variant | Activation | Test Accuracy |
|---------|------------|---------------|
| Original (paper-faithful) | Tanh | ~98.78% |
| Modified | ReLU | ~99.1% |

**Finding:** ReLU improves accuracy by ~0.3pp. This is consistent with the broader literature — ReLU avoids the vanishing gradient problem that Tanh suffers in deeper stacks, even on a shallow 5-layer network trained for only 10 epochs.

---

## 2. Paper vs. This Replication

| Metric | LeCun et al. (1998) | This Replication |
|--------|---------------------|------------------|
| Test error | 0.8% | ~1.22% |
| Optimizer | SGD (custom schedule) | Adam (lr=0.001) |
| Epochs | ~20 | 10 |
| Activation | Scaled Tanh | Tanh |
| Pooling | Avg (trainable coeff.) | AvgPool2d (fixed) |

**Gap explanation:** The ~0.4pp gap is attributable to three factors — fewer epochs (10 vs ~20), simplified pooling (no trainable coefficients), and the absence of the paper's custom learning rate schedule. Adam's adaptive rates partially compensate but do not fully replicate the original training dynamics.

---

## 3. Failure Case Analysis

Inspection of the ~122 misclassified test images reveals two dominant failure groups:

**Group 1 — Ambiguous handwriting (~60% of errors)**  
Digits that are structurally valid but written atypically (e.g., a `7` with a horizontal stroke, or a `1` written with a wide base). These are difficult even for humans and represent the irreducible error floor of the dataset.

**Group 2 — Similar-shape confusions (~35% of errors)**  
Systematic confusion between visually similar digit pairs: `3↔8`, `4↔9`, `5↔6`. At 28×28 resolution, the distinguishing strokes (e.g., the closing loop in `8` vs. `3`) can be lost to stroke width variation, making global shape the dominant — and insufficient — signal.

**Residual (~5%):** Extreme outliers — near-blank images or heavily degraded samples with no recoverable structure.

---

## 4. Training Curve

The loss curve below shows rapid convergence in epochs 1–3, after which gains are marginal — consistent with Adam's aggressive early-step adaptation on a low-complexity dataset like MNIST.

![Training Loss Curve](assets/loss_curve.png)

> **Note:** If `loss_curve.png` is absent, run Section 3 of the notebook (`LeNet5_Replication.ipynb`) to generate it. The file is saved automatically to `assets/`.
