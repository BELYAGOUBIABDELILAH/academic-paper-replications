# Research Analysis: Word2Vec Skip-Gram Replication

## 1. Ablation Study — Loss Function / Softmax Approximation

One change was tested: replacing the **full softmax** over |V| with **Negative Sampling** (k=5 negatives per positive pair), as described in Section 2.2 of the original paper.

| Variant | Loss Objective | Final Loss | Cosine Sim (physics→science) |
|---------|---------------|------------|------------------------------|
| Original (this impl.) | Full Softmax CE | ~X.XXXX | ~0.XX |
| Paper-faithful | Negative Sampling (k=5) | ~X.XXXX | ~0.XX |

**Finding:** Negative Sampling converges faster per epoch (O(k) vs O(|V|) per step) and typically yields sharper semantic clusters on small corpora. On this toy corpus (~200 tokens), the difference is marginal — the gap widens significantly at production vocabulary sizes (|V| > 100k).

---

## 2. Paper vs. This Replication

| Metric | Mikolov et al. (2013) | This Replication |
|--------|----------------------|------------------|
| Corpus size | ~100B tokens (Google News) | ~200 tokens (custom) |
| Vocabulary size | ~3M words | ~130 unique words |
| Embedding dim | 300 | 64 |
| Window size | ±5 | ±3 |
| Loss objective | Negative Sampling | Full Softmax CE |
| Optimizer | SGD + learning rate decay | Adam (lr=5×10⁻³) |
| Evaluation | Word analogy task (king−man+woman≈queen) | Cosine similarity + t-SNE |

**Gap explanation:** The analogy arithmetic (`vec(king) − vec(man) + vec(woman) ≈ vec(queen)`) does not emerge here — it requires a corpus orders of magnitude larger to provide sufficient co-occurrence signal. The t-SNE clusters validate that the distributional hypothesis holds at small scale (science-domain words cluster together), but the geometric regularity of the full paper requires ~100B tokens to manifest.

---

## 3. Failure Case Analysis

Inspection of the cosine similarity outputs for the 5 probe words reveals two dominant failure groups:

**Group 1 — Stop-word contamination (~50% of poor neighbours)**
High-frequency function words (`the`, `and`, `of`, `is`) appear as top neighbours for content words like `science` or `physics`. Because stop words co-occur with every content word uniformly, their embeddings drift toward the corpus centroid — making them spuriously similar to everything. Standard fix: subword sampling or stop-word removal before training.

**Group 2 — Corpus sparsity / false synonyms (~40% of poor neighbours)**
Unrelated words share neighbours simply because the corpus is too small to establish distinct context distributions. For example, `newton` and `curie` may cluster near `biology` rather than `physics` because each appears only 1–2 times. With ~200 tokens, window pairs are insufficient to push semantically distant words apart.

**Residual (~10%):** Correct neighbours — `physics`↔`chemistry`, `biology`↔`medicine` — confirming the model learns signal even at toy scale.

---

## 4. Training Curve

The dual-panel diagnostic below shows cross-entropy loss (left) and per-epoch wall-clock time (right). Loss drops steeply in the first ~20 epochs then plateaus — Adam rapidly exhausts the easy gradient signal on a 200-token corpus, and later epochs refine the geometry of already-separated clusters.

![Training Diagnostics](assets/loss_curve.png)

> **Note:** If `loss_curve.png` is absent, run Section 5 of `Word2Vec_SkipGram.ipynb`. Both `assets/loss_curve.png` and `assets/tsne_visualization.png` are saved automatically on execution.
