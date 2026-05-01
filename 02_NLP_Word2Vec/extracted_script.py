"""
Cell 1 — Imports & Global Configuration
----------------------------------------
All third-party dependencies are imported here for transparency and auditability.
The SEED constant ensures every stochastic operation (weight init, DataLoader
shuffling, t-SNE) produces identical results across independent runs.
"""

from __future__ import annotations

import re
import time
import random
import collections
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# ── Aesthetic configuration ────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

print(f"PyTorch  : {torch.__version__}")
print(f"NumPy    : {np.__version__}")
def seed_everything(seed: int = 42) -> None:
    """Seed all stochastic sources for fully deterministic execution.

    Sets identical seeds across Python's ``random`` module, NumPy, and PyTorch
    (both CPU and CUDA). Also enables CuDNN's deterministic mode, at the cost
    of a small runtime penalty on GPU.

    Args:
        seed: Integer seed value. Defaults to 42 (community convention).

    Example:
        >>> seed_everything(0)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Resolve the best available compute device in priority order.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
    Device-agnostic design ensures the same script runs identically on a
    MacBook, a Colab T4, or a university HPC cluster.

    Returns:
        torch.device: The selected compute device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Global constants ───────────────────────────────────────────────────────────
SEED: int = 42
DEVICE: torch.device = get_device()

seed_everything(SEED)
print(f"Seed     : {SEED}")
print(f"Device   : {DEVICE}")
# ── Raw corpus ─────────────────────────────────────────────────────────────────
RAW_CORPUS: str = """
Science is a systematic enterprise that builds and organises knowledge in the form of
testable explanations and predictions about the universe. Modern science is typically
divided into three major branches that consist of the natural sciences, which study
nature in the broadest sense, the social sciences, which study people and societies,
and the formal sciences, which study abstract concepts. There is disagreement about
whether the formal sciences actually constitute a science as they do not rely on
empirical evidence. The history of science and technology spans thousands of years.
Mathematics, logic, astronomy, medicine, and physics are among the oldest known
scientific disciplines. Physics describes and predicts the behaviour of objects, from
elementary particles to the entire cosmos. Chemistry is the scientific study of the
properties and behaviour of matter. Biology is the natural science that studies life
and living organisms including their physical structure, chemical processes, molecular
interactions, physiological mechanisms, development, and evolution. Scientists use
the scientific method to test their ideas by designing experiments and collecting data.
Data analysis allows scientists to identify patterns and form new hypotheses.
Universities and research institutions employ thousands of scientists who publish
their findings in peer-reviewed academic journals. Breakthroughs in quantum mechanics
led to advances in electronics, materials science, and modern computing. Einstein
revolutionised physics with his theories of relativity. Darwin proposed the theory
of evolution by natural selection which transformed biology and medicine. Newton
formulated laws of motion and universal gravitation that unified terrestrial and
celestial mechanics. Curie pioneered research in radioactivity and became the first
person to win two Nobel Prizes in different scientific disciplines including physics
and chemistry. Researchers continue to explore the boundaries of human knowledge
through experiments, mathematics, and computational modelling. Artificial intelligence
and machine learning are transforming how scientists analyse vast datasets and discover
hidden patterns in biology, physics, astronomy, and medicine.
"""
class Vocabulary:
    """Bidirectional word ↔ integer mapping with frequency statistics.

    Encapsulates all tokenisation and indexing logic so that the Dataset and
    Model classes remain decoupled from raw string processing.

    Attributes:
        word2idx: Mapping from token string to integer index.
        idx2word: Reverse mapping from integer index to token string.
        freq: Token frequency counts in the corpus.
        size: Total number of unique tokens (vocabulary size |V|).

    Example:
        >>> vocab = Vocabulary.from_corpus("the cat sat on the mat")
        >>> vocab.word2idx["cat"]
        1
    """

    def __init__(
        self,
        word2idx: Dict[str, int],
        idx2word: Dict[int, str],
        freq: Dict[str, int],
    ) -> None:
        self.word2idx: Dict[str, int] = word2idx
        self.idx2word: Dict[int, str] = idx2word
        self.freq: Dict[str, int] = freq
        self.size: int = len(word2idx)

    # ── Class-level factory ────────────────────────────────────────────────────
    @classmethod
    def from_corpus(
        cls,
        corpus: str,
        min_freq: int = 1,
    ) -> "Vocabulary":
        """Build a Vocabulary from a raw text string.

        Tokenises the corpus by lowercasing and removing non-alphabetic
        characters, then filters tokens whose frequency is below *min_freq*
        to reduce noise from hapax legomena.

        Args:
            corpus: Raw text string (whitespace-separated words).
            min_freq: Minimum token frequency to include in vocabulary.

        Returns:
            Vocabulary: Fully constructed Vocabulary instance.
        """
        # Lowercase and strip punctuation — standard NLP pre-processing
        tokens: List[str] = re.findall(r"[a-z]+", corpus.lower())
        freq: Dict[str, int] = dict(collections.Counter(tokens))

        # Retain only tokens that meet the minimum frequency threshold
        vocab_tokens: List[str] = sorted(
            [w for w, c in freq.items() if c >= min_freq]
        )

        word2idx: Dict[str, int] = {w: i for i, w in enumerate(vocab_tokens)}
        idx2word: Dict[int, str] = {i: w for w, i in word2idx.items()}

        return cls(word2idx, idx2word, freq)

    def encode(self, word: str) -> Optional[int]:
        """Look up the integer index for a token, returning None if unknown.

        Args:
            word: Lowercase token string.

        Returns:
            Integer index, or None if the token is out-of-vocabulary.
        """
        return self.word2idx.get(word)

    def decode(self, idx: int) -> str:
        """Retrieve the token string for an integer index.

        Args:
            idx: Integer word index.

        Returns:
            Token string.

        Raises:
            KeyError: If *idx* is not in the vocabulary.
        """
        return self.idx2word[idx]

    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size})"


# ── Build vocabulary ───────────────────────────────────────────────────────────
TOKENS: List[str] = re.findall(r"[a-z]+", RAW_CORPUS.lower())
vocab: Vocabulary = Vocabulary.from_corpus(RAW_CORPUS, min_freq=1)

print(vocab)
print(f"Corpus tokens : {len(TOKENS):,}")
print(f"Vocabulary    : {vocab.size:,} unique words")
print()
# Show the 10 most frequent tokens
top10 = sorted(vocab.freq.items(), key=lambda x: -x[1])[:10]
print("Top-10 tokens by frequency:")
for word, count in top10:
    print(f"  {word:<20} {count}")
class SkipGramDataset(Dataset):
    """PyTorch Dataset producing (center, context) index pairs for Skip-Gram.

    Implements the sliding-window procedure described in Section 4.1 of
    Mikolov et al. (2013). For each center token at position *t*, every token
    within the half-window ``[-window_size, +window_size]`` (excluding *t*
    itself) is emitted as a positive training pair.

    Args:
        tokens: Ordered list of tokenised words from the corpus.
        vocab: Vocabulary instance providing word ↔ index mappings.
        window_size: Half-width *k* of the context window.

    Attributes:
        pairs: List of (center_idx, context_idx) integer tuples.

    Example:
        >>> ds = SkipGramDataset(["the", "cat", "sat"], vocab, window_size=1)
        >>> len(ds)   # (center=cat → the), (center=cat → sat)
        4
    """

    def __init__(
        self,
        tokens: List[str],
        vocab: Vocabulary,
        window_size: int = 2,
    ) -> None:
        self.pairs: List[Tuple[int, int]] = self._build_pairs(
            tokens, vocab, window_size
        )

    @staticmethod
    def _build_pairs(
        tokens: List[str],
        vocab: Vocabulary,
        window_size: int,
    ) -> List[Tuple[int, int]]:
        """Slide a context window across the token sequence to generate pairs.

        Args:
            tokens: Flat list of string tokens (entire corpus).
            vocab: Vocabulary for index lookup.
            window_size: Half-width of the context window.

        Returns:
            List of (center_index, context_index) integer tuples.
        """
        pairs: List[Tuple[int, int]] = []
        n: int = len(tokens)

        for t, center_word in enumerate(tokens):
            center_idx: Optional[int] = vocab.encode(center_word)
            if center_idx is None:
                continue  # Skip out-of-vocabulary tokens

            # Determine valid window boundaries (clamp to corpus edges)
            left: int = max(0, t - window_size)
            right: int = min(n - 1, t + window_size)

            for j in range(left, right + 1):
                if j == t:
                    continue  # Exclude the center word from its own context
                context_idx: Optional[int] = vocab.encode(tokens[j])
                if context_idx is not None:
                    pairs.append((center_idx, context_idx))

        return pairs

    # ── Dataset protocol ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single (center, context) pair as LongTensors.

        Args:
            idx: Sample index.

        Returns:
            Tuple of scalar LongTensors ``(center_idx, context_idx)``.
        """
        center, context = self.pairs[idx]
        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
        )


# ── Hyperparameters ────────────────────────────────────────────────────────────
WINDOW_SIZE: int = 3       # Context half-width k
EMBEDDING_DIM: int = 64    # Dimensionality of the learned embedding space
BATCH_SIZE: int = 128      # Mini-batch size for stochastic gradient descent
NUM_EPOCHS: int = 120      # Training epochs
LEARNING_RATE: float = 5e-3

# Build dataset and wrap in a DataLoader
dataset = SkipGramDataset(TOKENS, vocab, window_size=WINDOW_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    # num_workers=0 avoids fork issues in Jupyter
    num_workers=0,
)

print(f"Training pairs : {len(dataset):,}")
print(f"Batches/epoch  : {len(dataloader)}")
print(f"Embedding dim  : {EMBEDDING_DIM}")
print(f"Window size    : ±{WINDOW_SIZE}")
class SkipGramWord2Vec(nn.Module):
    """Skip-Gram Word2Vec model (Mikolov et al., 2013 — arXiv:1301.3781).

    Architecture
    ------------
    1. **Input projection (Embedding layer):** Maps a discrete center-word
       index to a continuous *d*-dimensional vector. This lookup table is the
       object we train; it IS the Word2Vec model. After training, each row
       ``W[i]`` encodes the semantic neighbourhood of word *i*.

       Formally: :math:`\\mathbf{v}_c = \\mathbf{W}_{\\text{in}}[w_c]`
       where :math:`\\mathbf{W}_{\\text{in}} \\in \\mathbb{R}^{|V| \\times d}`.

    2. **Output projection (Linear layer):** Projects the embedding back to
       vocabulary space to produce unnormalised log-probabilities (logits)
       over all |V| words.

       Formally: :math:`\\mathbf{z} = \\mathbf{W}_{\\text{out}} \\mathbf{v}_c + \\mathbf{b}`
       where :math:`\\mathbf{W}_{\\text{out}} \\in \\mathbb{R}^{|V| \\times d}`.

    Loss
    ----
    ``nn.CrossEntropyLoss`` applies a numerically stable log-softmax internally
    and computes negative log-likelihood::

        L = -log P(w_context | w_center)

    Args:
        vocab_size: Size of the vocabulary |V|.
        embedding_dim: Dimensionality *d* of the embedding space.

    Note:
        The full softmax over |V| is computationally expensive for large
        vocabularies. Production implementations replace it with Negative
        Sampling (Section 2.2 of the paper). This implementation uses the
        full softmax for clarity.
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()

        # ── Layer 1: Input embedding matrix W_in ──────────────────────────────
        # Each row is a d-dimensional word vector; this IS the final artefact.
        self.embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        # ── Layer 2: Output projection W_out ──────────────────────────────────
        # Projects the embedding vector to unnormalised logits over |V| words.
        self.output_projection: nn.Linear = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
            bias=True,
        )

        # Initialise weights with a zero-centred Gaussian (σ=0.01)
        # to break initial symmetry while keeping gradients stable.
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.01)

    def forward(self, center_idx: torch.Tensor) -> torch.Tensor:
        """Compute logits over the vocabulary for a batch of center words.

        Args:
            center_idx: LongTensor of shape ``(B,)`` containing center-word
                integer indices for a mini-batch of size *B*.

        Returns:
            torch.Tensor: Logit matrix of shape ``(B, |V|)``.  Each row
            contains the unnormalised log-probability that each vocabulary
            word is a context neighbour of the corresponding center word.
        """
        # v_c  ∈ ℝ^(B × d)  — look up the embedding for each center word
        embed: torch.Tensor = self.embeddings(center_idx)

        # z  ∈ ℝ^(B × |V|) — project to vocabulary logits
        logits: torch.Tensor = self.output_projection(embed)
        return logits

    def get_embedding_matrix(self) -> np.ndarray:
        """Extract the trained embedding weight matrix as a NumPy array.

        Returns:
            ndarray of shape ``(|V|, d)`` on CPU, detached from the
            computational graph. Row *i* is the word vector for index *i*.
        """
        return self.embeddings.weight.detach().cpu().numpy()


# ── Instantiate model, loss, and optimiser ─────────────────────────────────────
model: SkipGramWord2Vec = SkipGramWord2Vec(
    vocab_size=vocab.size,
    embedding_dim=EMBEDDING_DIM,
).to(DEVICE)

criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer: optim.Adam = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Log model summary
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f"\nTrainable parameters : {total_params:,}")
print(f"Embedding matrix     : {vocab.size} × {EMBEDDING_DIM} = {vocab.size * EMBEDDING_DIM:,} parameters")
def train_epoch(
    model: SkipGramWord2Vec,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    device: torch.device,
) -> float:
    """Execute one full pass over the training dataset (one epoch).

    Performs the standard forward → loss → backward → step cycle for every
    mini-batch. Gradient accumulation across batches is NOT used here, as
    mini-batches are already small enough.

    Args:
        model: The SkipGramWord2Vec model in training mode.
        dataloader: DataLoader yielding (center, context) LongTensor pairs.
        criterion: Cross-entropy loss function.
        optimizer: Adam optimiser holding model parameter references.
        device: Compute device for tensor placement.

    Returns:
        float: Mean cross-entropy loss averaged over all mini-batches.
    """
    model.train()  # Activates dropout, BatchNorm in training mode (none here, but best practice)
    total_loss: float = 0.0

    for center_batch, context_batch in dataloader:
        # ── Move data to compute device ────────────────────────────────────────
        center_batch = center_batch.to(device)   # shape: (B,)
        context_batch = context_batch.to(device) # shape: (B,)  — ground-truth labels

        # ── Forward pass ───────────────────────────────────────────────────────
        # logits: (B, |V|) — unnormalised scores for every vocabulary word
        logits: torch.Tensor = model(center_batch)

        # CrossEntropyLoss internally applies log-softmax to `logits` then
        # computes NLL against the integer class indices in `context_batch`.
        loss: torch.Tensor = criterion(logits, context_batch)

        # ── Backward pass & parameter update ──────────────────────────────────
        optimizer.zero_grad()  # Clear accumulated gradients from previous step
        loss.backward()        # Compute ∂L/∂θ for all parameters θ
        optimizer.step()       # Apply Adam update rule: θ ← θ − α · m̂ / (√v̂ + ε)

        total_loss += loss.item()

    return total_loss / len(dataloader)


def run_training(
    model: SkipGramWord2Vec,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    device: torch.device,
    num_epochs: int,
    log_interval: int = 10,
) -> Dict[str, List[float]]:
    """Full training loop with comprehensive metric tracking.

    Args:
        model: SkipGramWord2Vec instance.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Parameter optimiser.
        device: Target compute device.
        num_epochs: Total number of training epochs.
        log_interval: Print progress every *log_interval* epochs.

    Returns:
        Dict with keys ``"loss"`` and ``"epoch_time_s"`` — each a list of
        per-epoch values for downstream plotting and analysis.
    """
    history: Dict[str, List[float]] = {"loss": [], "epoch_time_s": []}

    print(f"{'Epoch':>6}  {'Loss':>10}  {'Time (s)':>10}")
    print("-" * 32)

    for epoch in range(1, num_epochs + 1):
        t_start: float = time.perf_counter()
        epoch_loss: float = train_epoch(
            model, dataloader, criterion, optimizer, device
        )
        elapsed: float = time.perf_counter() - t_start

        history["loss"].append(epoch_loss)
        history["epoch_time_s"].append(elapsed)

        if epoch % log_interval == 0 or epoch == 1:
            print(f"{epoch:>6}  {epoch_loss:>10.4f}  {elapsed:>10.3f}")

    total_time = sum(history["epoch_time_s"])
    print("-" * 32)
    print(f"Training complete — {num_epochs} epochs in {total_time:.2f}s")
    return history


# ── Run training ───────────────────────────────────────────────────────────────
history = run_training(
    model=model,
    dataloader=dataloader,
    criterion=criterion,
    optimizer=optimizer,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    log_interval=10,
)
# ── Training curve ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor="#0d0d14")

epochs_range = range(1, NUM_EPOCHS + 1)

# Loss curve
ax1 = axes[0]
ax1.set_facecolor("#0d0d14")
ax1.plot(epochs_range, history["loss"], color="#7DF9FF", linewidth=2)
ax1.fill_between(epochs_range, history["loss"], alpha=0.15, color="#7DF9FF")
ax1.set_title("Cross-Entropy Loss per Epoch", color="white", pad=12, fontsize=12)
ax1.set_xlabel("Epoch", color="#aaaaaa")
ax1.set_ylabel("Loss", color="#aaaaaa")
ax1.tick_params(colors="#aaaaaa")
for spine in ax1.spines.values(): spine.set_color("#333344")

# Per-epoch wall-clock time
ax2 = axes[1]
ax2.set_facecolor("#0d0d14")
ax2.bar(epochs_range, history["epoch_time_s"], color="#FF6B9D", alpha=0.8, width=0.8)
ax2.axhline(np.mean(history["epoch_time_s"]), color="#FFD700",
            linestyle="--", linewidth=1.2, label=f"Mean = {np.mean(history['epoch_time_s']):.3f}s")
ax2.set_title("Wall-Clock Time per Epoch", color="white", pad=12, fontsize=12)
ax2.set_xlabel("Epoch", color="#aaaaaa")
ax2.set_ylabel("Time (s)", color="#aaaaaa")
ax2.tick_params(colors="#aaaaaa")
ax2.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333344")
for spine in ax2.spines.values(): spine.set_color("#333344")

fig.suptitle("Skip-Gram Word2Vec — Training Diagnostics",
             color="white", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
import os
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/loss_curve.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
pass
print(f"Final loss : {history['loss'][-1]:.4f}")
def get_similar_words(
    query: str,
    vocab: Vocabulary,
    embedding_matrix: np.ndarray,
    top_n: int = 6,
) -> List[Tuple[str, float]]:
    """Retrieve the *top_n* vocabulary words most similar to *query*.

    Similarity is measured by cosine similarity between the query word's
    embedding vector and all other vectors in the embedding matrix.

    Args:
        query: A lowercase word present in the vocabulary.
        vocab: Vocabulary instance for index lookups.
        embedding_matrix: NumPy array of shape ``(|V|, d)``.
        top_n: Number of similar words to return (excluding the query itself).

    Returns:
        List of ``(word, similarity_score)`` tuples sorted by descending
        similarity. The query word is always excluded from results.

    Raises:
        ValueError: If *query* is not present in the vocabulary.
    """
    query_idx: Optional[int] = vocab.encode(query)
    if query_idx is None:
        raise ValueError(f"'{query}' is not in the vocabulary.")

    # Query vector: shape (1, d)
    query_vec: np.ndarray = embedding_matrix[query_idx].reshape(1, -1)

    # Pairwise cosine similarities: shape (1, |V|) → (|V|,)
    similarities: np.ndarray = cosine_similarity(query_vec, embedding_matrix)[0]

    # Sort descending; skip index 0 (the query itself, similarity = 1.0)
    ranked_indices: np.ndarray = np.argsort(similarities)[::-1]

    results: List[Tuple[str, float]] = []
    for idx in ranked_indices:
        if idx == query_idx:
            continue
        results.append((vocab.decode(idx), float(similarities[idx])))
        if len(results) == top_n:
            break

    return results


# ── Extract trained embedding matrix ──────────────────────────────────────────
model.eval()  # Disable training-specific layers (best practice)
with torch.no_grad():
    embedding_matrix: np.ndarray = model.get_embedding_matrix()  # shape: (|V|, d)

print(f"Embedding matrix shape : {embedding_matrix.shape}\n")
print("=" * 52)

# Semantic probe words drawn from the corpus's scientific domain
probe_words: List[str] = ["science", "physics", "biology", "medicine", "mathematics"]

for probe in probe_words:
    if vocab.encode(probe) is None:
        print(f"  '{probe}' not in vocabulary — skipping.")
        continue
    neighbours = get_similar_words(probe, vocab, embedding_matrix, top_n=5)
    print(f"\n  Query: '{probe}'")
    print(f"  {'-'*46}")
    for word, score in neighbours:
        bar = "█" * int(score * 20)
        print(f"  {word:<22} {score:+.4f}  {bar}")

print("\n" + "=" * 52)
def plot_tsne_embeddings(
    embedding_matrix: np.ndarray,
    vocab: Vocabulary,
    top_n_words: int = 80,
    perplexity: float = 15.0,
    random_state: int = SEED,
    figsize: Tuple[int, int] = (18, 14),
    save_path: Optional[str] = None,
) -> None:
    """Project word embeddings to 2D with t-SNE and render an annotated scatter plot.

    Selects the *top_n_words* most frequent vocabulary tokens, reduces their
    *d*-dimensional embeddings to 2D using t-SNE, and renders a publication-
    quality scatter plot with word labels.

    Args:
        embedding_matrix: NumPy array ``(|V|, d)`` of trained embeddings.
        vocab: Vocabulary for decoding indices and retrieving frequencies.
        top_n_words: Number of most-frequent words to include in the plot.
        perplexity: t-SNE perplexity parameter (balances local vs. global
            structure; recommended range: 5–50, see van der Maaten 2008).
        random_state: Seed for t-SNE's internal stochastic gradient descent.
        figsize: Matplotlib figure dimensions ``(width, height)`` in inches.
        save_path: If provided, save the figure to this file path as PNG.
    """
    # ── Select the top-N most frequent words ──────────────────────────────────
    ranked_words: List[str] = [
        w for w, _ in sorted(vocab.freq.items(), key=lambda x: -x[1])
        if vocab.encode(w) is not None
    ][:top_n_words]

    indices: List[int] = [vocab.encode(w) for w in ranked_words]  # type: ignore
    selected_embeddings: np.ndarray = embedding_matrix[indices]   # (top_n, d)

    # ── t-SNE projection: ℝ^d → ℝ^2 ──────────────────────────────────────────
    # Perplexity controls the effective number of neighbours considered;
    # must be less than the number of samples.
    effective_perplexity = min(perplexity, len(ranked_words) - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        n_iter=2000,
        random_state=random_state,
        init="pca",        # PCA initialisation converges faster and more stably
        learning_rate="auto",
    )
    coords_2d: np.ndarray = tsne.fit_transform(selected_embeddings)  # (top_n, 2)

    # ── Render plot ───────────────────────────────────────────────────────────
    bg_colour = "#08080f"
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_colour)
    ax.set_facecolor(bg_colour)

    # Colour-code points by log-frequency for an extra information layer
    log_freqs = np.log1p([vocab.freq.get(w, 0) for w in ranked_words])
    sc = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=log_freqs,
        cmap="plasma",
        s=90,
        alpha=0.9,
        edgecolors="#ffffff22",
        linewidths=0.5,
        zorder=2,
    )

    # Annotate each point with the corresponding word
    for i, word in enumerate(ranked_words):
        ax.annotate(
            word,
            xy=(coords_2d[i, 0], coords_2d[i, 1]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8.5,
            color="#e0e0e0",
            path_effects=[
                pe.withStroke(linewidth=2, foreground=bg_colour)
            ],
            zorder=3,
        )

    # Colour bar indicating log-frequency
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label("log(frequency + 1)", color="#aaaaaa", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa")

    ax.set_title(
        f"t-SNE Projection of Word2Vec Embeddings  "
        f"(d={embedding_matrix.shape[1]}, top {top_n_words} words)",
        color="white", fontsize=14, fontweight="bold", pad=16,
    )
    ax.set_xlabel("t-SNE Dimension 1", color="#888888", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", color="#888888", fontsize=11)
    ax.tick_params(colors="#555566")
    for spine in ax.spines.values():
        spine.set_color("#222233")

    # Subtle grid for spatial reference
    ax.grid(True, color="#1a1a2a", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    fig.text(
        0.99, 0.01,
        "Mikolov et al. (2013) — Skip-Gram Word2Vec",
        ha="right", va="bottom", color="#444455", fontsize=8,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Figure saved → {save_path}")

    pass


# ── Render the t-SNE visualisation ────────────────────────────────────────────
plot_tsne_embeddings(
    embedding_matrix=embedding_matrix,
    vocab=vocab,
    top_n_words=min(80, vocab.size),
    perplexity=12.0,
    save_path="assets/tsne_visualization.png",
)