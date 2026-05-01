# PyTorch Implementation of Word2Vec (Skip-Gram)

## 📌 Objective
A from-scratch PyTorch replication of the Skip-Gram architecture introduced in *"Efficient Estimation of Word Representations in Vector Space"* (Mikolov et al., 2013). This project demonstrates how distributed representations of words can be learned by predicting context words, forming the foundational architecture for modern semantic search and NLP pipelines.

## 🧠 Architecture & Methodology
* **Architecture:** Skip-Gram model focusing on predicting $c$ context words from a single center word.
* **Corpus:** A small custom text corpus containing sample sentences to demonstrate embedding clustering based on simple context relationships.
* **Hyperparameters:** 
  * Embedding Dimension: 50
  * Window Size: 2
  * Optimizer: Adam

## 📊 Results & Semantic Validation
The model successfully maps semantically similar words close to each other in the vector space. 

### t-SNE Dimensionality Reduction
The visualization demonstrates the learned semantic clusters in the 50-dimensional space, reduced to 2D using t-SNE.
![t-SNE Visualization](assets/tsne_visualization.png)

### Training Loss
![Loss Curve](assets/loss_curve.png)

## 🛠️ Usage
```bash
pip install -r requirements.txt
jupyter notebook Word2Vec_SkipGram.ipynb
```
