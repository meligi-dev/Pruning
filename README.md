# Bi-LSTM Text Classification & Pruning Baseline

This project implements a high-performance **Bidirectional LSTM (Bi-LSTM)** neural network in PyTorch for multi-class text classification on the **AG News** benchmark dataset. Initialized with frozen **FastText 300-dimensional pre-trained embeddings**, this model serves as the foundational baseline architecture for evaluating neural network model compression and pruning techniques.

---

## 🏗️ Model Architecture

The model processes variable-length text sequences using dynamic packed sequence execution:

```
                  [ Class Probabilities (4 Topics) ]
                                   ^
                                   |  Linear (512 -> 4)
                  +----------------+-----------------+
                  |      Concatenated Hidden State   |
                  |     [h_forward ; h_backward]     |
                  +----------------+-----------------+
                                   ^
                  +----------------+-----------------+
                  |     2-Layer Bi-directional LSTM  |
                  |   (hid_dim=256, dropout=0.5)     |
                  +----------------+-----------------+
                                   ^
                  +----------------+-----------------+
                  |     pack_padded_sequence         |
                  +----------------+-----------------+
                                   ^
                  +----------------+-----------------+
                  |     Pre-trained FastText         |
                  |   Embeddings (300D, Frozen)      |
                  +----------------+-----------------+
                                   ^
                        [ Input Token Sequence ]
```

### Architectural Highlights

1. **Pre-trained Embeddings**:
   - Initialized with **FastText / Word2Vec** vectors (`wiki-news-300d-1M.vec`, 300 dimensions).
   - OOV and special tokens (`<UNK>`, `<BOS>`, `<EOS>`) are initialized from $\mathcal{N}(0, 0.6^2)$, while `<PAD>` is set to all zeros.
   - Embeddings are frozen (`freeze=True`) to retain general language semantic representations and accelerate training convergence.
2. **Dynamic RNN Processing**:
   - Uses PyTorch's `pack_padded_sequence` and `pad_packed_sequence` to skip padding computations across batches, preserving exact gradients and saving compute.
3. **Bidirectional Aggregation**:
   - 2-layer stacked bidirectional LSTM with inter-layer dropout ($p = 0.5$).
   - Concatenates the final forward hidden state $h_n[-2, :, :]$ and backward hidden state $h_n[-1, :, :]$ into a compact 512-dimensional document representation.
4. **Classification Head**:
   - Fully connected projection layer maps the 512-dimensional representation directly to the 4 target class logits.

---

## ⚙️ Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Embedding Dimension** | `300` | Pre-trained FastText embeddings |
| **Hidden Dimension** | `256` | Per-direction LSTM hidden state size ($2 \times 256 = 512$ total) |
| **Number of Layers** | `2` | Stacked bidirectional recurrent layers |
| **Recurrent Dropout** | `0.5` | Regularization between LSTM layers |
| **Total Parameters** | `16,812,496` | ~16.8M parameters |
| **Batch Size** | `64` | Mini-batch training size |
| **Learning Rate** | `1e-3` | Initial learning rate for Adam optimizer |
| **Epochs** | `3` | Rapid convergence |

---

## 📊 Dataset & Categories

The model is evaluated on the standard **AG News** corpus (`fancyzhx/ag_news`), comprising 120,000 training news articles and 7,600 test articles distributed evenly across 4 news domains:

| Label ID | Category | Description / Sample Focus |
| :---: | :--- | :--- |
| **0** | **World** | International politics, diplomacy, conflicts |
| **1** | **Sports** | Olympics, football, tennis, athletic tournaments |
| **2** | **Business** | Wall Street, corporate earnings, stock markets, aerospace |
| **3** | **Sci/Tech** | Hardware, consumer electronics, software, telecom |

* **Data Partitioning**: The training set is split 90/10 into **108,000 training samples** and **12,000 validation samples**.

---

## 📈 Training Convergence & Test Evaluation

The model achieves rapid convergence within only 3 epochs:

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Validation Macro F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | `0.302` | `89.65%` | `0.239` | `91.79%` | `0.9176` |
| **02** | `0.171` | `94.28%` | `0.222` | `92.22%` | `0.9222` |
| **03** | `0.115` | `96.08%` | `0.248` | `92.03%` | `0.9202` |

### 🏆 Test Set Performance (7,600 unseen articles)
* **Test Accuracy**: **`92.71%`**
* **Macro F1 Score**: **`0.9242`**
* **Test Loss**: `0.236`

### Detailed Classification Report

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **World** | `0.95` | `0.91` | `0.93` | 1,900 |
| **Sports** | `0.96` | `0.99` | `0.97` | 1,900 |
| **Business** | `0.90` | `0.89` | `0.90` | 1,900 |
| **Sci/Tech** | `0.89` | `0.91` | `0.90` | 1,900 |
| **Macro Average** | **`0.92`** | **`0.92`** | **`0.92`** | 7,600 |

---

## ✂️ Model Pruning & Compression Context

This model acts as the baseline for neural network compression experiments. Common pruning strategies that can be benchmarked against this checkpoint (`LSTM.pth`):

1. **Unstructured Magnitude Pruning**: Zeroing out weights whose absolute magnitude $|w_{ij}|$ is below a given percentile threshold using `torch.nn.utils.prune.l1_unstructured`.
2. **Structured Pruning**: Removing entire LSTM hidden units or projection channels using `ln_structured` pruning to achieve real hardware inference acceleration.
3. **Iterative Magnitude Pruning (IMP)**: Progressively pruning weights and retraining (fine-tuning) to recover accuracy according to the *Lottery Ticket Hypothesis*.

---

## 📁 Repository Structure

```
Pruning/
├── Pruning.ipynb     # Complete pipeline: data loading, FastText lookup, training, and evaluation
├── README.md         # Comprehensive model documentation and benchmark metrics
├── .gitignore        # Ignores large checkpoint weights and caches
└── LSTM.pth          # Saved trained model weights checkpoint (67.2 MB)
```

---

## 🚀 Getting Started

### 1. Requirements
Install dependencies:
```bash
pip install torch datasets gensim spacy scikit-learn pandas numpy
python -m spacy download en_core_web_sm
```

### 2. Pre-trained Embeddings
The notebook loads FastText vectors (`wiki-news-300d-1M.vec`). Download the pre-trained embeddings if needed:
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip wiki-news-300d-1M.vec.zip
```

### 3. Execution
Launch and run the notebook:
```bash
jupyter notebook Pruning/Pruning.ipynb
```
