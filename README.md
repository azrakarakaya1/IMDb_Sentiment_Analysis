# IMDb Sentiment Analysis with LSTM

A binary sentiment classifier for IMDb movie reviews built from scratch using a hand-crafted LSTM neural network in PyTorch. The project walks through the full machine learning pipeline — data exploration, preprocessing, model definition, training, and evaluation — across five Jupyter notebooks designed to run top-to-bottom in Google Colab.

---

## Project Overview

This project implements a sentiment analysis system that classifies IMDb movie reviews as **positive** or **negative**. The core model is a three-layer stacked LSTM with Batch Normalisation, trained end-to-end using binary cross-entropy loss with logits. All preprocessing logic (text cleaning, vocabulary building, sequence encoding) is implemented from scratch in `src/preprocessing.py` and is fully unit-tested and property-tested.

Key design choices:
- **Framework**: PyTorch — explicit training loop for full visibility and debuggability
- **Custom tokenizer**: Hand-built `Vocabulary` + `Tokenizer` classes, no external NLP libraries
- **Dataset split**: 80/10/10 (train/val/test) with fixed seed `42` for reproducibility
- **Early stopping**: Patience of 3 epochs on validation loss to prevent overfitting
- **Property-based testing**: Pure preprocessing functions are verified with [Hypothesis](https://hypothesis.readthedocs.io/)

---

## Dataset

**File**: `data/IMDB_Dataset.csv`

The dataset contains 50,000 IMDb movie reviews with two columns:
- `review` — raw review text (may contain HTML tags)
- `sentiment` — label: `"positive"` or `"negative"`

The dataset is perfectly balanced: 25,000 positive and 25,000 negative reviews.

**Download**: [Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

After downloading, place the CSV at `data/IMDB_Dataset.csv` relative to the repo root.

---

## Notebook Sequence

Run the notebooks in order. Each notebook saves artifacts consumed by the next.

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `notebooks/01_data_exploration.ipynb` | Load the CSV, verify shape and class balance, compute length statistics, render sample reviews, plot histograms and word clouds |
| 02 | `notebooks/02_preprocessing.ipynb` | Split data 80/10/10, clean text, build vocabulary from training split, encode all splits to fixed-length integer sequences, save `vocab.pkl` and `.npy` files |
| 03 | `notebooks/03_model_building.ipynb` | Define the final `SentimentLSTM` (3-layer + BatchNorm), print model summary, run smoke-test forward pass, preview experiment variants |
| 04 | `notebooks/04_training_and_evaluation.ipynb` | Train with early stopping, plot loss/accuracy curves, load best checkpoint, evaluate on test set, display confusion matrix, classification report, and custom review predictions |

---

## Running in Google Colab

1. Upload `data/IMDB_Dataset.csv` to your Colab session (or mount Google Drive and point the path there).
2. Clone or upload this repository to Colab.
3. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
4. Run notebooks `01` through `05` in order, top-to-bottom. Each notebook changes its working directory to the repo root automatically, so all relative paths (`data/`, `checkpoints/`) and `src/` imports work correctly.

> **Note**: Notebooks are designed to run without modification in a fresh Colab environment. GPU runtime is recommended for notebook 04 (training).

---

## Running Tests

Install dependencies first, then run the full test suite:

```bash
pip install -r requirements.txt
pytest tests/ -v
```

To run only property-based tests:

```bash
pytest tests/ -v -k "property"
```

All property-based tests use `@settings(max_examples=100)` from Hypothesis.

---

## Directory Structure

```
imdb-sentiment-lstm/
├── data/
│   └── IMDB_Dataset.csv          # source dataset (download from Kaggle)
├── checkpoints/
│   └── best_model.pt             # saved by 04_training.ipynb (after training)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # clean_text, Vocabulary, Tokenizer
│   └── model.py                  # SentimentLSTM
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py     # unit + property-based tests for preprocessing
│   └── test_model.py             # unit tests for SentimentLSTM
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_building.ipynb
│   ├── 04_training.ipynb
│   └── 05_evaluation.ipynb
├── report/
│   └── final_report.md           # written report (2,000–5,000 words)
├── requirements.txt
└── README.md
```

---

## Dependencies

All dependencies are pinned in `requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.2.2 | Neural network framework |
| `numpy` | 1.26.4 | Array operations and `.npy` file I/O |
| `pandas` | 2.2.2 | CSV loading and data manipulation |
| `scikit-learn` | 1.4.2 | Train/val/test splitting, evaluation metrics |
| `hypothesis` | 6.100.1 | Property-based testing |
| `pytest` | 8.1.1 | Test runner |
| `matplotlib` | 3.8.4 | Plotting loss/accuracy curves and histograms |
| `seaborn` | 0.13.2 | Confusion matrix heatmap |
| `wordcloud` | 1.9.3 | Word cloud visualizations in notebook 01 |
| `tqdm` | 4.66.2 | Progress bars during training |
| `ipykernel` | 6.29.4 | Jupyter kernel support |
| `nbformat` | 5.10.4 | Notebook format utilities |

---

## Model Architecture

```
Input (batch_size, 500)  — integer token indices
    │
    ▼
nn.Embedding(vocab_size=20002, embedding_dim=128, padding_idx=0)
    │
    ▼
nn.LSTM(input_size=128, hidden_size=256, num_layers=3,
        batch_first=True, dropout=0.4)
    │  (take final hidden state h_n[-1], shape: batch×256)
    ▼
nn.BatchNorm1d(256)
    │
    ▼
nn.Dropout(p=0.4)
    │
    ▼
nn.Linear(256, 1)  — raw logit
    │
    ▼
Output (batch_size, 1)  — logit; apply torch.sigmoid() for probability
```

Threshold: `sigmoid(logit) >= 0.5` → positive, `< 0.5` → negative.

---

## Results

Test accuracy is filled in after notebook 05 runs in Colab.

| Configuration | Val Accuracy | Test Accuracy |
|---------------|-------------|---------------|
| Baseline (1-layer LSTM, hidden=64, dropout=0.5) | ~87% | TBD |
| Experiment (2-layer LSTM, hidden=128, dropout=0.5) | ~89.56% | TBD |
| **Final (3-layer LSTM + BatchNorm, hidden=256, dropout=0.4)** | **90.52%** | **TBD** |
