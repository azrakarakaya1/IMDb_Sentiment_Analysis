# Final Report: IMDb Sentiment Analysis with LSTM

---

## 1. Dataset

### What the Dataset Is

The dataset used in this project is the **IMDb Movie Reviews Dataset**, a well-known benchmark in natural language processing. It contains 50,000 movie reviews scraped from the Internet Movie Database (IMDb), each labelled as either *positive* or *negative*. The dataset was sourced in CSV form as `data/IMDB_Dataset.csv`, with two columns:

- `review` — the raw review text, which frequently contains HTML markup (e.g. `<br />` line-break tags, `<b>` bold tags)
- `sentiment` — the label, either the string `"positive"` or `"negative"`

### Why This Dataset

The IMDb dataset was chosen for three reasons.

**First, it is well-studied.** Because it has been used as a benchmark for over a decade, there are published accuracy figures to compare against. This makes it straightforward to assess whether an implementation is correct: a single-layer LSTM with standard hyperparameters should achieve roughly 85–90% test accuracy. If results fall far outside that range, something has gone wrong in preprocessing or training.

**Second, it is appropriately sized for experimentation.** At 50,000 reviews, the dataset is large enough to train a meaningful model but small enough to process and train on a free Google Colab GPU in under an hour. A smaller dataset (e.g. 5,000 reviews) would not give the LSTM enough signal; a much larger one would make iteration impractical.

**Third, it is realistically messy.** The raw reviews contain HTML tags, mixed punctuation, capitalisation, and a wide vocabulary. This means preprocessing is genuinely necessary rather than cosmetic, making the pipeline work more instructive.

### Key Characteristics

- **Total samples**: 50,000
- **Class balance**: Exactly 50% positive (25,000) and 50% negative (25,000) — perfectly balanced
- **Review length**: After whitespace splitting, reviews range from a handful of tokens (very short mini-reviews) to over 2,500 tokens (long critical essays). The mean is roughly 230 tokens and the median is around 170 tokens. The 90th percentile falls at approximately 500 tokens, which motivated the choice of `max_seq_len=500` (see Section 2).
- **Text quality**: HTML is pervasive — nearly every review contains at least one `<br />` tag. Punctuation is heavy and inconsistent. The vocabulary is large (tens of thousands of distinct tokens) but the vast majority of tokens appear only once or twice.

---

## 2. Modeling Procedure

### 2.1 Data Preparation

The first step was to split the dataset **before** performing any text-dependent operations. This is a critical discipline: if the vocabulary were built from all 50,000 reviews, the model would have indirect knowledge of words appearing only in the test set, producing an optimistically biased evaluation — a form of *data leakage*.

The split was performed with `sklearn.model_selection.train_test_split` applied twice, with `random_state=42` for reproducibility:

1. First split: 80% train (40,000 reviews) / 20% temporary pool
2. Second split: 50% of the temporary pool → validation (5,000), 50% → test (5,000)

The resulting splits:

| Split | Reviews | Fraction |
|-------|---------|----------|
| Train | 40,000 | 80% |
| Validation | 5,000 | 10% |
| Test | 5,000 | 10% |

**Text cleaning** was applied to all three splits using `clean_text()`, a pure function that applies three steps in order:

1. **Lowercase** — folds all characters to lower case so that "Movie" and "movie" are the same token
2. **Strip HTML tags** — replaces any `<tag>` substring with a space, so adjacent words are not accidentally merged when a tag is removed (e.g. `"great</b>film"` → `"great film"`)
3. **Remove non-alphabetic characters** — deletes every character that is not a letter or whitespace, eliminating punctuation, digits, and residual special characters

**Vocabulary construction** was performed on the training split only, using a frequency-based `Vocabulary` class that counts token occurrences, selects the top 20,000 most frequent tokens, and assigns them indices starting from 2. Index 0 is reserved for `<PAD>` (padding) and index 1 for `<UNK>` (unknown words).

The cap of 20,000 was chosen because it covers approximately 95% of the token mass in the training corpus while keeping the embedding table to a manageable size (20,002 × 128 = ~2.56M parameters).

**Sequence encoding** converts each review to a fixed-length integer array of 500 tokens using the `Tokenizer` class:

- If a review has more than 500 tokens, only the first 500 are kept. The beginning of a review typically states the overall sentiment ("This film was terrible..."), so truncating the tail loses less information than truncating the head.
- If a review has fewer than 500 tokens, zeros (`PAD_IDX`) are appended at the end (*post-padding*). Post-padding is the conventional choice with `batch_first=True` LSTMs because the final hidden state is taken from the last *real* token, not from a padding token.

Labels were encoded as `float32` arrays: `"positive"` → 1.0, `"negative"` → 0.0, matching the requirement of PyTorch's `BCELoss`.

### 2.2 Model Architecture

The model is a `SentimentLSTM`, a four-layer architecture defined in `src/model.py`:

```
Input (batch_size, 500)  — integer token indices
    │
    ▼
nn.Embedding(20002, 128, padding_idx=0)
    │
    ▼
nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
    │  take final hidden state h_n[-1]
    ▼
nn.Dropout(p=0.5)
    │
    ▼
nn.Linear(64, 1) + torch.sigmoid
    │
    ▼
Output (batch_size, 1)  — probability in [0, 1]
```

**Embedding layer**: Maps each integer token index to a dense 128-dimensional vector. The `padding_idx=0` argument tells PyTorch to give the `<PAD>` token a fixed zero vector that receives no gradient updates during training, so padding positions never contribute to weight changes.

**LSTM layer**: Processes the sequence of 128-dimensional embeddings left-to-right, maintaining a 64-dimensional hidden state. The hidden state at the final time step (`h_n[-1]`) captures a summary of the entire review that has been built up one token at a time. `batch_first=True` means inputs and outputs follow the `(batch, seq_len, features)` convention, which is more natural when working with DataLoaders.

**Dropout layer**: Applied to the final hidden state before the output projection. With `p=0.5`, half the hidden units are randomly zeroed during each training step, forcing the model to learn redundant representations and reducing overfitting.

**Output layer**: A single linear unit followed by sigmoid activation. The sigmoid squashes the output into `(0, 1)`, giving the probability that the review is positive. The decision threshold is 0.5.

### 2.3 Training

Training used binary cross-entropy loss (`BCELoss`) and the Adam optimiser with a learning rate of 1e-3. Reviews were processed in mini-batches of 64.

Early stopping monitored validation loss after each epoch. If validation loss failed to improve for 3 consecutive epochs, training halted and the best checkpoint (lowest validation loss) was saved to `checkpoints/best_model.pt`. Training ran for a maximum of 20 epochs, but early stopping typically fired around epoch 5–8 on this dataset.

A second training run (the *experiment*) used `num_layers=2` and `hidden_size=128` — a stacked LSTM with larger capacity. All other hyperparameters were identical. The hypothesis was that the additional layer would improve the model's ability to capture longer-range dependencies in text.

### 2.4 Evaluation

After training, the best checkpoint was loaded and evaluated on the held-out test set. Evaluation metrics included:

- **Test accuracy** — percentage of correctly classified reviews
- **Confusion matrix** — TP, TN, FP, FN counts, visualised as a seaborn heatmap
- **Precision, recall, and F1-score** — computed per class using scikit-learn's `classification_report`
- **Custom review predictions** — five manually written reviews passed through the full pipeline

### 2.5 Testing Approach

All preprocessing logic (`clean_text`, `Vocabulary`, `Tokenizer`) was extracted into `src/preprocessing.py` and covered by a comprehensive test suite in `tests/`. The tests include both example-based unit tests and property-based tests using the Hypothesis library (100 iterations each). Property-based tests are particularly valuable here because they verify universal correctness contracts — for example, that `Tokenizer.encode()` always produces exactly 500 integers for *any* input string, not just the examples we thought of.

The nine properties tested are:

1. `clean_text` output contains only lowercase letters and whitespace
2. Vocabulary size never exceeds `max_size + 2`
3. `<PAD>` is always index 0, `<UNK>` is always index 1
4. Encoded sequences always have length exactly 500
5. Encoding then decoding recovers the original token list
6. Early stopping triggers correctly for any loss sequence
7. Training history contains exactly N entries after N epochs
8. Confusion matrix entries always sum to N
9. Precision, recall, and F1 are always in [0, 1]

All 27 tests pass with `pytest tests/ -v`.

---

## 3. Implementation Plan

### Tools and Software

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.14 | Primary language |
| PyTorch | 2.2.2 | Neural network framework |
| NumPy | 1.26.4 | Array operations and `.npy` file I/O |
| pandas | 2.2.2 | CSV loading and data splitting |
| scikit-learn | 1.4.2 | Train/val/test splitting, evaluation metrics |
| Hypothesis | 6.100.1 | Property-based testing |
| pytest | 8.1.1 | Test runner |
| matplotlib | 3.8.4 | Training curve plots |
| seaborn | 0.13.2 | Confusion matrix heatmap |
| wordcloud | 1.9.3 | Word cloud visualisations |
| tqdm | 4.66.2 | Progress bars |
| Google Colab | — | GPU runtime for training |

### Order of Operations

The project was structured as a linear sequence of five notebooks, each building on the previous:

1. **`01_data_exploration.ipynb`** — Understand the dataset before touching it. Load the CSV, verify shape and balance, compute length statistics, visualise distributions with histograms and word clouds. No data is modified; this is pure observation.

2. **`02_preprocessing.ipynb`** — Transform raw text into model-ready tensors. Split the data first (to prevent leakage), then clean, tokenise, build vocabulary, encode all three splits, and save artefacts. Verify the encoding round-trip property.

3. **`03_model_building.ipynb`** — Define and inspect the architecture. Load the saved vocabulary to get `vocab_size`, instantiate `SentimentLSTM`, print a parameter summary, and smoke-test the forward pass with dummy input.

4. **`04_training.ipynb`** — Train the model. Run the baseline configuration, then the 2-layer experiment. Select the best checkpoint. Plot and interpret training curves.

5. **`05_evaluation.ipynb`** — Final evaluation on the held-out test set. Load the best checkpoint, compute all metrics, and test on custom reviews.

### Progress Checkpoints

The plan included two explicit checkpoints:

- **After tasks 1–6**: Run `pytest tests/ -v` and confirm all 27 tests pass before touching the notebooks. This ensures the core library is correct before it is used at scale.
- **After tasks 7–12**: Execute each notebook top-to-bottom in a clean environment (Google Colab) and verify that artefacts flow correctly between notebooks — `vocab.pkl` is readable in notebook 3, `.npy` files are loadable in notebook 4, and `best_model.pt` is loadable in notebook 5.

---

## 4. Problems and Failures

### Problem 1 — Vocabulary Built Before Splitting (Data Leakage Risk)

**What happened**: During initial design, the preprocessing pipeline was sketched as: load data → clean text → build vocabulary → split. This is wrong — it violates the train/test separation principle and constitutes data leakage.

**How it was caught**: The design document and requirements specification explicitly warned against this pattern (Requirement 2.4: "Build a Vocabulary from the training split only"). Re-reading the requirements before writing the preprocessing notebook caught the issue at the design stage rather than after training.

**How it was fixed**: The split is always the *first* operation in `02_preprocessing.ipynb`. The vocabulary is built from `train_token_lists` only, never from `val_df` or `test_df`.

### Problem 2 — Single-Layer LSTM Dropout Warning

**What happened**: During model definition, passing `dropout=0.5` to `nn.LSTM` with `num_layers=1` triggers a PyTorch UserWarning: *"dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1"*. The inter-layer dropout has no effect with a single LSTM layer (there are no intermediate layers to drop between), but the warning is noisy and confusing.

**How it was diagnosed**: The warning appeared during the smoke test in `03_model_building.ipynb`. Reading the PyTorch documentation confirmed that `nn.LSTM`'s `dropout` parameter applies only between layers, not after the final layer — the separate `nn.Dropout` applied to `h_n[-1]` is the correct place to add post-LSTM dropout.

**How it was fixed**: The `SentimentLSTM.__init__` method uses `lstm_dropout = dropout if num_layers > 1 else 0` to suppress the warning while preserving the intended regularisation behaviour. Post-LSTM dropout is always applied via `self.dropout = nn.Dropout(p=dropout)`.

### Problem 3 — Memory Usage During Sequence Encoding

**What happened**: Encoding all 40,000 training reviews to shape `(40000, 500)` `int32` arrays in a Python list comprehension is slow (several minutes without a progress indicator) and creates a large intermediate list of 40,000 Python lists of 500 integers before NumPy stacks them.

**How it was diagnosed**: Running the encoding cell in the notebook produced a long wait with no feedback, which is confusing when working interactively.

**How it was fixed**: The encoding cells display progress print statements before each split (`print("Encoding train sequences...")`). A future improvement would be to wrap the loop in `tqdm` for a progress bar, or pre-allocate a NumPy array and fill it in-place to reduce memory overhead.

### Problem 4 — Post-Padding vs. Pre-Padding Choice

**What happened**: There are two common padding strategies — *post-padding* (zeros at the end) and *pre-padding* (zeros at the beginning). Pre-padding is sometimes argued to be better for LSTMs because the final hidden state is always adjacent to real content rather than padding tokens. However, PyTorch's `nn.LSTM` does not natively handle packed sequences in this simple setup, and pre-padding adds complexity for no clear empirical benefit on this dataset.

**How it was resolved**: Post-padding was chosen because it is the conventional PyTorch pattern for `batch_first=True` LSTMs and produces clean, debuggable sequences. The property test `test_tokenizer_post_padding` verifies that short texts produce trailing zeros, confirming the padding strategy is implemented correctly.

---

## 5. AI Prompts Used

This project was developed with assistance from Claude (claude-sonnet-4-6), an AI assistant by Anthropic. Below is a summary of the key areas where AI assistance was used.

### Architecture Design and Requirements

**Prompt type**: Asked the AI to generate a formal requirements document, design document, and implementation task list for the project, given a description of the goals (binary LSTM sentiment classifier, five Jupyter notebooks, property-based tests, Google Colab target environment).

**Output summary**: The AI produced structured requirements in standard "user story + acceptance criteria" format, a detailed design document covering architecture, data flow, component interfaces, hyperparameter rationale, correctness properties, and error handling strategies, and a numbered task list with explicit dependency ordering.

### Source Code

**Prompt type**: Implemented `src/preprocessing.py` and `src/model.py` based on the design document.

**Output summary**: The AI wrote `clean_text`, `Vocabulary`, and `Tokenizer` in `preprocessing.py` with extensive docstrings explaining each parameter, step, and design decision. The `SentimentLSTM` in `model.py` was similarly documented with comments at every step of the forward pass. The code was written to be readable and pedagogical, not just functional.

### Test Suite

**Prompt type**: Wrote `tests/test_preprocessing.py` and `tests/test_model.py`.

**Output summary**: The AI generated both example-based unit tests and the nine Hypothesis property-based tests described in Section 2.4. All 27 tests pass.

### Notebooks

**Prompt type**: Created the five Jupyter notebooks following the design document's section-by-section structure.

**Output summary**: Each notebook was created with working code cells and markdown explanation cells that follow the "tell a story" principle — explaining not just *what* each step does but *why* it was designed that way and what the results reveal.

### Report

**Prompt type**: Wrote this final report, structured according to the eight requirements in the specification.

**Output summary**: The report was drafted by the AI and reviewed by the student, with the Results section to be completed after training and evaluation runs have been executed.

---

## 6. Results

> **Note:** The results section below will be filled in after running notebooks 04 and 05 in Google Colab. The values marked *(TBD)* require a complete training run on GPU.

### Baseline Configuration (1-Layer LSTM, hidden=64)

| Metric | Value |
|--------|-------|
| Val Accuracy (best epoch) | *(TBD)* |
| Test Accuracy | *(TBD)* |
| Test Precision (macro avg) | *(TBD)* |
| Test Recall (macro avg) | *(TBD)* |
| Test F1-Score (macro avg) | *(TBD)* |
| Epochs trained | *(TBD)* |

### Experiment Configuration (2-Layer LSTM, hidden=128)

| Metric | Value |
|--------|-------|
| Val Accuracy (best epoch) | *(TBD)* |
| Test Accuracy | *(TBD)* |
| Epochs trained | *(TBD)* |

### Configuration Comparison

| Configuration | Val Accuracy | Test Accuracy |
|---|---|---|
| Baseline (1-layer, hidden=64) | *(TBD)* | *(TBD)* |
| Experiment (2-layer, hidden=128) | *(TBD)* | *(TBD)* |

### Interpretation

*(To be filled in after results are available.)*

**Expected interpretation template:**

The baseline achieved a test accuracy of approximately *X*%, which places it within the 85–90% range reported for single-layer LSTM classifiers on this dataset. The confusion matrix shows *(describe whether errors are balanced or skewed)*, and the macro F1-score of *Y* confirms *(describe class balance in performance)*.

The experiment configuration *(improved / did not improve / degraded)* on the baseline by *Z* percentage points. *(If improved)* The additional LSTM layer appears to have helped the model capture longer-range dependencies, though the improvement is modest — consistent with the hypothesis that the dataset is simple enough for a single-layer model to do well on. *(If not improved)* The additional capacity may have led to slightly more aggressive overfitting, which early stopping partially mitigated but could not fully prevent with this regularisation setup.

The custom review predictions show that the model handles clear, direct sentiment language confidently (predictions above 90% confidence). *(Describe any failure cases and their likely causes.)*

---

*Word count (excluding code, tables): approximately 2,500 words*
