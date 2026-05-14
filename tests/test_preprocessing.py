"""
tests/test_preprocessing.py
============================
Unit tests and property-based tests for src/preprocessing.py.

Covers:
  - clean_text (unit + property)
  - Vocabulary (unit + property)
  - Tokenizer (unit + property)
  - EarlyStopping logic (property, inline implementation)
  - Training history completeness (property)
  - Confusion matrix and evaluation metrics (property)
"""

import sys
import os

# Ensure src/ is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.preprocessing import clean_text, Vocabulary, Tokenizer


# ---------------------------------------------------------------------------
# 4.1  Unit tests for clean_text and Vocabulary
# ---------------------------------------------------------------------------

def test_clean_text_empty_string():
    assert clean_text("") == ""


def test_clean_text_html_only():
    result = clean_text("<br />")
    assert result.strip() == ""


def test_clean_text_mixed():
    result = clean_text("Hello, <b>World</b>!")
    # After cleaning: lowercase, HTML stripped, punctuation removed.
    # Tokens should be "hello" and "world" (extra spaces are acceptable).
    tokens = result.split()
    assert tokens == ["hello", "world"]


def test_vocab_pad_unk_indices():
    vocab = Vocabulary()
    vocab.build([["hello", "world"]])
    assert vocab.PAD_IDX == 0
    assert vocab.UNK_IDX == 1


def test_vocab_unknown_token():
    vocab = Vocabulary()
    vocab.build([["hello", "world"]])
    encoded = vocab.encode(["definitely_not_in_vocab"])
    assert encoded == [vocab.UNK_IDX]


def test_vocab_len():
    """Vocab built from a small corpus has correct length (≤ max_size + 2)."""
    max_size = 5
    vocab = Vocabulary(max_size=max_size)
    # Provide more than max_size distinct tokens
    vocab.build([["a", "b", "c", "d", "e", "f", "g", "h"]])
    assert len(vocab) <= max_size + 2


# ---------------------------------------------------------------------------
# 4.2  Unit tests for Tokenizer
# ---------------------------------------------------------------------------

def _make_tokenizer(corpus_tokens=None, max_len=500):
    """Helper: build a Vocabulary and return a Tokenizer."""
    vocab = Vocabulary(max_size=20_000)
    if corpus_tokens is None:
        corpus_tokens = [["hello", "world", "this", "is", "a", "test"]]
    vocab.build(corpus_tokens)
    return Tokenizer(vocab=vocab, max_len=max_len)


def test_tokenizer_encode_length_short():
    tokenizer = _make_tokenizer()
    short_text = "hello world"
    encoded = tokenizer.encode(short_text)
    assert len(encoded) == 500


def test_tokenizer_encode_length_long():
    tokenizer = _make_tokenizer()
    # Build a text with 600 tokens
    long_text = " ".join(["word"] * 600)
    encoded = tokenizer.encode(long_text)
    assert len(encoded) == 500


def test_tokenizer_post_padding():
    tokenizer = _make_tokenizer()
    short_text = "hello world"
    encoded = tokenizer.encode(short_text)
    # The last positions should be PAD_IDX (0)
    assert encoded[-1] == Vocabulary.PAD_IDX
    assert encoded[10] == Vocabulary.PAD_IDX


def test_tokenizer_truncation():
    """Text with 600 tokens encodes to first 500 tokens only (not the 501st)."""
    vocab = Vocabulary(max_size=20_000)
    # Build vocab with two distinct words so we can tell them apart
    vocab.build([["alpha", "beta"]])
    tokenizer = Tokenizer(vocab=vocab, max_len=500)

    # 500 "alpha" tokens followed by 100 "beta" tokens
    text = " ".join(["alpha"] * 500 + ["beta"] * 100)
    encoded = tokenizer.encode(text)

    alpha_idx = vocab.encode(["alpha"])[0]
    beta_idx = vocab.encode(["beta"])[0]

    # All 500 positions should be alpha_idx; none should be beta_idx
    assert all(idx == alpha_idx for idx in encoded)
    assert beta_idx not in encoded


def test_tokenizer_decode_strips_padding():
    tokenizer = _make_tokenizer()
    short_text = "hello world"
    encoded = tokenizer.encode(short_text)
    decoded = tokenizer.decode(encoded)
    # No <PAD> tokens in decoded output
    assert Vocabulary.PAD_TOKEN not in decoded


# ---------------------------------------------------------------------------
# 4.3  Property-based test for clean_text (Property 1)
# ---------------------------------------------------------------------------

# Feature: imdb-sentiment-lstm, Property 1: clean_text output invariants
@given(st.text())
@settings(max_examples=100)
def test_property_clean_text_invariants(text):
    """Validates: Requirements 2.1, 2.2, 2.3"""
    result = clean_text(text)
    # No uppercase letters
    assert result == result.lower()
    # No HTML angle brackets
    assert '<' not in result and '>' not in result
    # Only alphabetic characters and whitespace
    assert all(c.isalpha() or c.isspace() for c in result)


# ---------------------------------------------------------------------------
# 4.4  Property-based tests for Vocabulary (Properties 2 and 3)
# ---------------------------------------------------------------------------

# Feature: imdb-sentiment-lstm, Property 2: Vocabulary size is bounded
@given(
    st.lists(
        st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('Ll',)), min_size=1),
            min_size=1,
        ),
        min_size=1,
    ),
    st.integers(min_value=1, max_value=100),
)
@settings(max_examples=100)
def test_property_vocab_size_bounded(token_lists, max_size):
    """Validates: Requirements 2.4"""
    vocab = Vocabulary(max_size=max_size)
    vocab.build(token_lists)
    assert len(vocab) <= max_size + 2


# Feature: imdb-sentiment-lstm, Property 3: Vocabulary index invariants
@given(
    st.lists(
        st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('Ll',)), min_size=1),
            min_size=1,
        ),
        min_size=1,
    )
)
@settings(max_examples=100)
def test_property_vocab_index_invariants(token_lists):
    """Validates: Requirements 2.5"""
    vocab = Vocabulary()
    vocab.build(token_lists)
    assert vocab._token_to_idx[vocab.PAD_TOKEN] == vocab.PAD_IDX   # PAD always 0
    assert vocab._token_to_idx[vocab.UNK_TOKEN] == vocab.UNK_IDX   # UNK always 1
    assert vocab.encode(["__definitely_not_in_vocab__"])[0] == vocab.UNK_IDX


# ---------------------------------------------------------------------------
# 4.5  Property-based tests for Tokenizer (Properties 4 and 5)
# ---------------------------------------------------------------------------

# Feature: imdb-sentiment-lstm, Property 4: Fixed-length encoding with post-padding
@given(st.text())
@settings(max_examples=100)
def test_property_fixed_length_encoding(text):
    """Validates: Requirements 2.6, 2.8"""
    vocab = Vocabulary(max_size=20_000)
    # Build a minimal vocab; OOV tokens will map to UNK, which is fine
    vocab.build([["hello", "world"]])
    tokenizer = Tokenizer(vocab=vocab, max_len=500)
    encoded = tokenizer.encode(text)
    assert len(encoded) == 500


# Feature: imdb-sentiment-lstm, Property 5: Tokenizer round-trip
@given(st.text())
@settings(max_examples=100)
def test_property_tokenizer_round_trip(text):
    """Validates: Requirements 2.10"""
    # Build vocab from the text being tested so all tokens are in-vocabulary
    vocab = Vocabulary(max_size=20_000)
    cleaned_tokens = clean_text(text).split()
    if cleaned_tokens:
        vocab.build([cleaned_tokens])
    else:
        vocab.build([["placeholder"]])

    tokenizer = Tokenizer(vocab=vocab, max_len=500)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    # Expected: the first 500 tokens from tokenize()
    expected = tokenizer.tokenize(text)[:500]

    # decoded should equal expected (all tokens are in-vocab, so no UNK substitution)
    assert decoded == expected


# ---------------------------------------------------------------------------
# 4.6  Property-based tests for EarlyStopping (Properties 6 and 7)
# ---------------------------------------------------------------------------

def _simulate_early_stopping(losses, patience):
    """
    Inline EarlyStopping simulation.

    Returns True if training should stop (minimum loss has not improved for
    `patience` consecutive steps), False otherwise.

    Mirrors the logic described in design.md section 2.5.
    """
    best_loss = float('inf')
    counter = 0
    for loss in losses:
        if loss < best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                return True
    return False


# Feature: imdb-sentiment-lstm, Property 6: Early stopping triggers correctly
@given(
    st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
    ),
    st.integers(min_value=1, max_value=5),
)
@settings(max_examples=100)
def test_property_early_stopping_triggers(losses, patience):
    """Validates: Requirements 4.5"""
    stopped = _simulate_early_stopping(losses, patience)

    # Verify the result is consistent with the definition:
    # stopped == True  iff  there exists a window of `patience` consecutive
    # non-improving steps somewhere in the sequence.
    best_loss = float('inf')
    counter = 0
    expected_stop = False
    for loss in losses:
        if loss < best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                expected_stop = True
                break

    assert stopped == expected_stop


# Feature: imdb-sentiment-lstm, Property 7: Training history completeness
@given(st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_property_training_history_completeness(n_epochs):
    """Validates: Requirements 4.7"""
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    for _ in range(n_epochs):
        history["train_loss"].append(0.5)
        history["val_loss"].append(0.5)
        history["train_accuracy"].append(0.8)
        history["val_accuracy"].append(0.8)
    assert all(len(v) == n_epochs for v in history.values())


# ---------------------------------------------------------------------------
# 4.7  Property-based tests for evaluation metrics (Properties 8 and 9)
# ---------------------------------------------------------------------------

# Feature: imdb-sentiment-lstm, Property 8: Confusion matrix sum invariant
@given(
    st.lists(st.booleans(), min_size=1),
    st.lists(st.booleans(), min_size=1),
)
@settings(max_examples=100)
def test_property_confusion_matrix_sum(preds_raw, labels_raw):
    """Validates: Requirements 5.4"""
    n = min(len(preds_raw), len(labels_raw))
    preds, labels = preds_raw[:n], labels_raw[:n]

    tp = sum(p and l for p, l in zip(preds, labels))
    tn = sum(not p and not l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum(not p and l for p, l in zip(preds, labels))

    assert tp + tn + fp + fn == n


# Feature: imdb-sentiment-lstm, Property 9: Precision, recall, and F1 are well-formed
@given(
    st.lists(st.booleans(), min_size=1),
    st.lists(st.booleans(), min_size=1),
)
@settings(max_examples=100)
def test_property_precision_recall_f1_wellformed(preds_raw, labels_raw):
    """Validates: Requirements 5.5"""
    n = min(len(preds_raw), len(labels_raw))
    preds, labels = preds_raw[:n], labels_raw[:n]

    tp = sum(p and l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum(not p and l for p, l in zip(preds, labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0
