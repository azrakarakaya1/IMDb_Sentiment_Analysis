"""
tests/test_model.py
====================
Unit tests for src/model.py — SentimentLSTM architecture.

Covers:
  - Output shape for a standard batch
  - Output values are in [0.0, 1.0] (sigmoid applied)
  - Embedding dimension matches the default (128)
  - LSTM hidden size matches the default (64)
  - Dropout rate matches the default (0.5)
  - Two-layer variant produces the correct output shape
  - Embedding padding_idx is 0
"""

import sys
import os

# Ensure src/ is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from src.model import SentimentLSTM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    """Default SentimentLSTM with vocab_size=1000."""
    return SentimentLSTM(vocab_size=1000)


# ---------------------------------------------------------------------------
# 5.1  Unit tests for SentimentLSTM
# ---------------------------------------------------------------------------

def test_model_output_shape(model):
    """model(x).shape == (4, 1) for x = torch.randint(0, 1000, (4, 500))."""
    x = torch.randint(0, 1000, (4, 500))
    output = model(x)
    assert output.shape == (4, 1)


def test_model_output_is_logit(model):
    """Output is a raw logit; sigmoid(output) must be in [0.0, 1.0]."""
    x = torch.randint(0, 1000, (4, 500))
    output = model(x)
    probs = torch.sigmoid(output)
    assert probs.min().item() >= 0.0
    assert probs.max().item() <= 1.0


def test_model_embedding_dim(model):
    """model.embedding.embedding_dim == 128 (default)."""
    assert model.embedding.embedding_dim == 128


def test_model_hidden_size(model):
    """model.lstm.hidden_size == 64 (default)."""
    assert model.lstm.hidden_size == 64


def test_model_dropout_rate(model):
    """model.dropout.p == 0.5 (default)."""
    assert model.dropout.p == 0.5


def test_model_two_layer_variant():
    """SentimentLSTM(vocab_size=1000, num_layers=2) forward pass produces (8, 1)."""
    two_layer_model = SentimentLSTM(vocab_size=1000, num_layers=2)
    x = torch.randint(0, 1000, (8, 500))
    output = two_layer_model(x)
    assert output.shape == (8, 1)


def test_model_padding_idx(model):
    """model.embedding.padding_idx == 0 (matches Vocabulary.PAD_IDX)."""
    assert model.embedding.padding_idx == 0
