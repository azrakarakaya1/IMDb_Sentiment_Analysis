"""
tests/test_model.py — Unit tests for src/model.py (SentimentLSTM).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from src.model import SentimentLSTM


@pytest.fixture
def model():
    """Default SentimentLSTM with vocab_size=1000."""
    return SentimentLSTM(vocab_size=1000)


def test_model_output_shape(model):
    x = torch.randint(0, 1000, (4, 500))
    assert model(x).shape == (4, 1)


def test_model_output_is_logit(model):
    x = torch.randint(0, 1000, (4, 500))
    probs = torch.sigmoid(model(x))
    assert probs.min().item() >= 0.0
    assert probs.max().item() <= 1.0


def test_model_embedding_dim(model):
    assert model.embedding.embedding_dim == 128


def test_model_hidden_size(model):
    assert model.lstm.hidden_size == 64


def test_model_dropout_rate(model):
    assert model.dropout.p == 0.5


def test_model_padding_idx(model):
    assert model.embedding.padding_idx == 0


def test_model_batch_norm_enabled_by_default(model):
    assert model.batch_norm is not None


def test_model_batch_norm_disabled():
    m = SentimentLSTM(vocab_size=1000, use_batch_norm=False)
    assert m.batch_norm is None
    x = torch.randint(0, 1000, (4, 500))
    assert m(x).shape == (4, 1)


def test_model_three_layer_with_batch_norm():
    m = SentimentLSTM(vocab_size=1000, num_layers=3, hidden_size=256)
    x = torch.randint(0, 1000, (8, 500))
    assert m(x).shape == (8, 1)
