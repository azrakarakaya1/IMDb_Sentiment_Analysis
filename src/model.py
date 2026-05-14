"""
src/model.py — SentimentLSTM architecture for IMDb binary sentiment classification.

This module defines the ``SentimentLSTM`` neural network, which is the core
predictive component of the IMDb sentiment analysis pipeline.  It sits between
the preprocessing stage (``src/preprocessing.py``) and the training notebook
(``04_training.ipynb``):

    raw text
        → clean_text / Tokenizer.encode   (src/preprocessing.py)
        → integer sequence of length 500  (torch.Tensor, shape (batch, 500))
        → SentimentLSTM.forward           (this file)
        → scalar probability in [0, 1]    (shape (batch, 1))

Architecture overview
---------------------
1. **Embedding layer** — converts each integer token index into a dense
   floating-point vector of size ``embedding_dim``.  The special PAD token
   (index 0) is given a fixed zero vector that is never updated during training.

2. **LSTM layer(s)** — processes the embedded sequence left-to-right,
   maintaining a hidden state that accumulates contextual information.  When
   ``num_layers > 1``, PyTorch stacks the LSTM layers internally and passes the
   full sequence between them (analogous to Keras's ``return_sequences=True``).

3. **Dropout** — applied to the final hidden state to reduce overfitting by
   randomly zeroing activations during training.

4. **Linear + Sigmoid** — projects the hidden state down to a single logit and
   squashes it into the (0, 1) range, giving the probability that the review is
   *positive*.

Usage example
-------------
>>> import torch
>>> from src.model import SentimentLSTM
>>> model = SentimentLSTM(vocab_size=20_000)
>>> x = torch.randint(0, 20_000, (64, 500))   # batch of 64 reviews, 500 tokens each
>>> probs = model(x)                           # shape (64, 1), values in [0, 1]
"""

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """Binary sentiment classifier built on an LSTM backbone.

    The network maps a batch of integer token sequences to a batch of
    probabilities in [0, 1], where values close to 1 indicate *positive*
    sentiment and values close to 0 indicate *negative* sentiment.

    Parameters
    ----------
    vocab_size : int
        Total number of tokens in the vocabulary (including the reserved
        ``<PAD>`` and ``<UNK>`` tokens).  This determines the number of rows
        in the embedding table.
    embedding_dim : int, optional
        Dimensionality of each token embedding vector.  Default ``128`` is a
        standard starting point that balances representational capacity and
        training speed.
    hidden_size : int, optional
        Number of features in the LSTM hidden state.  Default ``64`` is
        sufficient for binary classification on this dataset.
    num_layers : int, optional
        Number of stacked LSTM layers.  ``1`` (default) is the baseline;
        ``2`` is the experimental variant explored in ``04_training.ipynb``.
    dropout : float, optional
        Dropout probability applied *after* the LSTM (on the final hidden
        state).  When ``num_layers > 1``, the same rate is also applied
        *between* LSTM layers by PyTorch internally.  Default ``0.5``.
    padding_idx : int, optional
        Index of the PAD token in the vocabulary.  The embedding for this
        index is fixed at zero and receives no gradient updates.  Default
        ``0`` matches ``Vocabulary.PAD_IDX``.

    Layers
    ------
    embedding : nn.Embedding
        Token-index → dense-vector lookup table.
    lstm : nn.LSTM
        Recurrent layer(s) that process the embedded sequence.
    dropout : nn.Dropout
        Regularisation applied to the final hidden state.
    fc : nn.Linear
        Fully-connected projection from ``hidden_size`` to a single logit.

    Notes
    -----
    * ``batch_first=True`` is set on the LSTM so that input/output tensors
      follow the ``(batch, seq_len, features)`` convention, which is more
      natural when working with DataLoaders.
    * The inter-layer dropout inside ``nn.LSTM`` is only active when
      ``num_layers > 1``; passing ``dropout > 0`` with a single layer would
      trigger a PyTorch warning, so we guard against it explicitly.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.5,
        padding_idx: int = 0,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Layer 1: Embedding
        # ------------------------------------------------------------------
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        # ------------------------------------------------------------------
        # Layer 2: LSTM
        # ``batch_first=True``  → input shape is (batch, seq_len, embedding_dim)
        # inter-layer dropout only valid when num_layers > 1
        # ------------------------------------------------------------------
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # ------------------------------------------------------------------
        # Layer 3 (optional): Batch Normalisation
        # Normalises the final hidden state across the batch before the
        # linear layer.  Stabilises training, reduces internal covariate
        # shift, and acts as a mild regulariser.
        # ------------------------------------------------------------------
        self.batch_norm = nn.BatchNorm1d(hidden_size) if use_batch_norm else None

        # ------------------------------------------------------------------
        # Layer 4: Dropout
        # ------------------------------------------------------------------
        self.dropout = nn.Dropout(p=dropout)

        # ------------------------------------------------------------------
        # Layer 5: Fully-connected output layer
        # ------------------------------------------------------------------
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Integer tensor of token indices with shape
            ``(batch_size, seq_len)``.  Values must be in
            ``[0, vocab_size)``.

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``(batch_size, 1)`` containing raw logits
            (unbounded real values).  Apply ``torch.sigmoid()`` to obtain
            probabilities in ``(0, 1)``; use ``BCEWithLogitsLoss`` during
            training for numerical stability.
        """

        # Step 1 — Embed token indices
        # x:        (batch_size, seq_len)
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # Step 2 — Pass through LSTM
        # We discard the full output sequence (first return value) because we
        # only need the final hidden state for classification.
        # h_n shape: (num_layers, batch_size, hidden_size)
        # c_n shape: (num_layers, batch_size, hidden_size)  — cell state, unused
        _, (h_n, _) = self.lstm(embedded)

        # Step 3 — Extract the last layer's hidden state
        # h_n[-1] selects the hidden state from the topmost LSTM layer at the
        # final time step.
        # Shape: (batch_size, hidden_size)
        last_hidden = h_n[-1]

        # Step 4 — Batch normalisation (if enabled)
        if self.batch_norm is not None:
            last_hidden = self.batch_norm(last_hidden)

        # Step 5 — Dropout
        out = self.dropout(last_hidden)

        # Step 6 — Linear projection → raw logit (batch_size, 1)
        return self.fc(out)
