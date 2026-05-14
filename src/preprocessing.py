"""
src/preprocessing.py
====================
Text preprocessing utilities for the IMDb sentiment classification pipeline.

This module sits between the raw CSV data and the model. It provides three
components that are used in sequence:

1. ``clean_text`` — a pure function that normalises a raw review string by
   lowercasing it, stripping HTML tags, and removing every character that is
   not a letter or whitespace.

2. ``Vocabulary`` — builds a token-to-index mapping from a training corpus,
   reserves special indices for padding (0) and unknown tokens (1), and
   supports serialisation to disk so the same mapping can be reloaded in
   later notebooks without rebuilding.

3. ``Tokenizer`` — wraps ``clean_text`` and ``Vocabulary`` together and adds
   fixed-length encoding: every review is truncated or post-padded to exactly
   ``max_len`` integer indices, which is the format expected by
   ``nn.Embedding`` inside ``SentimentLSTM``.

Pipeline sketch::

    raw review (str)
        │  clean_text()
        ▼
    cleaned string (str)
        │  str.split()
        ▼
    token list (List[str])
        │  Vocabulary.encode()
        ▼
    index list (List[int])
        │  truncate / post-pad to max_len
        ▼
    fixed-length sequence (List[int], length == max_len)

All three components are pure or nearly-pure (``Vocabulary.build`` mutates
internal state but has no other side effects), which makes them straightforward
to unit-test and property-test with Hypothesis.
"""

import pickle
import re
from collections import Counter
from typing import List


# ---------------------------------------------------------------------------
# 1. clean_text
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Return a cleaned, normalised version of *text*.

    The three cleaning steps are applied in order:

    1. **Lowercase** — fold every character to lower case so that "Movie" and
       "movie" are treated as the same token.
    2. **Strip HTML tags** — replace any ``<tag>`` or ``</tag>`` substring
       with a single space so that HTML-formatted reviews (common in the IMDb
       dataset) do not leave stray angle-bracket characters.
    3. **Remove non-alphabetic characters** — delete every character that is
       not a lower-case letter or whitespace.  This removes punctuation,
       digits, and any residual special characters.

    Parameters
    ----------
    text:
        The raw input string.  May contain HTML, mixed case, punctuation, etc.

    Returns
    -------
    str
        A string containing only lower-case ASCII letters and whitespace.
        Multiple consecutive spaces may appear; callers that need a single
        space between tokens should call ``str.split()`` followed by
        ``" ".join(...)``.

    Examples
    --------
    >>> clean_text("Hello, <b>World</b>!")
    'hello  world '
    >>> clean_text("<br />Great film!!!")
    ' great film'
    >>> clean_text("")
    ''
    """
    # Step 1: fold to lower case
    text = text.lower()

    # Step 2: replace HTML tags with a space so adjacent words are not merged
    # The pattern <[^>]+> matches an opening '<', any characters that are not
    # '>', and a closing '>'.
    text = re.sub(r'<[^>]+>', ' ', text)

    # Step 3: delete every character that is not a lower-case letter or
    # whitespace (\s covers space, tab, newline, etc.)
    text = re.sub(r'[^a-z\s]', '', text)

    return text


# ---------------------------------------------------------------------------
# 2. Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Token-to-index mapping built from a training corpus.

    The vocabulary reserves two special indices:

    * **Index 0** — ``<PAD>`` padding token.  Used to fill sequences that are
      shorter than ``max_len`` so that every batch element has the same length.
    * **Index 1** — ``<UNK>`` unknown token.  Used for tokens that appear in
      validation/test data but were not seen (or were too rare) in training.

    All other tokens are assigned indices starting from 2, ordered by
    descending frequency in the training corpus.  If the corpus contains more
    than ``max_size`` distinct tokens, only the ``max_size`` most frequent ones
    are kept; the rest are silently mapped to ``<UNK>`` at encode time.

    Attributes
    ----------
    PAD_IDX : int
        Class-level constant — always 0.
    UNK_IDX : int
        Class-level constant — always 1.
    PAD_TOKEN : str
        String representation of the padding token.
    UNK_TOKEN : str
        String representation of the unknown token.

    Parameters
    ----------
    max_size:
        Maximum number of *non-special* tokens to keep.  The total vocabulary
        size (including ``<PAD>`` and ``<UNK>``) is at most ``max_size + 2``.
        Defaults to 20 000, which covers roughly 95 % of the token mass in the
        IMDb corpus.
    """

    # Class-level constants so callers can reference them without an instance
    PAD_IDX: int = 0
    UNK_IDX: int = 1
    PAD_TOKEN: str = "<PAD>"
    UNK_TOKEN: str = "<UNK>"

    def __init__(self, max_size: int = 20_000) -> None:
        self.max_size = max_size

        # These dicts are populated by build(); they are empty until then.
        self._token_to_idx: dict = {}
        self._idx_to_token: dict = {}

    # ------------------------------------------------------------------
    # Building the vocabulary
    # ------------------------------------------------------------------

    def build(self, token_lists: List[List[str]]) -> None:
        """Populate the vocabulary from a collection of token lists.

        Only the training split should be passed here.  Building from
        validation or test data would constitute data leakage.

        The method:

        1. Counts token frequencies across *all* lists in ``token_lists``.
        2. Selects the ``max_size`` most frequent tokens.
        3. Assigns index 0 to ``<PAD>`` and index 1 to ``<UNK>``.
        4. Assigns indices 2, 3, … to the selected tokens in descending
           frequency order.

        Calling ``build`` a second time on the same instance replaces the
        previous vocabulary entirely.

        Parameters
        ----------
        token_lists:
            A list of token lists, e.g. one inner list per training review.
            Each inner list is the output of ``str.split()`` on a cleaned
            review string.
        """
        # Count how often each token appears across the entire training corpus
        counter: Counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        # Keep only the top max_size tokens by frequency; most_common returns
        # a list of (token, count) pairs sorted by count descending.
        most_common = counter.most_common(self.max_size)

        # Reserve indices 0 and 1 for the special tokens, then assign the
        # remaining indices starting from 2.
        self._token_to_idx = {self.PAD_TOKEN: self.PAD_IDX,
                               self.UNK_TOKEN: self.UNK_IDX}
        self._idx_to_token = {self.PAD_IDX: self.PAD_TOKEN,
                               self.UNK_IDX: self.UNK_TOKEN}

        for idx, (token, _count) in enumerate(most_common, start=2):
            self._token_to_idx[token] = idx
            self._idx_to_token[idx] = token

    # ------------------------------------------------------------------
    # Encoding and decoding
    # ------------------------------------------------------------------

    def encode(self, tokens: List[str]) -> List[int]:
        """Map a list of token strings to a list of integer indices.

        Tokens that are not in the vocabulary are silently mapped to
        ``UNK_IDX`` (1).  This is the expected behaviour, not an error.

        Parameters
        ----------
        tokens:
            A list of token strings, typically the output of
            ``clean_text(text).split()``.

        Returns
        -------
        List[int]
            A list of the same length as *tokens*, where each element is the
            vocabulary index of the corresponding token (or ``UNK_IDX`` if the
            token is absent from the vocabulary).
        """
        # dict.get with a default is O(1) and avoids a KeyError for OOV tokens
        return [self._token_to_idx.get(token, self.UNK_IDX) for token in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Map a list of integer indices back to token strings.

        Indices that are not in the vocabulary (e.g. out-of-range values) are
        mapped to ``UNK_TOKEN``.

        Parameters
        ----------
        indices:
            A list of integer vocabulary indices.

        Returns
        -------
        List[str]
            A list of the same length as *indices* containing the corresponding
            token strings.
        """
        return [self._idx_to_token.get(idx, self.UNK_TOKEN) for idx in indices]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of entries in the vocabulary.

        This includes the two special tokens (``<PAD>`` and ``<UNK>``), so the
        return value is at most ``max_size + 2``.
        """
        return len(self._token_to_idx)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the entire ``Vocabulary`` object to *path* using pickle.

        The file at *path* is created (or overwritten) in binary mode.  The
        saved object can be restored with :meth:`load`.

        Parameters
        ----------
        path:
            File system path where the pickle file should be written, e.g.
            ``"data/vocab.pkl"``.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Deserialise a ``Vocabulary`` object from a pickle file at *path*.

        Parameters
        ----------
        path:
            File system path of a pickle file previously written by
            :meth:`save`.

        Returns
        -------
        Vocabulary
            The restored ``Vocabulary`` instance, including its internal
            ``_token_to_idx`` and ``_idx_to_token`` mappings.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.  Run ``02_preprocessing.ipynb`` first to
            generate ``data/vocab.pkl``.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# 3. Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    """Combines text cleaning, vocabulary encoding, and sequence padding.

    ``Tokenizer`` is the single entry point for converting a raw review string
    into the fixed-length integer sequence that ``SentimentLSTM`` expects.

    The encoding pipeline is::

        raw text (str)
            │  clean_text()
            ▼
        cleaned text (str)
            │  str.split()
            ▼
        token list (List[str])
            │  vocab.encode()
            ▼
        index list (List[int])
            │  truncate to max_len  (if len > max_len)
            │  post-pad with PAD_IDX  (if len < max_len)
            ▼
        fixed-length sequence (List[int], length == max_len)

    Padding strategy: **pre-padding** — padding tokens are prepended at the
    *beginning* of the sequence.  This ensures that ``h_n[-1]`` (the final
    LSTM hidden state used for classification) is computed after the last
    *real* token, not after a run of meaningless zero-embedding padding steps.

    Parameters
    ----------
    vocab:
        A built ``Vocabulary`` instance.
    max_len:
        The fixed output length for :meth:`encode`.  Sequences longer than
        ``max_len`` are truncated to their first ``max_len`` tokens; shorter
        sequences are post-padded with ``PAD_IDX`` (0).  Defaults to 500,
        which covers roughly the 90th percentile of IMDb review lengths.
    """

    def __init__(self, vocab: Vocabulary, max_len: int = 500) -> None:
        self.vocab = vocab
        self.max_len = max_len

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> List[str]:
        """Clean *text* and split it into a list of tokens.

        This is a two-step operation:

        1. ``clean_text(text)`` — lowercase, strip HTML, remove punctuation.
        2. ``str.split()`` — split on any whitespace, discarding empty strings.

        Parameters
        ----------
        text:
            Raw input string (may contain HTML, punctuation, mixed case).

        Returns
        -------
        List[str]
            A list of lower-case alphabetic tokens.  May be empty if *text*
            contains no alphabetic characters after cleaning.
        """
        # clean_text handles normalisation; split() handles tokenisation.
        # str.split() with no argument splits on any whitespace and ignores
        # leading/trailing whitespace, which is exactly what we want.
        return clean_text(text).split()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Convert *text* to a fixed-length list of vocabulary indices.

        Steps:

        1. Tokenise with :meth:`tokenize`.
        2. Encode tokens to indices with ``vocab.encode``.
        3. Truncate to the first ``max_len`` indices if the sequence is too
           long.
        4. Post-pad with ``PAD_IDX`` (0) if the sequence is too short.

        The output always has length exactly ``max_len``.

        Parameters
        ----------
        text:
            Raw input string.

        Returns
        -------
        List[int]
            A list of exactly ``max_len`` integer vocabulary indices.
        """
        tokens = self.tokenize(text)

        # Map tokens to indices; OOV tokens become UNK_IDX (1)
        indices = self.vocab.encode(tokens)

        # Truncate: keep only the first max_len indices
        indices = indices[:self.max_len]

        # Pre-pad: prepend PAD_IDX (0) so real content ends at position max_len-1.
        # h_n[-1] then reflects the hidden state after the last real token, not
        # after a run of zero-embedding padding steps.
        padding_needed = self.max_len - len(indices)
        indices = [Vocabulary.PAD_IDX] * padding_needed + indices

        return indices

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, indices: List[int]) -> List[str]:
        """Convert a list of vocabulary indices back to tokens, stripping padding.

        This is the inverse of :meth:`encode` (modulo information lost during
        truncation and UNK substitution).

        Steps:

        1. Map indices to token strings with ``vocab.decode``.
        2. Filter out ``<PAD>`` tokens so the result contains only real tokens.

        Parameters
        ----------
        indices:
            A list of integer vocabulary indices, typically the output of
            :meth:`encode`.

        Returns
        -------
        List[str]
            A list of token strings with all ``<PAD>`` entries removed.
        """
        tokens = self.vocab.decode(indices)

        # Remove padding tokens; they carry no semantic content
        return [token for token in tokens if token != Vocabulary.PAD_TOKEN]
