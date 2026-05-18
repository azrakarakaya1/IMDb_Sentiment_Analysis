"""
app/app.py — Local web server for the IMDb Sentiment Analyzer.

Usage (from repo root):
    pip install flask
    python app/app.py

Then open http://localhost:5000 in your browser.

Prerequisites:
    - data/vocab.pkl            (produced by 02_preprocessing.ipynb)
    - checkpoints/best_model.pt (produced by 04_training_and_evaluation.ipynb)
"""

import sys, os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from flask import Flask, request, jsonify, send_from_directory
import torch
import numpy as np

from src.preprocessing import Vocabulary, Tokenizer
from src.model import SentimentLSTM

# ── Paths ─────────────────────────────────────────────────────────────────
VOCAB_PATH      = os.path.join(REPO_ROOT, 'data', 'vocab.pkl')
CHECKPOINT_PATH = os.path.join(REPO_ROOT, 'data', 'checkpoints', 'best_model.pt')
TEMPLATES_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

# ── Load model at startup ─────────────────────────────────────────────────
print("Loading vocabulary and model…")

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab     = Vocabulary.load(VOCAB_PATH)
tokenizer = Tokenizer(vocab=vocab, max_len=500)

state_dict  = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
num_layers  = sum(1 for k in state_dict if k.startswith('lstm.weight_hh_l'))
hidden_size = state_dict['fc.weight'].shape[1]
use_bn      = 'batch_norm.weight' in state_dict

model = SentimentLSTM(
    vocab_size     = len(vocab),
    embedding_dim  = 128,
    hidden_size    = hidden_size,
    num_layers     = num_layers,
    dropout        = 0.4,
    use_batch_norm = use_bn,
).to(device)
model.load_state_dict(state_dict)
model.eval()

print(f"  Architecture : {num_layers}-layer LSTM, hidden={hidden_size}, BatchNorm={use_bn}")
print(f"  Vocabulary   : {len(vocab):,} tokens")
print(f"  Device       : {device}")


# ── Saliency computation ──────────────────────────────────────────────────
def compute_saliency(text: str):
    """
    Returns (tokens, signed_importance, probability, logit).

    signed_importance[i] > 0  →  token i pushes the logit toward positive
    signed_importance[i] < 0  →  token i pushes the logit toward negative
    Magnitudes are normalised to [-1, 1] relative to the strongest token.
    """
    tokens = tokenizer.tokenize(text)
    n      = min(len(tokens), 500)
    tokens = tokens[:n]

    encoded = tokenizer.encode(text)
    x       = torch.tensor([encoded], dtype=torch.long).to(device)

    # Forward pass keeping the computation graph so we can backprop
    embed = model.embedding(x)      # (1, 500, embed_dim)
    embed.retain_grad()

    _, (h_n, _) = model.lstm(embed)
    last_hidden  = h_n[-1]
    if model.batch_norm is not None:
        last_hidden = model.batch_norm(last_hidden)
    logit = model.fc(model.dropout(last_hidden))   # (1, 1)

    model.zero_grad()
    logit.backward()

    # input × gradient summed over embedding dimension → signed scalar per token
    raw_imp   = (embed.grad * embed).sum(dim=-1).squeeze(0).detach().cpu().numpy()
    token_imp = raw_imp[500 - n:]                  # actual tokens only (pre-padding)

    max_abs   = max(float(np.abs(token_imp).max()), 1e-8)
    norm_imp  = (token_imp / max_abs).tolist()

    prob = torch.sigmoid(logit).item()
    return tokens, norm_imp, prob, logit.item()


# ── Flask app ─────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route('/')
def index():
    return send_from_directory(TEMPLATES_DIR, 'index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    text = (data.get('text') or '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        tokens, scores, prob, logit_val = compute_saliency(text)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    oov = [t not in vocab._token_to_idx for t in tokens]

    return jsonify({
        'sentiment'  : 'POSITIVE' if prob >= 0.5 else 'NEGATIVE',
        'confidence' : float(prob if prob >= 0.5 else 1 - prob),
        'probability': float(prob),
        'logit'      : float(logit_val),
        'tokens'     : tokens,
        'scores'     : scores,
        'oov'        : oov,
    })


@app.route('/timeline', methods=['POST'])
def timeline():
    data = request.get_json(force=True)
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        tokens_list = tokenizer.tokenize(text)
        n           = min(len(tokens_list), 500)
        tokens_list = tokens_list[:n]

        encoded = tokenizer.encode(text)
        x       = torch.tensor([encoded], dtype=torch.long).to(device)

        with torch.no_grad():
            embed     = model.embedding(x)          # (1, 500, embed_dim)
            output, _ = model.lstm(embed)            # (1, 500, hidden_size)
            hidden    = output[0, 500 - n:]          # (n, hidden_size)
            if model.batch_norm is not None:
                hidden = model.batch_norm(hidden)
            probs = torch.sigmoid(model.fc(hidden)).squeeze(-1).cpu().tolist()

        return jsonify({'tokens': tokens_list, 'probs': probs})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    print('\nServer ready — open http://localhost:5000\n')
    app.run(debug=False, port=5000, host='127.0.0.1')
