"""Classify your own sentences with the saved models.

    python predict.py "Operating profit rose to EUR 12 mn from EUR 8 mn." \
                      "The company will cut 300 jobs."
"""
import sys
from pathlib import Path

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import data
import finetune

HERE = Path(__file__).resolve().parent


def load():
    path = HERE / "models" / "distilbert"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return tok, model, joblib.load(HERE / "models" / "tfidf_lr.joblib")


def classify(sentences: list[str], tok, model, tfidf) -> list[dict]:
    pieces = tok(sentences, truncation=True, max_length=finetune.MAX_TOKENS)["input_ids"]
    bert = finetune.predict_proba(model, pieces, tok.pad_token_id, torch.device("cpu"))
    classic = tfidf.predict_proba(sentences)
    return [{"text": s,
             "distilbert": data.LABELS[b.argmax()], "distilbert_confidence": float(b.max()),
             "tfidf": data.LABELS[c.argmax()], "tfidf_confidence": float(c.max())}
            for s, b, c in zip(sentences, bert, classic)]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for r in classify(sys.argv[1:], *load()):
        print(f"{r['distilbert']:>8} {r['distilbert_confidence']:.1%}  "
              f"(TF-IDF: {r['tfidf']} {r['tfidf_confidence']:.1%})  {r['text']}")
