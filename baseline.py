"""Two baselines the transformer has to beat.

  majority   always answer the most common label in training ("neutral")
  tf-idf     word and word-pair counts, weighted by rarity, into a logistic
             regression — the classic text classifier

The regression's strength (C) and whether to re-weight the rare classes are
chosen on the VALIDATION set; the test set is scored once, by the chosen
model. Writes results/baseline.json and models/tfidf_lr.joblib.

    python baseline.py        # a few seconds
"""
import json
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

import data

HERE = Path(__file__).resolve().parent
C_GRID = [0.3, 1, 3, 10, 30, 100]
WEIGHTS = [None, "balanced"]


def tfidf_lr(c: float, weight) -> object:
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        LogisticRegression(C=c, class_weight=weight, max_iter=5000))


def main() -> None:
    df, _ = data.load()
    train, val, test = data.split(df)

    majority = train["label"].mode()[0]

    grid = []
    for c, weight in product(C_GRID, WEIGHTS):
        model = tfidf_lr(c, weight).fit(train["text"], train["label"])
        score = f1_score(val["label"], model.predict(val["text"]), average="macro")
        grid.append({"C": c, "class_weight": weight, "val_macro_f1": round(score, 4)})
    best = max(grid, key=lambda g: g["val_macro_f1"])      # first one wins a tie
    model = tfidf_lr(best["C"], best["class_weight"]).fit(train["text"], train["label"])

    start = time.perf_counter()
    for _ in range(20):
        model.predict_proba(test["text"])
    per_sentence = (time.perf_counter() - start) / (20 * len(test))

    (HERE / "models").mkdir(exist_ok=True)
    joblib.dump(model, HERE / "models" / "tfidf_lr.joblib")
    classes = list(model.classes_)
    assert classes == data.LABELS

    def probs(part):
        return np.round(model.predict_proba(part["text"]), 5).tolist()

    out = {"majority_label": majority,
           "grid": grid, "chosen": best,
           "test_fingerprint": data.fingerprint(test),
           "val_ids": val["id"].tolist(), "val_probs": probs(val),
           "test_ids": test["id"].tolist(), "test_probs": probs(test),
           "microseconds_per_sentence_cpu": round(per_sentence * 1e6, 1),
           "model_mb": round((HERE / "models" / "tfidf_lr.joblib").stat().st_size / 1e6, 1),
           "vocabulary": len(model[0].vocabulary_)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "baseline.json").write_text(json.dumps(out, indent=1))
    test_f1 = f1_score(test["label"], np.array(classes)[np.argmax(out["test_probs"], 1)],
                       average="macro")
    print(f"chosen C={best['C']} class_weight={best['class_weight']}  "
          f"val macro-F1 {best['val_macro_f1']:.3f}  test macro-F1 {test_f1:.3f}  "
          f"{out['microseconds_per_sentence_cpu']} µs/sentence  {out['model_mb']} MB")


if __name__ == "__main__":
    main()
