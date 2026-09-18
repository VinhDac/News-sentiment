"""Score every model on the test set, once, and draw the charts.

Reads what baseline.py and finetune.py saved (probabilities per sentence,
matched back to the split by sentence fingerprint), and writes
results/metrics.json and the charts. Nothing here trains or chooses anything
from the test set.

    python evaluate.py
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import data

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNS = RESULTS / "runs"
# Each sentence counts once, at the strongest agreement it reached (the
# dataset's own levels: more than 50%, 66%, 75% of annotators, or all).
BUCKETS = {0.50: "50", 0.66: "66", 0.75: "75", 1.00: "100"}
SHOWN = {"50": "Just over half", "66": "Over two-thirds", "75": "Over three-quarters",
         "100": "All"}
TARGET = 0.95          # accuracy wanted from the sentences the model labels on its own


def aligned(ids: list[str], probs: list, part: pd.DataFrame) -> np.ndarray:
    """Saved probabilities in the split's row order — refusing to guess if
    the saved results were made on a different split."""
    if sorted(ids) != sorted(part["id"]):
        raise SystemExit("saved results were made on a different split — re-run training")
    return pd.DataFrame(probs, index=ids).loc[part["id"]].to_numpy()


def scores(y: np.ndarray, pred: np.ndarray, agreement: np.ndarray) -> dict:
    f1 = f1_score(y, pred, average=None, labels=range(len(data.LABELS)), zero_division=0)
    out = {"accuracy": float((pred == y).mean()),
           "macro_f1": float(f1.mean()),
           **{f"f1_{name}": float(v) for name, v in zip(data.LABELS, f1)}}
    for level, name in BUCKETS.items():
        mask = agreement == level
        out[f"accuracy_agree_{name}"] = float((pred[mask] == y[mask]).mean())
    return out


def selective(val_p, val_y, test_p, test_y) -> dict:
    """The lowest confidence cut-off at which the VALIDATION sentences above
    it are at least TARGET accurate; then what that cut-off does on test."""
    conf = val_p.max(1)
    right = val_p.argmax(1) == val_y
    order = np.argsort(-conf, kind="stable")
    running = np.cumsum(right[order]) / np.arange(1, len(order) + 1)
    ok = np.nonzero(running >= TARGET)[0]
    threshold = float(conf[order][ok.max()]) if len(ok) else 1.0
    t_conf = test_p.max(1)
    kept = t_conf >= threshold
    v_kept = conf >= threshold
    return {"threshold": round(threshold, 4),
            "val_coverage": float(v_kept.mean()),
            "val_accuracy": float(right[v_kept].mean()),
            "test_coverage": float(kept.mean()),
            "test_accuracy": float((test_p.argmax(1)[kept] == test_y[kept]).mean()),
            "test_accuracy_rest": float((test_p.argmax(1)[~kept] == test_y[~kept]).mean())}


def curve(probs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-probs.max(1), kind="stable")
    right = (probs.argmax(1) == y)[order]
    n = np.arange(1, len(y) + 1)
    return n / len(y), np.cumsum(right) / n


def speed(test: pd.DataFrame) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import finetune
    path = HERE / "models" / "distilbert"
    tok = AutoTokenizer.from_pretrained(path)
    pieces = tok(list(test["text"]), truncation=True, max_length=finetune.MAX_TOKENS)["input_ids"]
    out = {"distilbert_mb": round(sum(f.stat().st_size for f in path.iterdir()) / 1e6, 1)}
    for name in ("mps", "cpu"):
        if name == "mps" and not torch.backends.mps.is_available():
            continue
        dev = torch.device(name)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(dev)
        finetune.predict_proba(model, pieces[:64], tok.pad_token_id, dev)       # warm-up
        start = time.perf_counter()
        finetune.predict_proba(model, pieces, tok.pad_token_id, dev)
        out[f"distilbert_sentences_per_second_{name}"] = round(
            len(pieces) / (time.perf_counter() - start))
    return out


def main() -> None:
    df, removed = data.load()
    train, val, test = data.split(df)
    label_id = {name: i for i, name in enumerate(data.LABELS)}
    val_y = val["label"].map(label_id).to_numpy()
    y = test["label"].map(label_id).to_numpy()
    agree = test["agreement"].to_numpy()

    base = json.loads((RESULTS / "baseline.json").read_text())
    summary = json.loads((RUNS / "summary.json").read_text())
    runs = {s: json.loads((RUNS / f"lr{summary['chosen_lr']:g}_seed{s}.json").read_text())
            for s in summary["seeds"]}
    grid = {lr: json.loads((RUNS / f"lr{float(lr):g}_seed{summary['seeds'][0]}.json").read_text())
            for lr in summary["learning_rates"]}

    majority = np.full(len(y), label_id[base["majority_label"]])
    tfidf_p = aligned(base["test_ids"], base["test_probs"], test)
    tfidf_val = aligned(base["val_ids"], base["val_probs"], val)
    bert_p = {s: aligned(r["test_ids"], r["test_probs"], test) for s, r in runs.items()}
    bert_val = {s: aligned(r["val_ids"], r["val_probs"], val) for s, r in runs.items()}
    keep = summary["saved_seed"]

    per_seed = {s: scores(y, p.argmax(1), agree) for s, p in bert_p.items()}
    keys = per_seed[keep].keys()
    metrics = {
        "data": {**removed, "train": len(train), "validation": len(val), "test": len(test),
                 "test_all_agree": int((agree == 1).sum()),
                 "label_share": df["label"].value_counts(normalize=True).round(4).to_dict()},
        "test_fingerprint": data.fingerprint(test),
        "majority": scores(y, majority, agree),
        "tfidf_lr": {**scores(y, tfidf_p.argmax(1), agree), "chosen": base["chosen"]},
        "distilbert": {
            "mean": {k: float(np.mean([per_seed[s][k] for s in per_seed])) for k in keys},
            "std": {k: float(np.std([per_seed[s][k] for s in per_seed], ddof=1)) for k in keys},
            "per_seed": {str(s): v for s, v in per_seed.items()},
            "learning_rates_val_macro_f1": summary["learning_rates"],
            "chosen_lr": summary["chosen_lr"],
            "best_epochs": {str(s): r["best_epoch"] for s, r in runs.items()},
            "train_minutes": {str(s): round(r["train_seconds"] / 60, 1) for s, r in runs.items()},
            "saved_seed": keep},
        "selective": {"target": TARGET,
                      "distilbert": selective(bert_val[keep], val_y, bert_p[keep], y),
                      "tfidf_lr": selective(tfidf_val, val_y, tfidf_p, y)},
        "speed": {"tfidf_lr_microseconds_per_sentence_cpu": base["microseconds_per_sentence_cpu"],
                  "tfidf_lr_mb": base["model_mb"], **speed(test)},
    }
    for name, r in (("baseline", base), *((f"run seed {s}", r) for s, r in runs.items())):
        if r["test_fingerprint"] != metrics["test_fingerprint"]:
            raise SystemExit(f"{name} was scored on a different test set")

    # The saved model's most confident mistakes: what is it getting wrong?
    p = bert_p[keep]
    wrong = np.nonzero(p.argmax(1) != y)[0]
    wrong = wrong[np.argsort(-p[wrong].max(1))]
    metrics["errors"] = [{"text": test["text"].iloc[i], "label": data.LABELS[y[i]],
                          "predicted": data.LABELS[p[i].argmax()],
                          "confidence": round(float(p[i].max()), 3),
                          "agreement": float(agree[i])} for i in wrong[:12]]
    metrics["confusion"] = confusion_matrix(y, p.argmax(1)).tolist()
    # How many of the mistakes are on sentences the annotators split over?
    split_share = [float((agree[q.argmax(1) != y] < 1).mean()) for q in bert_p.values()]
    metrics["distilbert"]["errors_where_annotators_split"] = {
        "mean_share": float(np.mean(split_share)),
        "per_seed": dict(zip(map(str, bert_p), split_share)),
        "test_share_of_split_sentences": float((agree < 1).mean())}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    charts(metrics, y, agree, tfidf_p, bert_p[keep], grid)

    d, t, m = metrics["distilbert"], metrics["tfidf_lr"], metrics["majority"]
    print(f"{'':12}{'accuracy':>10}{'macro-F1':>10}{'all agree':>11}")
    print(f"{'majority':12}{m['accuracy']:10.3f}{m['macro_f1']:10.3f}{m['accuracy_agree_100']:11.3f}")
    print(f"{'tf-idf + lr':12}{t['accuracy']:10.3f}{t['macro_f1']:10.3f}{t['accuracy_agree_100']:11.3f}")
    print(f"{'distilbert':12}{d['mean']['accuracy']:10.3f}{d['mean']['macro_f1']:10.3f}"
          f"{d['mean']['accuracy_agree_100']:11.3f}   (mean of {len(runs)} seeds; "
          f"std {d['std']['accuracy']:.3f} / {d['std']['macro_f1']:.3f})")


def charts(metrics, y, agree, tfidf_p, bert_p, grid) -> None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})
    colours = {"Always 'neutral'": "#c9c9c9", "TF-IDF + logistic regression": "#7aa6c2",
               "DistilBERT (fine-tuned)": "#1f4e79"}

    # 1. Accuracy by how strongly the human annotators agreed.
    groups = ["All test"] + [f"{SHOWN[name]}\nagreed" for name in BUCKETS.values()]
    counts = [len(y)] + [int((agree == level).sum()) for level in BUCKETS]
    rows = {"Always 'neutral'": metrics["majority"], "TF-IDF + logistic regression": metrics["tfidf_lr"],
            "DistilBERT (fine-tuned)": metrics["distilbert"]["mean"]}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.27
    for k, (name, s) in enumerate(rows.items()):
        values = [s["accuracy"]] + [s[f"accuracy_agree_{b}"] for b in BUCKETS.values()]
        bars = ax.bar(np.arange(len(groups)) + (k - 1) * width, values, width,
                      label=name, color=colours[name])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.0%}",
                    ha="center", fontsize=7.5)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([f"{g}\n({n} sentences)" for g, n in zip(groups, counts)], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy on the test set")
    ax.set_title("Accuracy follows how strongly the annotators agreed "
                 "(each sentence counted once, at its highest level)", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, ncol=3)
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy_by_agreement.png", dpi=150)
    plt.close(fig)

    # 2. Confusion matrix of the saved model.
    cm = np.array(metrics["confusion"])
    share = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(3), data.LABELS)
    ax.set_yticks(range(3), data.LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Labelled by annotators")
    ax.set_title("DistilBERT on the test set", loc="left")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion.png", dpi=150)
    plt.close(fig)

    # 3. Accuracy if the model only labels the sentences it is surest about.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, p, key in (("DistilBERT (fine-tuned)", bert_p, "distilbert"),
                         ("TF-IDF + logistic regression", tfidf_p, "tfidf_lr")):
        cov, acc = curve(p, y)
        ax.plot(cov, acc, color=colours[name], label=name)
        s = metrics["selective"][key]
        ax.plot(s["test_coverage"], s["test_accuracy"], "o", color=colours[name])
        ax.annotate(f"{s['test_coverage']:.0%} labelled, {s['test_accuracy']:.1%} right",
                    (s["test_coverage"], s["test_accuracy"]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8.5, color=colours[name])
    ax.axhline(TARGET, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("Share of test sentences the model labels (most confident first)")
    ax.set_ylabel("Accuracy on those sentences")
    ax.set_ylim(0.6, 1.01)
    ax.set_title(f"Confidence cut-off chosen on validation for {TARGET:.0%} accuracy", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "coverage.png", dpi=150)
    plt.close(fig)

    # 4. The learning-rate search, on validation only.
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for lr, run in grid.items():
        epochs = [h["epoch"] for h in run["history"]]
        ax.plot(epochs, [h["val_macro_f1"] for h in run["history"]], marker="o",
                label=f"learning rate {lr}")
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Learning rate and epoch, chosen on validation", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_rate.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
