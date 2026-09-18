"""Fine-tune DistilBERT to read the sentiment of a financial-news sentence.

A plain PyTorch loop rather than a black-box trainer, so every step is visible:

  tokenise    word pieces, nothing cut (the longest sentence is 150 tokens)
  batches     16 sentences, padded only to the longest one in the batch
  optimiser   AdamW, linear warm-up over the first 10% of steps, then decay
  selection   after every epoch, score the VALIDATION set; keep the epoch with
              the best macro-F1

The learning rate is chosen on validation (seed 0 for each candidate), then
the chosen rate is re-run with two more seeds so the result comes with its
spread. Each run is saved to results/runs/ and skipped on the next start, so
an interrupted job resumes where it stopped. The run with the best VALIDATION
score (never test) is saved to models/distilbert/ for predict.py.

    python finetune.py        # about 15 minutes on an Apple M1
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import data

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results" / "runs"
MODEL = "distilbert-base-uncased"
MAX_TOKENS = 160
BATCH = 16
EPOCHS = 4
WARMUP = 0.10
LEARNING_RATES = [2e-5, 3e-5, 5e-5]
SEEDS = [0, 1, 2]


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate(pieces: list[list[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch to its own longest sentence; the mask marks real tokens."""
    width = max(len(p) for p in pieces)
    ids = torch.full((len(pieces), width), pad, dtype=torch.long)
    mask = torch.zeros((len(pieces), width), dtype=torch.long)
    for row, p in enumerate(pieces):
        ids[row, :len(p)] = torch.tensor(p)
        mask[row, :len(p)] = 1
    return ids, mask


@torch.no_grad()
def predict_proba(model, pieces: list[list[int]], pad: int, dev, batch: int = 64) -> np.ndarray:
    model.eval()
    # Longest-first so each batch pads to similar lengths; order restored after.
    order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i]))
    out = np.zeros((len(pieces), len(data.LABELS)), dtype=np.float32)
    for start in range(0, len(order), batch):
        idx = order[start:start + batch]
        ids, mask = collate([pieces[i] for i in idx], pad)
        logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
        out[idx] = torch.softmax(logits.float(), -1).cpu().numpy()
    return out


def macro_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    return f1_score(labels, probs.argmax(1), average="macro")


def pick_best(history: list[dict]) -> int:
    """The epoch with the highest validation macro-F1; the earliest wins a tie."""
    return max(history, key=lambda h: (h["val_macro_f1"], -h["epoch"]))["epoch"]


def train_run(lr: float, seed: int, tok, splits, dev, keep_model: bool = False,
              epochs: int = EPOCHS, limit: int | None = None):
    train, val, test = splits
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    label_id = {name: i for i, name in enumerate(data.LABELS)}
    pieces = tok(list(train["text"]), truncation=True, max_length=MAX_TOKENS)["input_ids"]
    y = torch.tensor(train["label"].map(label_id).values)
    if limit:
        pieces, y = pieces[:limit], y[:limit]
    val_pieces = tok(list(val["text"]), truncation=True, max_length=MAX_TOKENS)["input_ids"]
    val_y = val["label"].map(label_id).values

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(data.LABELS)).to(dev)
    steps = epochs * math.ceil(len(pieces) / BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)

    history, best_state, best_f1 = [], None, -1.0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(pieces))
        losses = []
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            ids, mask = collate([pieces[i] for i in idx], tok.pad_token_id)
            loss = model(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                         labels=y[idx].to(dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            schedule.step()
            opt.zero_grad()
            losses.append(loss.item())
        val_probs = predict_proba(model, val_pieces, tok.pad_token_id, dev)
        f1 = macro_f1(val_y, val_probs)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                        "val_macro_f1": round(f1, 4),
                        "val_accuracy": round(float((val_probs.argmax(1) == val_y).mean()), 4),
                        "seconds": round(time.perf_counter() - started, 1)})
        print(f"  lr {lr:g} seed {seed} epoch {epoch}: loss {history[-1]['train_loss']:.3f} "
              f"val macro-F1 {f1:.4f}  ({history[-1]['seconds']:.0f}s)", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    best = pick_best(history)
    test_pieces = tok(list(test["text"]), truncation=True, max_length=MAX_TOKENS)["input_ids"]
    result = {"lr": lr, "seed": seed, "epochs": epochs, "batch": BATCH, "device": str(dev),
              "history": history, "best_epoch": best,
              "val_macro_f1": next(h["val_macro_f1"] for h in history if h["epoch"] == best),
              "test_fingerprint": data.fingerprint(test),
              "val_ids": val["id"].tolist(),
              "val_probs": np.round(predict_proba(model, val_pieces, tok.pad_token_id, dev), 5).tolist(),
              "test_ids": test["id"].tolist(),
              "test_probs": np.round(predict_proba(model, test_pieces, tok.pad_token_id, dev), 5).tolist(),
              "train_seconds": history[-1]["seconds"]}
    return (result, model) if keep_model else (result, None)


def run_file(lr: float, seed: int) -> Path:
    return RUNS / f"lr{lr:g}_seed{seed}.json"


def main() -> None:
    dev = device()
    df, _ = data.load()
    splits = data.split(df)
    tok = AutoTokenizer.from_pretrained(MODEL)
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"device {dev} · train {len(splits[0])} · val {len(splits[1])} · test {len(splits[2])}")

    def run(lr, seed):
        path = run_file(lr, seed)
        if path.exists():
            return json.loads(path.read_text())
        result, model = train_run(lr, seed, tok, splits, dev, keep_model=True)
        path.write_text(json.dumps(result))
        # Keep the weights of every run on disk until the winner is known;
        # the losers are removed at the end.
        model.save_pretrained(HERE / "models" / f"lr{lr:g}_seed{seed}")
        tok.save_pretrained(HERE / "models" / f"lr{lr:g}_seed{seed}")
        return result

    # 1. Learning rate, chosen on validation with seed 0.
    grid = {lr: run(lr, SEEDS[0]) for lr in LEARNING_RATES}
    chosen = max(LEARNING_RATES, key=lambda lr: (grid[lr]["val_macro_f1"], -lr))
    print(f"chosen learning rate {chosen:g} "
          f"(validation macro-F1 {grid[chosen]['val_macro_f1']:.4f})")
    # 2. The chosen rate with the remaining seeds.
    seeds = {s: run(chosen, s) for s in SEEDS}
    # 3. The run with the best VALIDATION score becomes the saved model.
    keep = max(SEEDS, key=lambda s: (seeds[s]["val_macro_f1"], -s))
    summary = {"learning_rates": {f"{lr:g}": grid[lr]["val_macro_f1"] for lr in LEARNING_RATES},
               "chosen_lr": chosen, "seeds": SEEDS, "saved_seed": keep}
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=1))

    import shutil
    final = HERE / "models" / "distilbert"
    if final.exists():
        shutil.rmtree(final)
    (HERE / "models" / f"lr{chosen:g}_seed{keep}").rename(final)
    for other in (HERE / "models").glob("lr*_seed*"):
        shutil.rmtree(other)
    print(f"saved seed {keep} to {final}")


if __name__ == "__main__":
    if "--time" in sys.argv:
        # One short epoch on 480 sentences, to estimate the full run's length.
        df, _ = data.load()
        splits = data.split(df)
        tok = AutoTokenizer.from_pretrained(MODEL)
        res, _ = train_run(3e-5, 0, tok, splits, device(), epochs=1, limit=480)
        print(f"{res['train_seconds']}s for 30 steps + validation")
    else:
        main()
