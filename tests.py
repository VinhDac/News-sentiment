"""Checks for the sentiment models: clean data, no leak between the splits,
choices made on validation only, and reported numbers that match the saved
predictions.

    python tests.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import data
import finetune
from baseline import tfidf_lr

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results" / "runs"
ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'' if cond else ' - ' + detail}")


print("\n[the raw files]")
files = {level: data.read(level) for level in data.LEVELS}
for level, n in {"50Agree": 4846, "66Agree": 4217, "75Agree": 3453, "AllAgree": 2264}.items():
    check(f"{level}: {n} lines, each labelled negative, neutral or positive",
          len(files[level]) == n and files[level]["label"].isin(data.LABELS).all())
for low, high in (("50Agree", "66Agree"), ("66Agree", "75Agree"), ("75Agree", "AllAgree")):
    lower = set(zip(files[low]["text"], files[low]["label"]))
    check(f"every {high} sentence is also in {low}, with the same label",
          all(pair in lower for pair in zip(files[high]["text"], files[high]["label"])))

print("\n[cleaning]")
df, removed = data.load()
raw = files["50Agree"]
per_text = raw.groupby("text")["label"].nunique()
check("no sentence appears twice", df["text"].is_unique)
check("no sentence whose copies carry different labels is kept",
      not df["text"].isin(per_text[per_text > 1].index).any())
check("lines = kept sentences + duplicate copies + conflicting rows",
      removed["lines"] == removed["sentences"] + removed["exact_duplicates_removed"]
      + removed["conflicting_rows_dropped"])
merged = df.merge(raw.drop_duplicates(), on="text", suffixes=("", "_file"))
check("every kept sentence has the label its file gives it",
      len(merged) == len(df) and (merged["label"] == merged["label_file"]).all())
for level, share in data.LEVELS.items():
    inside = df["text"].isin(set(files[level]["text"]))
    check(f"agreement marked {share:.0%} or more exactly for the sentences in {level}",
          ((df["agreement"] >= share) == inside).all())

print("\n[the split - no sentence, nor a copy with other numbers, on two sides]")
train, val, test = data.split(df)
sides = {"train": train, "validation": val, "test": test}
check("the three parts hold every sentence exactly once",
      sorted(pd.concat([train, val, test])["id"]) == sorted(df["id"]))
for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
    check(f"{a} and {b} share no sentence template",
          not set(sides[a]["template"]) & set(sides[b]["template"]))
check("a template joins sentences that differ only in their numbers",
      data.template("Profit rose to EUR 9.4 mn from EUR 11,7 mn")
      == data.template("Profit rose to EUR 12 mn from EUR 3 mn"))
shares = df["label"].value_counts(normalize=True)
for name, part in sides.items():
    gap = (part["label"].value_counts(normalize=True) - shares).abs().max()
    check(f"{name} has the full data's label mix, within 1 point", gap < 0.01, f"{gap:.3f}")
check("the parts are 70 / 15 / 15 within 1 point",
      all(abs(len(p) / len(df) - s) < 0.01 for p, s in ((train, .70), (val, .15), (test, .15))))
check("the split comes out the same every time",
      data.fingerprint(data.split(data.load()[0])[2]) == data.fingerprint(test))

print("\n[the transformer's input and its choices]")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(finetune.MODEL)
longest = max(len(x) for x in tok(list(df["text"]))["input_ids"])
check(f"no sentence is cut: the longest is {longest} tokens, the limit {finetune.MAX_TOKENS}",
      longest <= finetune.MAX_TOKENS)
ids, mask = finetune.collate([[101, 7, 102], [101, 102]], pad=0)
check("a batch is padded to its longest sentence and the padding is masked",
      ids.tolist() == [[101, 7, 102], [101, 102, 0]] and mask.tolist() == [[1, 1, 1], [1, 1, 0]])
history = [{"epoch": e, "val_macro_f1": f} for e, f in ((1, .80), (2, .84), (3, .84), (4, .83))]
check("the kept epoch is the best on validation, the earliest on a tie",
      finetune.pick_best(history) == 2)

print("\n[the baseline learns from the words, not from anything else]")
model = tfidf_lr(1, "balanced").fit(train["text"], train["label"])
f1_real = f1_score(val["label"], model.predict(val["text"]), average="macro")
f1_majority = f1_score(val["label"], [train["label"].mode()[0]] * len(val), average="macro")
check(f"TF-IDF beats always-neutral on validation ({f1_real:.2f} vs {f1_majority:.2f})",
      f1_real > f1_majority + 0.2)
shuffled = tfidf_lr(1, "balanced").fit(train["text"], train["label"].sample(frac=1, random_state=0))
f1_shuffled = f1_score(val["label"], shuffled.predict(val["text"]), average="macro")
check(f"trained on shuffled labels it learns nothing (macro-F1 {f1_shuffled:.2f} < 0.40)",
      f1_shuffled < 0.40)

print("\n[choices were made on validation, never on test]")
label_id = {name: i for i, name in enumerate(data.LABELS)}
y, val_y = test["label"].map(label_id).to_numpy(), val["label"].map(label_id).to_numpy()


def same_split(run) -> bool:
    return sorted(run["val_ids"]) == sorted(val["id"]) and sorted(run["test_ids"]) == sorted(test["id"])


def probs(run, key, part):
    return pd.DataFrame(run[f"{key}_probs"], index=run[f"{key}_ids"]).loc[part["id"]].to_numpy()


summary = json.loads((RUNS / "summary.json").read_text())
grid = {float(lr): json.loads((RUNS / f"lr{float(lr):g}_seed{summary['seeds'][0]}.json").read_text())
        for lr in summary["learning_rates"]}
paths = {s: RUNS / f"lr{summary['chosen_lr']:g}_seed{s}.json" for s in summary["seeds"]}
runs = {s: json.loads(path.read_text()) for s, path in paths.items() if path.exists()}
base = json.loads((HERE / "results" / "baseline.json").read_text())
saved = [base, *grid.values(), *runs.values()]
check("every saved result was made on the validation and test sets the code makes now",
      all(same_split(r) for r in saved))
if all(same_split(r) for r in saved):
    val_f1 = {lr: f1_score(val_y, probs(r, "val", val).argmax(1), average="macro")
              for lr, r in grid.items()}
    check("the chosen learning rate has the best validation macro-F1",
          max(val_f1, key=val_f1.get) == summary["chosen_lr"])
    check("every seed was run with the chosen learning rate", len(runs) == len(paths))
    for r in {(r["lr"], r["seed"]): r for r in [*grid.values(), *runs.values()]}.values():
        kept = finetune.pick_best(r["history"])
        at = next(h["val_macro_f1"] for h in r["history"] if h["epoch"] == kept)
        again = f1_score(val_y, probs(r, "val", val).argmax(1), average="macro")
        check(f"lr {r['lr']:g} seed {r['seed']}: the kept epoch ({kept}) is the validation best, "
              f"and its saved predictions score it", r["best_epoch"] == kept and abs(again - at) < 2e-3)
    seed_f1 = {s: r["val_macro_f1"] for s, r in runs.items()}
    check("the saved model is the seed with the best validation score",
          max(seed_f1, key=lambda s: (seed_f1[s], -s)) == summary["saved_seed"])

print("\n[every reported number comes from the saved predictions]")
metrics = json.loads((HERE / "results" / "metrics.json").read_text())
check("the results were computed on the test set the code makes now",
      metrics["test_fingerprint"] == data.fingerprint(test))
if all(same_split(r) for r in saved) and len(runs) == len(paths):
    p = probs(base, "test", test).argmax(1)
    check("TF-IDF: accuracy and macro-F1 match its saved predictions",
          abs((p == y).mean() - metrics["tfidf_lr"]["accuracy"]) < 1e-12
          and abs(f1_score(y, p, average="macro") - metrics["tfidf_lr"]["macro_f1"]) < 1e-12)
    preds = [probs(r, "test", test).argmax(1) for r in runs.values()]
    acc = [(q == y).mean() for q in preds]
    f1 = [f1_score(y, q, average="macro") for q in preds]
    d = metrics["distilbert"]
    check("DistilBERT: mean accuracy and macro-F1 over the seeds match the saved predictions",
          abs(np.mean(acc) - d["mean"]["accuracy"]) < 1e-12
          and abs(np.mean(f1) - d["mean"]["macro_f1"]) < 1e-12)
    check("DistilBERT: the ± is the standard deviation across seeds",
          abs(np.std(f1, ddof=1) - d["std"]["macro_f1"]) < 1e-12)
    all_agree = test["agreement"].to_numpy() == 1
    check("DistilBERT: the all-annotators-agreed accuracy matches too",
          abs(np.mean([(q == y)[all_agree].mean() for q in preds])
              - d["mean"]["accuracy_agree_100"]) < 1e-12)
    split_share = np.mean([(test["agreement"].to_numpy()[q != y] < 1).mean() for q in preds])
    check("DistilBERT: the share of mistakes on sentences the annotators split over matches",
          abs(split_share - d["errors_where_annotators_split"]["mean_share"]) < 1e-12)
    cut = metrics["selective"]["distilbert"]
    vp = probs(runs[summary["saved_seed"]], "val", val)
    kept = vp.max(1) >= cut["threshold"]
    check(f"the confidence cut-off gives at least {metrics['selective']['target']:.0%} "
          f"accuracy on validation", (vp.argmax(1)[kept] == val_y[kept]).mean()
          >= metrics["selective"]["target"])
    # ... and the rest of the table, the chart and the text.
    import evaluate
    from sklearn.metrics import confusion_matrix
    agree = test["agreement"].to_numpy()
    check("DistilBERT: each class's F1 matches (mean of the seeds)",
          np.allclose(np.mean([f1_score(y, q, average=None) for q in preds], axis=0),
                      [d["mean"][f"f1_{c}"] for c in data.LABELS]))
    check("DistilBERT: accuracy at each level of annotator agreement matches",
          all(np.isclose(np.mean([(q == y)[agree == lvl].mean() for q in preds]),
                         d["mean"][f"accuracy_agree_{name}"]) for lvl, name in evaluate.BUCKETS.items()))
    t_ = metrics["tfidf_lr"]
    check("TF-IDF: each class's F1 and each agreement level's accuracy match",
          np.allclose(f1_score(y, p, average=None), [t_[f"f1_{c}"] for c in data.LABELS])
          and all(np.isclose((p == y)[agree == lvl].mean(), t_[f"accuracy_agree_{name}"])
                  for lvl, name in evaluate.BUCKETS.items()))
    majority = np.full(len(y), label_id[base["majority_label"]])
    check("always-neutral: accuracy at each agreement level matches",
          all(np.isclose((majority == y)[agree == lvl].mean(), metrics["majority"][f"accuracy_agree_{name}"])
              for lvl, name in evaluate.BUCKETS.items()))
    kept_test = probs(runs[summary["saved_seed"]], "test", test).argmax(1)
    check("the saved model's confusion matrix matches (and so the 10 of 725 and 72 of 98)",
          (confusion_matrix(y, kept_test) == np.array(metrics["confusion"])).all())
    for key, run_ in (("distilbert", runs[summary["saved_seed"]]), ("tfidf_lr", base)):
        again = evaluate.selective(probs(run_, "val", val), val_y, probs(run_, "test", test), y)
        check(f"{key}: the confidence cut-off, re-derived from validation, and its test coverage and "
              f"accuracy match", all(np.isclose(again[k], metrics["selective"][key][k]) for k in again))
else:
    print("  (skipped: the saved results do not match the split or the chosen runs)")
best = max(base["grid"], key=lambda g: g["val_macro_f1"])
check("TF-IDF: the chosen strength and class weights have the best validation macro-F1",
      best["C"] == base["chosen"]["C"] and best["class_weight"] == base["chosen"]["class_weight"])

print("\n[the saved model on sentences written for this test]")
if (HERE / "models" / "distilbert").exists():
    import predict
    cases = [("Operating profit rose to EUR 12.3 mn from EUR 8.1 mn a year earlier.", "positive"),
             ("The company won a EUR 20 mn order from a German car maker.", "positive"),
             ("The company swung to a net loss and will cut 300 jobs.", "negative"),
             ("Net sales fell 15 % to EUR 40 mn as demand weakened.", "negative"),
             ("The annual general meeting will be held in Helsinki on 5 April.", "neutral"),
             ("The company's head office is located in Espoo, Finland.", "neutral")]
    for (text, want), r in zip(cases, predict.classify([c[0] for c in cases], *predict.load())):
        check(f"{want}: {text}", r["distilbert"] == want, f"said {r['distilbert']}")
else:
    print("  (no saved model - run finetune.py first)")

print(f"\n{ok} ok, {fail} fail")
raise SystemExit(1 if fail else 0)
