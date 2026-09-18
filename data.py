"""Load Financial PhraseBank, remove duplicates, and split it.

ONE place, imported by training, evaluation and the tests, so the test set is
defined once and can never drift between them.
"""
import hashlib
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path(__file__).resolve().parent / "data" / "raw" / "FinancialPhraseBank-v1.0"
LABELS = ["negative", "neutral", "positive"]
# Each file is a subset of the one before: at least 50% of annotators agreed,
# then 66%, 75%, and finally all of them.
LEVELS = {"50Agree": 0.50, "66Agree": 0.66, "75Agree": 0.75, "AllAgree": 1.00}
SEED = 42


def read(level: str) -> pd.DataFrame:
    rows = []
    for raw in (RAW / f"Sentences_{level}.txt").read_bytes().splitlines():
        # The files are Latin-1, but a line that is valid UTF-8 is read as UTF-8,
        # so accented company names come out right either way.
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            line = raw.decode("latin-1")
        text, _, label = line.strip().rpartition("@")
        rows.append((" ".join(text.split()), label.strip()))
    return pd.DataFrame(rows, columns=["text", "label"])


def template(text: str) -> str:
    """The sentence with every number masked. "Profit rose to EUR 9.4 mn" and
    "Profit rose to EUR 11.7 mn" are the same sentence for a model: one in
    training and the other in the test set would be a leak."""
    return " ".join(re.sub(r"\d+([.,]\d+)*", "#", text.lower()).split())


def sentence_id(text: str) -> str:
    """A short fingerprint, so saved predictions can be matched back to their
    sentence without storing the dataset's text in the repository."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load() -> tuple[pd.DataFrame, dict]:
    """Every distinct sentence once, with its label and the strongest level of
    annotator agreement it reached. Returns the table and what was removed."""
    df = read("50Agree")
    n_raw = len(df)
    # The same sentence twice would let a test sentence be learnt in training.
    # Keep one copy when the labels agree; drop every copy when they conflict.
    labels_per_text = df.groupby("text")["label"].nunique()
    conflicting = set(labels_per_text[labels_per_text > 1].index)
    n_conflict_rows = int(df["text"].isin(conflicting).sum())
    df = df[~df["text"].isin(conflicting)].drop_duplicates("text").reset_index(drop=True)
    df["agreement"] = 0.50
    for level, share in LEVELS.items():
        df.loc[df["text"].isin(set(read(level)["text"])), "agreement"] = share
    df["template"] = df["text"].map(template)
    df["id"] = df["text"].map(sentence_id)
    removed = {"lines": n_raw,
               "exact_duplicates_removed": n_raw - n_conflict_rows - len(df),
               "conflicting_sentences_dropped": len(conflicting),
               "conflicting_rows_dropped": n_conflict_rows,
               "sentences": len(df),
               "templates": int(df["template"].nunique())}
    return df, removed


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70% train, 15% validation, 15% test, each with the same mix of labels.

    Split by TEMPLATE, not by sentence: sentences that differ only in their
    numbers all land on the same side."""
    one_per_template = df.drop_duplicates("template")
    train, rest = train_test_split(one_per_template, test_size=0.30,
                                   stratify=one_per_template["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"],
                                 random_state=SEED)

    def members(part: pd.DataFrame) -> pd.DataFrame:
        return df[df["template"].isin(set(part["template"]))].reset_index(drop=True)

    return members(train), members(val), members(test)


def fingerprint(part: pd.DataFrame) -> str:
    """One hash for a whole split: the saved results name the test set they
    were computed on, and the tests check it is still the one the code makes."""
    return hashlib.sha1("\n".join(sorted(part["id"])).encode()).hexdigest()[:16]
