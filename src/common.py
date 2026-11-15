"""Shared paths, constants and metric helpers."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
SPLITS = ROOT / "splits"
PREDS = ROOT / "outputs" / "preds"
CKPT = ROOT / "outputs" / "checkpoints"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL = CLASSES.index("mel")
SPLIT_SEED = 42
IMG_SIZE = 224


def load_metadata():
    """Metadata in a fixed order (sorted by image_id); row i matches cache row i."""
    df = pd.read_csv(RAW / "HAM10000_metadata.tab")
    df = df.sort_values("image_id").reset_index(drop=True)
    df["y"] = df["dx"].map(CLASSES.index)
    df["age_recorded"] = df["age"]
    df.loc[df["age"] == 0, "age"] = np.nan  # age 0 is implausible for these lesions; treat as missing
    return df


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, y):
    """Scalar temperature minimising validation NLL (grid search on log T)."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 400))
    nll = [-np.log(softmax(logits, T)[np.arange(len(y)), y] + 1e-12).mean() for T in grid]
    return float(grid[int(np.argmin(nll))])


def lesion_aggregate(probs, lesion_ids, y):
    """Average image probabilities within each lesion."""
    d = pd.DataFrame(probs, columns=CLASSES)
    d["lesion_id"] = np.asarray(lesion_ids)
    d["y"] = np.asarray(y)
    g = d.groupby("lesion_id", sort=True)
    return g[CLASSES].mean().to_numpy(), g["y"].first().to_numpy(), np.array(list(g.groups.keys()))


def threshold_for_sensitivity(p_mel, is_mel, target=0.90):
    """Largest threshold whose sensitivity is at least `target` (fit on validation)."""
    pos = np.sort(p_mel[is_mel == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k])


def ece(p, y, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def binary_metrics(p_mel, is_mel, thr):
    pred = p_mel >= thr
    pos, neg = is_mel == 1, is_mel == 0
    out = {
        "sens": pred[pos].mean() if pos.any() else np.nan,
        "spec": (~pred[neg]).mean() if neg.any() else np.nan,
        "brier": float(np.mean((p_mel - is_mel) ** 2)),
        "ece": ece(p_mel, is_mel),
    }
    if pos.any() and neg.any():
        out["auroc"] = roc_auc_score(is_mel, p_mel)
        out["auprc"] = average_precision_score(is_mel, p_mel)
    else:
        out["auroc"] = out["auprc"] = np.nan
    return out


def multiclass_metrics(probs, y):
    pred = probs.argmax(1)
    return {"bal_acc": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", labels=list(range(len(CLASSES))),
                                 zero_division=0)}


def all_metrics(probs, y, thr):
    is_mel = (y == MEL).astype(int)
    return {**binary_metrics(probs[:, MEL], is_mel, thr), **multiclass_metrics(probs, y)}
