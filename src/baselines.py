"""Reference models: majority class, metadata-only logistic regression, and a ResNet-50 linear probe.

Every learned step (imputation, encoding, scaling, regularisation strength) is fit on the
training split, and C is chosen by lesion-level validation melanoma AUROC.
"""
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import CACHE, CLASSES, IMG_SIZE, MEL, PREDS, SPLITS, lesion_aggregate, load_metadata, softmax


def save(name, df, parts, logits):
    rows = []
    for part, idx in parts:
        d = pd.DataFrame(logits[idx], columns=[f"logit_{c}" for c in CLASSES])
        d.insert(0, "part", part)
        d.insert(0, "lesion_id", df.lesion_id.values[idx])
        d.insert(0, "image_id", df.image_id.values[idx])
        rows.append(d)
    pd.concat(rows).to_csv(PREDS / f"{name}.csv", index=False)


def val_auc(logits, df, va):
    pl, yl, _ = lesion_aggregate(softmax(logits[va]), df.lesion_id.values[va], df.y.values[va])
    return roc_auc_score(yl == MEL, pl[:, MEL])


def metadata_features(df, tr):
    age = df["age"].astype(float)
    age = age.fillna(age.iloc[tr].median())
    age = (age - age.iloc[tr].mean()) / age.iloc[tr].std()
    cats = pd.get_dummies(df[["sex", "localization"]].astype(str))
    cats = cats.loc[:, cats.iloc[tr].sum() > 0]  # levels seen in training only
    return np.column_stack([age.values, cats.values.astype(float)])


def fit_lr(F, y, tr, va, df, grid):
    best = None
    for C in grid:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        m.fit(F[tr], y[tr])
        auc = val_auc(m.decision_function(F), df, va)
        print(f"  C={C:g} val lesion AUROC={auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, C, m)
    print(f"  chosen C={best[1]:g}")
    return best[2].decision_function(F), best[1]


@torch.no_grad()
def resnet50_features():
    out = CACHE / "resnet50_feats.npy"
    if out.exists():
        return np.load(out)
    m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    m.fc = torch.nn.Identity()
    m = m.cuda().eval()
    X = np.load(CACHE / f"images_{IMG_SIZE}.npy")
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    feats = []
    for i in range(0, len(X), 128):
        xb = torch.from_numpy(X[i:i + 128]).cuda().permute(0, 3, 1, 2).float() / 255.0
        with torch.autocast("cuda", dtype=torch.float16):
            feats.append(m((xb - mean) / std).float().cpu().numpy())
    F = np.concatenate(feats)
    np.save(out, F)
    return F


def main():
    PREDS.mkdir(parents=True, exist_ok=True)
    df = load_metadata()
    y = df.y.values
    feats = resnet50_features()
    for split in ["lesion", "image"]:
        sp = pd.read_csv(SPLITS / f"{split}_split.csv").set_index("image_id").loc[df.image_id]
        tr, va, te = (np.where(sp.split.values == p)[0] for p in ["train", "val", "test"])
        parts = [("val", va), ("test", te)]

        prior = np.bincount(y[tr], minlength=len(CLASSES)) / len(tr)
        save(f"trivial_{split}", df, parts, np.tile(np.log(prior), (len(df), 1)))

        print(f"[{split}] metadata-only LR")
        lg, _ = fit_lr(metadata_features(df, tr), y, tr, va, df, [0.001, 0.01, 0.1, 1, 10])
        save(f"metadata_lr_{split}", df, parts, lg)

        print(f"[{split}] ResNet-50 linear probe")
        sc = StandardScaler().fit(feats[tr])
        lg, _ = fit_lr(sc.transform(feats), y, tr, va, df, [0.0003, 0.001, 0.003, 0.01, 0.03])
        save(f"linear_probe_{split}", df, parts, lg)


if __name__ == "__main__":
    main()
