"""Paired leakage check on identical images.

Images in the image-split test set whose lesion lies in the lesion-split validation or test part were
never trained on by the lesion-split models. Both models are scored on these same images, separately
for images whose lesion has a sibling photo in the image-split training set ("seen") and those that do
not ("unseen"). A larger image-split advantage on seen lesions than on unseen ones is leakage.
Seed-averaged probabilities, lesion-level bootstrap for the difference-in-differences.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from common import CLASSES, MEL, PREDS, RESULTS, SPLITS, load_metadata, softmax

rng = np.random.default_rng(0)


def probs(names, ids):
    out = []
    for n in names:
        d = pd.read_csv(PREDS / f"{n}.csv").drop_duplicates("image_id").set_index("image_id")
        out.append(softmax(d.loc[ids, [f"logit_{c}" for c in CLASSES]].values))
    return np.mean(out, 0)


def scores(p, y):
    nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    is_mel = y == MEL
    auc = roc_auc_score(is_mel, p[:, MEL]) if 0 < is_mel.sum() < len(y) else np.nan
    return {"nll": nll, "mel_auroc": auc, "bal_acc": balanced_accuracy_score(y, p.argmax(1))}


def main():
    meta = load_metadata().set_index("image_id")
    sel = json.load(open(RESULTS / "selected_runs.json"))["runs"]
    ls = pd.read_csv(SPLITS / "lesion_split.csv").set_index("image_id")
    isp = pd.read_csv(SPLITS / "image_split.csv").set_index("image_id")
    train_les_img = set(isp.loc[isp.split == "train", "lesion_id"])
    test = isp[(isp.split == "test") & (ls.loc[isp.index, "split"] != "train").values].copy()
    test["seen"] = test.lesion_id.isin(train_les_img)
    ids = test.index.values
    y = meta.loc[ids, "y"].values
    p_les = probs(sel["effnet_b0_lesion"], ids)
    p_img = probs(sel["effnet_b0_image"], ids)
    lids = test.lesion_id.values
    seen = test.seen.values

    def table(idx):
        r = {}
        for tag, m in [("seen", seen[idx]), ("unseen", ~seen[idx])]:
            a, b = scores(p_les[idx][m], y[idx][m]), scores(p_img[idx][m], y[idx][m])
            for k in a:
                r[f"{tag}_{k}_lesion_model"], r[f"{tag}_{k}_image_model"] = a[k], b[k]
                r[f"{tag}_{k}_gap"] = b[k] - a[k] if k != "nll" else a[k] - b[k]
        for k in ["nll", "mel_auroc", "bal_acc"]:
            r[f"did_{k}"] = r[f"seen_{k}_gap"] - r[f"unseen_{k}_gap"]
        return r

    point = table(np.arange(len(ids)))
    groups = pd.Series(np.arange(len(ids))).groupby(lids).apply(np.array).values
    boots = [table(np.concatenate(groups[rng.integers(0, len(groups), len(groups))])) for _ in range(1000)]
    out = {"n_images_seen": int(seen.sum()), "n_images_unseen": int((~seen).sum()),
           "n_mel_images_seen": int((y[seen] == MEL).sum()), "n_mel_images_unseen": int((y[~seen] == MEL).sum())}
    for k, v in point.items():
        b = np.array([bb[k] for bb in boots])
        out[k] = {"est": float(v), "lo": float(np.nanpercentile(b, 2.5)), "hi": float(np.nanpercentile(b, 97.5))}
    with open(RESULTS / "leakage_paired.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
