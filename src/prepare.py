"""Data-quality checks, fixed lesion-grouped and naive image-level splits, and a resized image cache.

Run once after downloading HAM10000 into data/raw (see README).
"""
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from common import CACHE, IMG_SIZE, RAW, RESULTS, SPLIT_SEED, SPLITS, load_metadata


def quality_checks(df):
    per_lesion = df.groupby("lesion_id").size()
    checks = {
        "n_images": len(df),
        "n_lesions": int(df["lesion_id"].nunique()),
        "lesions_with_2plus_images": int((per_lesion >= 2).sum()),
        "images_in_multi_image_lesions": int(per_lesion[per_lesion >= 2].sum()),
        "max_images_per_lesion": int(per_lesion.max()),
        "lesions_with_inconsistent_dx": int((df.groupby("lesion_id")["dx"].nunique() > 1).sum()),
        "lesions_with_inconsistent_meta": int((df.groupby("lesion_id")[["age", "sex", "localization"]]
                                               .nunique(dropna=False).max(axis=1) > 1).sum()),
        "age_missing": int(df["age_recorded"].isna().sum()),
        "age_zero": int((df["age_recorded"] == 0).sum()),
        "sex_unknown": int((df["sex"] == "unknown").sum()),
        "site_unknown": int((df["localization"] == "unknown").sum()),
        "dx_counts": df["dx"].value_counts().to_dict(),
        "dx_type_counts": df["dx_type"].value_counts().to_dict(),
        "dataset_counts": df["dataset"].value_counts().to_dict(),
        "dx_by_dx_type": pd.crosstab(df["dx_type"], df["dx"]).to_dict(),
    }
    return checks


def make_splits(df):
    les = df.groupby("lesion_id", sort=True)["dx"].first().reset_index()
    tr, rest = train_test_split(les, test_size=0.30, stratify=les["dx"], random_state=SPLIT_SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=rest["dx"], random_state=SPLIT_SEED)
    lmap = {**{l: "train" for l in tr.lesion_id}, **{l: "val" for l in va.lesion_id},
            **{l: "test" for l in te.lesion_id}}
    lesion_split = df[["image_id", "lesion_id"]].assign(split=df["lesion_id"].map(lmap))

    idx = np.arange(len(df))
    tr_i, rest_i = train_test_split(idx, test_size=0.30, stratify=df["dx"], random_state=SPLIT_SEED)
    va_i, te_i = train_test_split(rest_i, test_size=0.50, stratify=df["dx"].iloc[rest_i],
                                  random_state=SPLIT_SEED)
    s = np.empty(len(df), dtype=object)
    s[tr_i], s[va_i], s[te_i] = "train", "val", "test"
    image_split = df[["image_id", "lesion_id"]].assign(split=s)
    return lesion_split, image_split


def leakage_summary(image_split):
    train_lesions = set(image_split.loc[image_split.split == "train", "lesion_id"])
    test = image_split[image_split.split == "test"]
    shared = test["lesion_id"].isin(train_lesions)
    return {"image_split_test_images": len(test),
            "test_images_whose_lesion_is_in_train": int(shared.sum()),
            "fraction": float(shared.mean())}


def build_cache(df):
    out = CACHE / f"images_{IMG_SIZE}.npy"
    if out.exists():
        return
    members = {}
    zips = [zipfile.ZipFile(p) for p in sorted(RAW.glob("HAM10000_images_part_*.zip"))]
    for z in zips:
        for n in z.namelist():
            if n.lower().endswith(".jpg"):
                members[n.split("/")[-1][:-4]] = (z, n)
    missing = set(df.image_id) - set(members)
    assert not missing, f"{len(missing)} images missing from zips"
    raw = [members[i][0].read(members[i][1]) for i in df.image_id]

    def load(b):
        im = Image.open(BytesIO(b)).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)

    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(load, raw)))
    np.save(out, arr)


def main():
    for d in (CACHE, SPLITS, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
    df = load_metadata()
    checks = quality_checks(df)
    lesion_split, image_split = make_splits(df)
    for name, sp in [("lesion", lesion_split), ("image", image_split)]:
        sp.to_csv(SPLITS / f"{name}_split.csv", index=False)
        checks[f"{name}_split_sizes"] = {
            k: {"images": int((sp.split == k).sum()),
                "lesions": int(sp.loc[sp.split == k, "lesion_id"].nunique()),
                "mel_images": int(((sp.split == k) & (df.dx == "mel")).sum())}
            for k in ["train", "val", "test"]}
    checks["image_split_leakage"] = leakage_summary(image_split)
    # lesion split must not share any lesion across parts
    g = lesion_split.groupby("lesion_id")["split"].nunique()
    assert (g == 1).all()
    with open(RESULTS / "data_checks.json", "w") as f:
        json.dump(checks, f, indent=2, default=int)
    print(json.dumps(checks, indent=2, default=int))
    build_cache(df)


if __name__ == "__main__":
    main()
