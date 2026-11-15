"""Test-set evaluation: headline metrics, leakage gap, calibration and subgroup error audit.

All thresholds and temperatures are fit on validation predictions. Confidence intervals come from
a lesion-level bootstrap of the test set; for multi-seed models each bootstrap replicate averages
the metric over seeds, so the interval is for the seed-averaged metric.
"""
import json
import warnings
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (CLASSES, FIGS, MEL, PREDS, RESULTS, all_metrics, fit_temperature,
                    lesion_aggregate, load_metadata, softmax, threshold_for_sensitivity)

B = int(os.environ.get("BOOT", 1000))
METRICS = ["auroc", "auprc", "sens", "spec", "bal_acc", "macro_f1", "brier", "ece"]
rng = np.random.default_rng(0)
warnings.filterwarnings("ignore", category=UserWarning)


class Run:
    """One trained model's validation/test predictions, at image and lesion level."""

    def __init__(self, name, meta):
        d = pd.read_csv(PREDS / f"{name}.csv")
        ycol = meta.set_index("image_id").loc[d.image_id, "y"].values
        lg = d[[f"logit_{c}" for c in CLASSES]].values
        self.name = name
        self.T = fit_temperature(lg[d.part.values == "val"], ycol[d.part.values == "val"])
        self.levels = {}
        for cal, T in [("raw", 1.0), ("ts", self.T)]:
            for part in ["val", "test"]:
                m = d.part.values == part
                p, y, ids = softmax(lg[m], T), ycol[m], d.image_id.values[m]
                self.levels[(cal, part, "image")] = (p, y, ids)
                pl, yl, lids = lesion_aggregate(p, d.lesion_id.values[m], y)
                self.levels[(cal, part, "lesion")] = (pl, yl, lids)
        self.thr = {(cal, lev): threshold_for_sensitivity(self.levels[(cal, "val", lev)][0][:, MEL],
                                                          (self.levels[(cal, "val", lev)][1] == MEL).astype(int))
                    for cal in ["raw", "ts"] for lev in ["image", "lesion"]}
        if name.startswith("trivial"):  # always predicts the majority class (nevus): never flags melanoma
            self.thr = {k: np.inf for k in self.thr}
        test_img = d[d.part == "test"]
        self.img_lesion = test_img.lesion_id.values

    def test(self, cal, level):
        return self.levels[(cal, "test", level)]


def boot_indices(runs, level, subset=None):
    """Bootstrap replicate index arrays (resampling lesions) for the test set of `runs`."""
    r0 = runs[0]
    _, _, lids = r0.test("raw", "lesion")
    if subset is not None:
        lids = lids[subset]
    if level == "lesion":
        base = np.arange(len(r0.test("raw", "lesion")[2]))
        base = base if subset is None else base[subset]
        return [base[rng.integers(0, len(base), len(base))] for _ in range(B)]
    groups = pd.Series(np.arange(len(r0.img_lesion))).groupby(r0.img_lesion).apply(np.array)
    g = groups.loc[lids].values
    return [np.concatenate(g[rng.integers(0, len(g), len(g))]) for _ in range(B)]


def evaluate(runs, level, cal="raw", subset=None, metrics=METRICS):
    """Seed-averaged point estimate, seed SD and 95% bootstrap CI for each metric."""
    def score(idx):
        vals = []
        for r in runs:
            p, y, _ = r.test(cal, level)
            if idx is not None:
                p, y = p[idx], y[idx]
            vals.append(all_metrics(p, y, r.thr[(cal, level)]))
        return {k: [v[k] for v in vals] for k in metrics}

    if subset is not None and level == "lesion":
        full_idx = np.arange(len(runs[0].test(cal, "lesion")[1]))[subset]
    elif subset is not None:
        _, _, lids = runs[0].test(cal, "lesion")
        full_idx = np.where(np.isin(runs[0].img_lesion, lids[subset]))[0]
    else:
        full_idx = None
    point = score(full_idx)
    boots = [score(i) for i in boot_indices(runs, level, subset)]
    out = {}
    for k in metrics:
        bk = np.array([np.nanmean(b[k]) for b in boots])
        out[k] = {"mean": float(np.nanmean(point[k])), "seed_sd": float(np.nanstd(point[k], ddof=1)) if len(runs) > 1 else 0.0,
                  "lo": float(np.nanpercentile(bk, 2.5)), "hi": float(np.nanpercentile(bk, 97.5))}
    return out


def flat(model, res, **extra):
    return [{"model": model, **extra, "metric": k, **v} for k, v in res.items()]


def reliability(ax, runs, cal, label, color):
    p = np.concatenate([r.test(cal, "lesion")[0][:, MEL] for r in runs])
    y = np.concatenate([(r.test(cal, "lesion")[1] == MEL).astype(int) for r in runs])
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= 5:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    ax.plot(xs, ys, "o-", color=color, label=label, ms=4)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    sel = json.load(open(RESULTS / "selected_runs.json"))["runs"]
    groups = {
        "Majority class": ["trivial_lesion"],
        "Metadata-only LR": ["metadata_lr_lesion"],
        "ResNet-50 linear probe": ["linear_probe_lesion"],
        "ResNet-50 fine-tuned": sel["resnet50_lesion"],
        "EfficientNet-B0 fine-tuned": sel["effnet_b0_lesion"],
    }
    leak_groups = {
        "ResNet-50 linear probe": (["linear_probe_lesion"], ["linear_probe_image"]),
        "EfficientNet-B0 fine-tuned": (sel["effnet_b0_lesion"], sel["effnet_b0_image"]),
    }
    runs = {n: [Run(r, meta) for r in rs] for n, rs in groups.items()}
    summary = {"temperatures": {n: [r.T for r in rs] for n, rs in runs.items()},
               "val_thresholds": {n: [r.thr[("raw", "lesion")] for r in rs] for n, rs in runs.items()}}

    # 1. headline comparison (lesion-grouped split), image and lesion level, before/after TS
    rows = []
    for n, rs in runs.items():
        for level in ["lesion", "image"]:
            for cal in ["raw", "ts"]:
                rows += flat(n, evaluate(rs, level, cal), level=level, cal=cal)
        print("done", n, flush=True)
    main_df = pd.DataFrame(rows)
    main_df.to_csv(RESULTS / "main_results.csv", index=False)

    # 2. leakage: lesion-grouped vs naive image-level split (image-level metrics, same recipe)
    rows = []
    for n, (lr_, ir_) in leak_groups.items():
        for split, rs in [("lesion", lr_), ("image", ir_)]:
            rr = [Run(r, meta) for r in rs]
            rows += flat(n, evaluate(rr, "image"), split=split, subset="all")
            if split == "image":
                # within the image-split test set: images whose lesion also has a photo in training
                sp = pd.read_csv(RESULTS.parent / "splits" / "image_split.csv")
                train_les = set(sp.loc[sp.split == "train", "lesion_id"])
                lids = rr[0].test("raw", "lesion")[2]
                seen = np.isin(lids, list(train_les))
                for tag, sub in [("lesion seen in train", seen), ("lesion unseen", ~seen)]:
                    rows += flat(n, evaluate(rr, "image", subset=sub), split=split, subset=tag,
                                 n_lesions=int(sub.sum()))
    pd.DataFrame(rows).to_csv(RESULTS / "leakage_results.csv", index=False)
    print("done leakage", flush=True)

    # 3. per-class recall of the main model (7-class task)
    main_runs = runs["EfficientNet-B0 fine-tuned"]
    rec = []
    for r in main_runs:
        p, y, _ = r.test("raw", "lesion")
        pred = p.argmax(1)
        rec.append([(pred[y == k] == k).mean() for k in range(len(CLASSES))])
    summary["per_class_recall_lesion"] = dict(zip(CLASSES, np.mean(rec, 0).round(3).tolist()))
    summary["n_test_lesions_per_class"] = dict(zip(CLASSES, np.bincount(main_runs[0].test("raw", "lesion")[1],
                                                                        minlength=7).tolist()))

    # 4. subgroup error and calibration audit (main model, lesion level)
    lmeta = meta.groupby("lesion_id").first().loc[main_runs[0].test("raw", "lesion")[2]]
    age = lmeta["age"]
    bands = pd.cut(age, [-1, 29, 44, 59, 74, 200], labels=["<30", "30-44", "45-59", "60-74", "75+"])
    loc = lmeta["localization"]
    top = loc.value_counts()
    loc = loc.where(loc.isin(top[top >= 40].index), "other sites")
    facets = {"Confirmation": lmeta["dx_type"], "Body site": loc,
              "Age band": bands.astype(str).replace("nan", "unknown"), "Sex": lmeta["sex"],
              "Source": lmeta["dataset"]}
    rows = []
    sub_metrics = ["sens", "spec", "auroc", "ece"]
    for facet, col in facets.items():
        for g in sorted(col.unique()):
            sub = (col == g).values
            y = main_runs[0].test("raw", "lesion")[1][sub]
            n_mel, n_other = int((y == MEL).sum()), int((y != MEL).sum())
            res = {}
            for cal in ["raw", "ts"]:
                ev = evaluate(main_runs, "lesion", cal, subset=sub, metrics=sub_metrics)
                for k, v in ev.items():
                    res[f"{k}_{cal}"] = v
            pm = np.mean([r.test("raw", "lesion")[0][sub, MEL].mean() for r in main_runs])
            pm_ts = np.mean([r.test("ts", "lesion")[0][sub, MEL].mean() for r in main_runs])
            rows.append({"facet": facet, "group": g, "n_lesions": int(sub.sum()), "n_mel": n_mel,
                         "n_other": n_other, "small": n_mel < 15 or n_other < 15,
                         "mean_p_mel_raw": pm, "mean_p_mel_ts": pm_ts, "observed_mel_rate": n_mel / sub.sum(),
                         **{f"{k}_{s}": v[s] for k, v in res.items() for s in ["mean", "lo", "hi"]}})
        print("done facet", facet, flush=True)
    sg = pd.DataFrame(rows)
    sg.to_csv(RESULTS / "subgroup_audit.csv", index=False)

    # missed melanomas (seed-averaged lesion probability below the averaged threshold), for Grad-CAM
    p_avg = np.mean([r.test("raw", "lesion")[0][:, MEL] for r in main_runs], 0)
    thr_avg = float(np.mean([r.thr[("raw", "lesion")] for r in main_runs]))
    yl, lids = main_runs[0].test("raw", "lesion")[1], main_runs[0].test("raw", "lesion")[2]
    missed = pd.DataFrame({"lesion_id": lids, "p_mel": p_avg})[(yl == MEL) & (p_avg < thr_avg)]
    missed = missed.join(lmeta[["dx_type", "localization", "age", "sex", "dataset"]], on="lesion_id")
    missed.sort_values("p_mel").to_csv(RESULTS / "missed_melanomas.csv", index=False)
    summary["missed_melanomas"] = {"n": len(missed), "n_test_mel_lesions": int((yl == MEL).sum()),
                                   "threshold_avg": thr_avg}

    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    make_figure(main_runs, runs["ResNet-50 linear probe"], sg)
    print("done")


def make_figure(main_runs, probe_runs, sg):
    """Reliability diagrams and subgroup sensitivity/specificity."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.25))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
    reliability(ax, main_runs, "raw", "EffNet-B0, before TS", "#c0392b")
    reliability(ax, main_runs, "ts", "EffNet-B0, after TS", "#2471a3")
    reliability(ax, probe_runs, "raw", "Linear probe", "#7f8c8d")
    ax.set_xlabel("Predicted melanoma probability")
    ax.set_ylabel("Observed melanoma rate")
    ax.set_title("(a) Reliability, test lesions", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    ax = axes[1]
    show = sg[(sg.n_mel >= 5)].reset_index(drop=True)
    ypos = np.arange(len(show))[::-1]
    ax.errorbar(show.sens_raw_mean, ypos + 0.15, xerr=[show.sens_raw_mean - show.sens_raw_lo,
                show.sens_raw_hi - show.sens_raw_mean], fmt="o", ms=3, color="#c0392b", lw=1, label="Sensitivity")
    ax.errorbar(show.spec_raw_mean, ypos - 0.15, xerr=[show.spec_raw_mean - show.spec_raw_lo,
                show.spec_raw_hi - show.spec_raw_mean], fmt="s", ms=3, color="#2471a3", lw=1, label="Specificity")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{g} ({m})" for g, m in zip(show.group, show.n_mel)], fontsize=6)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Lesion-level rate (95% CI)")
    ax.set_title("(b) Subgroups with $\\geq$5 test melanomas", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    for a in axes:
        a.tick_params(labelsize=7)
        a.xaxis.label.set_size(8)
        a.yaxis.label.set_size(8)
    fig.tight_layout()
    fig.savefig(FIGS / "calibration_subgroups.pdf")
    fig.savefig(FIGS / "calibration_subgroups.png", dpi=200)


if __name__ == "__main__":
    if "--figures-only" in sys.argv:  # redraw from saved predictions and subgroup table, no bootstrap
        meta = load_metadata()
        sel = json.load(open(RESULTS / "selected_runs.json"))["runs"]
        make_figure([Run(r, meta) for r in sel["effnet_b0_lesion"]], [Run("linear_probe_lesion", meta)],
                    pd.read_csv(RESULTS / "subgroup_audit.csv"))
    else:
        main()
