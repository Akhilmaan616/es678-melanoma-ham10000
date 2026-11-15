"""Grad-CAM for missed test melanomas (EfficientNet-B0, seed 0 checkpoint, melanoma logit).

Writes a full contact sheet of every missed melanoma for manual inspection and a compact
figure for the report.
"""
import json

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CACHE, CKPT, FIGS, IMG_SIZE, MEL, RESULTS, load_metadata
from train_cnn import build_model, to_input


def gradcam(model, x):
    acts, grads = {}, {}
    layer = model.features[-1]
    def fwd(m, i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.setdefault("g", g))

    h = layer.register_forward_hook(fwd)
    model.zero_grad()
    out = model(x)
    out[:, MEL].sum().backward()
    h.remove()
    a, g = acts["a"], grads["g"]
    cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.detach().cpu().numpy(), torch.softmax(out, 1)[:, MEL].detach().cpu().numpy()


def main():
    meta = load_metadata()
    run = json.load(open(RESULTS / "selected_runs.json"))["runs"]["effnet_b0_lesion"][0]
    model = build_model("effnet_b0").cuda().eval()
    model.load_state_dict(torch.load(CKPT / f"{run}.pt", map_location="cuda"))
    X = np.load(CACHE / f"images_{IMG_SIZE}.npy", mmap_mode="r")
    missed = pd.read_csv(RESULTS / "missed_melanomas.csv")
    first_row = meta.reset_index().groupby("lesion_id")["index"].first()
    rows = []
    for _, m in missed.iterrows():
        i = int(first_row[m.lesion_id])
        x = to_input(torch.from_numpy(np.array(X[i:i + 1])), "cuda").requires_grad_(True)
        cam, p = gradcam(model, x)
        rows.append((m, np.array(X[i]), cam[0], float(p[0]), meta.image_id[i]))

    n = len(rows)
    cols = 6
    r_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(2 * r_, cols, figsize=(cols * 2, 4 * r_))
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
    for k, (m, img, cam, p, iid) in enumerate(rows):
        r, c = 2 * (k // cols), k % cols
        axes[r, c].imshow(img)
        axes[r, c].set_title(f"{k}: {iid}\n{m.localization}, {m.dx_type}, p={m.p_mel:.2f}", fontsize=6)
        axes[r + 1, c].imshow(img)
        axes[r + 1, c].imshow(cam, cmap="jet", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "gradcam_all_missed.png", dpi=110)

    # chosen by hand to show varied cases: nevus-like, dense hair, vignette, ruler marks
    pick = [rows[i] for i in (0, 6, 10, 13) if i < len(rows)]
    fig, axes = plt.subplots(1, 2 * len(pick), figsize=(7.2, 1.15))
    for k, (m, img, cam, p, iid) in enumerate(pick):
        axes[2 * k].imshow(img)
        axes[2 * k].set_title(f"p={m.p_mel:.2f}", fontsize=7)
        axes[2 * k + 1].imshow(img)
        axes[2 * k + 1].imshow(cam, cmap="jet", alpha=0.4)
    for a in axes:
        a.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.84, wspace=0.04)
    fig.savefig(FIGS / "gradcam_missed.pdf")
    fig.savefig(FIGS / "gradcam_missed.png", dpi=200)
    pd.DataFrame([{"k": k, "image_id": iid, "lesion_id": m.lesion_id, "p_mel_seedavg": m.p_mel,
                   "p_mel_seed0_image": p} for k, (m, _, _, p, iid) in enumerate(rows)]
                 ).to_csv(RESULTS / "gradcam_index.csv", index=False)


if __name__ == "__main__":
    main()
