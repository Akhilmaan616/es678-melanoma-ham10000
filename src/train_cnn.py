"""Fine-tune an ImageNet-pretrained CNN on HAM10000 (7 classes, class-weighted cross-entropy).

The checkpoint is chosen by lesion-level validation melanoma AUROC. Validation and test
logits of the chosen checkpoint are written to outputs/preds; the test set is scored only once.

Example:
    python src/train_cnn.py --arch effnet_b0 --split lesion --lr 3e-4 --seed 0
"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import roc_auc_score

from common import CACHE, CKPT, CLASSES, IMG_SIZE, MEL, PREDS, SPLITS, lesion_aggregate, load_metadata, softmax

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_model(arch):
    if arch == "effnet_b0":
        m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    elif arch == "resnet50":
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    else:
        raise ValueError(arch)
    return m


def to_input(x_uint8, device):
    x = x_uint8.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN.to(device)) / STD.to(device)


def augment(x, gen):
    """Per-sample flips, 90-degree rotations and mild colour jitter on an unnormalised [0,1] batch."""
    n = x.shape[0]
    dev = x.device
    flip_h = torch.rand(n, device=dev, generator=gen) < 0.5
    flip_v = torch.rand(n, device=dev, generator=gen) < 0.5
    x = torch.where(flip_h.view(-1, 1, 1, 1), x.flip(3), x)
    x = torch.where(flip_v.view(-1, 1, 1, 1), x.flip(2), x)
    k = torch.randint(0, 4, (n,), device=dev, generator=gen)
    x = torch.stack([torch.rot90(xi, int(ki), dims=(1, 2)) for xi, ki in zip(x, k)])
    u = lambda: (0.9 + 0.2 * torch.rand(n, 1, 1, 1, device=dev, generator=gen))
    x = x * u()  # brightness
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * u() + m  # contrast
    g = x.mean(dim=1, keepdim=True)
    x = (x - g) * u() + g  # saturation
    return x.clamp(0, 1)


@torch.no_grad()
def predict(model, X, idx, device, bs=128):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        xb = to_input(X[idx[i:i + bs]], device)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(xb).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="effnet_b0")
    ap.add_argument("--split", default="lesion", choices=["lesion", "image"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save_ckpt", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda"
    df = load_metadata()
    sp = pd.read_csv(SPLITS / f"{a.split}_split.csv").set_index("image_id").loc[df.image_id]
    tr = np.where(sp.split.values == "train")[0]
    va = np.where(sp.split.values == "val")[0]
    te = np.where(sp.split.values == "test")[0]
    y = df.y.values
    X = torch.from_numpy(np.load(CACHE / f"images_{IMG_SIZE}.npy"))  # uint8 N,H,W,3 on CPU

    counts = np.bincount(y[tr], minlength=len(CLASSES))
    w = torch.tensor(len(tr) / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    model = build_model(a.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(tr) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    lossf = nn.CrossEntropyLoss(weight=w)
    gen = torch.Generator(device=device).manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    y_t = torch.from_numpy(y)

    name = a.tag or f"{a.arch}_{a.split}_lr{a.lr:g}_s{a.seed}"
    best, best_state, log = -1, None, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        perm = rng.permutation(tr)
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            xb = X[b].to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
            xb = augment(xb, gen)
            xb = (xb - MEAN.to(device)) / STD.to(device)
            yb = y_t[b].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(b)
        lv = predict(model, X, va, device)
        pl, yl, _ = lesion_aggregate(softmax(lv), df.lesion_id.values[va], y[va])
        auc = roc_auc_score(yl == MEL, pl[:, MEL])
        log.append({"epoch": ep, "train_loss": tot / len(tr), "val_lesion_mel_auroc": auc,
                    "sec": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        if auc > best:
            best, best_state = auc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    PREDS.mkdir(parents=True, exist_ok=True)
    rows = []
    for part, idx in [("val", va), ("test", te)]:
        lg = predict(model, X, idx, device)
        d = pd.DataFrame(lg, columns=[f"logit_{c}" for c in CLASSES])
        d.insert(0, "part", part)
        d.insert(0, "lesion_id", df.lesion_id.values[idx])
        d.insert(0, "image_id", df.image_id.values[idx])
        rows.append(d)
    pd.concat(rows).to_csv(PREDS / f"{name}.csv", index=False)
    with open(PREDS / f"{name}_log.json", "w") as f:
        json.dump({"args": vars(a), "best_val_lesion_mel_auroc": best, "log": log}, f, indent=2)
    if a.save_ckpt:
        CKPT.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, CKPT / f"{name}.pt")
    print("best val lesion mel AUROC", best)


if __name__ == "__main__":
    main()
