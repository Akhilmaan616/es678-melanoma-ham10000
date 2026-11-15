"""Run the full CNN experiment grid.

1. Learning-rate sweep (seed 0) on the lesion-grouped split, chosen by validation lesion-level
   melanoma AUROC.
2. Two more seeds at the chosen learning rate.
3. The same EfficientNet-B0 recipe (three seeds) on the naive image-level split, for the
   leakage analysis only.
"""
import json
import subprocess
import sys

from common import PREDS, RESULTS

SWEEP = {"effnet_b0": [1e-4, 3e-4, 1e-3], "resnet50": [1e-4, 3e-4]}
SEEDS = [0, 1, 2]


def run(arch, split, lr, seed):
    name = f"{arch}_{split}_lr{lr:g}_s{seed}"
    if not (PREDS / f"{name}.csv").exists():
        cmd = [sys.executable, "src/train_cnn.py", "--arch", arch, "--split", split,
               "--lr", str(lr), "--seed", str(seed), "--save_ckpt"]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    return name


def best_val(name):
    with open(PREDS / f"{name}_log.json") as f:
        return json.load(f)["best_val_lesion_mel_auroc"]


def main():
    selected = {"sweep": {}, "runs": {}}
    for arch, lrs in SWEEP.items():
        scores = {lr: best_val(run(arch, "lesion", lr, 0)) for lr in lrs}
        lr = max(scores, key=scores.get)
        selected["sweep"][arch] = {"val_auroc_by_lr": {str(k): v for k, v in scores.items()}, "chosen_lr": lr}
        selected["runs"][f"{arch}_lesion"] = [run(arch, "lesion", lr, s) for s in SEEDS]
    lr = selected["sweep"]["effnet_b0"]["chosen_lr"]
    selected["runs"]["effnet_b0_image"] = [run("effnet_b0", "image", lr, s) for s in SEEDS]
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "selected_runs.json", "w") as f:
        json.dump(selected, f, indent=2)


if __name__ == "__main__":
    main()
