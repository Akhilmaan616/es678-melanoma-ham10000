# Melanoma detection on HAM10000: lesion-level evaluation and failure-mode audit

ES 678 (Graduate) course project, IIT Gandhinagar. Akhil Maan.

How well can a classifier separate melanoma from other pigmented lesions in HAM10000 dermoscopy
images when it is tested only on lesions it has never seen? How much does a naive image-level split
overstate that, and where do errors and miscalibration concentrate?

Report: [`report/main.pdf`](report/main.pdf) (LaTeX source `report/main.tex`, ES 678 template, 4 pages).

## Main results (lesion-grouped test set, 1,121 lesions, 92 melanomas)

| Model | Melanoma AUROC (95% CI) | AUPRC | Sens / Spec at val 90%-sens threshold | Balanced acc (7 classes) |
| --- | --- | --- | --- | --- |
| Majority class | 0.500 | 0.082 | 0.00 / 1.00 | 0.143 |
| Metadata-only logistic regression | 0.645 [0.580, 0.703] | 0.136 | 0.87 / 0.22 | 0.321 |
| ResNet-50 linear probe | 0.852 [0.812, 0.887] | 0.396 | 0.79 / 0.75 | 0.616 |
| ResNet-50 fine-tuned (3 seeds) | 0.940 [0.919, 0.957] | 0.642 | 0.84 / 0.88 | 0.785 |
| EfficientNet-B0 fine-tuned (3 seeds) | 0.939 [0.921, 0.956] | 0.663 | 0.79 / 0.90 | 0.782 |

* The 90% sensitivity target set on validation gave 79% on test (19 of 92 melanomas missed).
* **Leakage:** on identical test images, a model trained on an image-level split beats the
  lesion-split model only on lesions whose sibling photos it saw in training (AUROC 0.962 vs 0.897),
  and not on unseen lesions (0.950 vs 0.954).
* **Failure modes:** false positives concentrate in biopsied benign lesions (specificity 0.76,
  AUROC 0.86 within biopsied lesions vs 0.94 overall). Grad-CAM on all missed melanomas points at the
  lesion, not at rulers, hair or vignettes.
* **Calibration:** the CNNs are over-confident (temperature 1.4 to 1.6). Temperature scaling helps
  image-level ECE (0.078 to 0.069) but does not fix subgroup miscalibration.

Full tables are in `results/` (`main_results.csv`, `leakage_results.csv`, `leakage_paired.json`,
`subgroup_audit.csv`, `summary.json`, `data_checks.json`).

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/prepare.py` | Data-quality checks, the fixed lesion-grouped split and the naive image-level split, 224 x 224 image cache |
| `src/baselines.py` | Majority-class reference, metadata-only logistic regression, ResNet-50 linear probe |
| `src/train_cnn.py` | Fine-tuning of EfficientNet-B0 / ResNet-50 with class-weighted cross-entropy |
| `src/run_all.py` | Learning-rate sweep on validation, three seeds per model, image-split runs for the leakage analysis |
| `src/evaluate.py` | Test metrics, lesion-level bootstrap CIs, temperature scaling, leakage gap, subgroup audit, figures |
| `src/leakage_paired.py` | Paired leakage check on identical images (seen vs unseen lesions) |
| `src/gradcam.py` | Grad-CAM on missed test melanomas |
| `splits/` | Saved split files (`image_id, lesion_id, split`), seed 42 |
| `outputs/preds/` | Validation and test logits of every trained model, plus training logs |
| `results/` | Result tables (CSV/JSON) and figures used in the report |
| `report/` | LaTeX source, style file, figures and compiled PDF |

Images are not redistributed and model checkpoints are not committed (they can be regenerated).

## How to reproduce

1. Download `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip` and `HAM10000_metadata.tab`
   from Harvard Dataverse (doi:10.7910/DVN/DBW86T) into `data/raw/`. Leave the zips unextracted.
2. Create an environment (Python 3.11, CUDA GPU) and install `requirements.txt`.
3. Run, from the repository root:

```bash
python src/prepare.py          # checks, splits, image cache
python src/baselines.py        # majority, metadata LR, linear probe (both splits)
python src/run_all.py          # CNN sweep + seeds (about 1.5 h on an RTX 4050 laptop GPU)
python src/evaluate.py         # tables and figures in results/ (about 12 min of bootstrap)
python src/leakage_paired.py   # paired leakage check
python src/gradcam.py          # Grad-CAM of missed melanomas
```

`evaluate.py --figures-only` redraws the calibration/subgroup figure without rerunning the bootstrap.

## Design rules followed

* One row is one image; every learned step is fit on the training split only.
* The lesion-grouped split (70/15/15 of lesions, stratified by diagnosis) is the main evaluation.
  The image-level split is only used to measure leakage, never to choose a model.
* Learning rate, checkpoint epoch, regularisation strength, the 90%-sensitivity threshold and the
  temperature are all chosen on validation. The test set is scored once per trained model.
* Inputs are the image (and, for the metadata baseline, age, sex and body site). The confirmation
  method (`dx_type`) is never an input; it is used only to analyse errors.
* Confidence intervals come from resampling test lesions (1,000 replicates); for three-seed models
  each replicate averages the metric over seeds.
* Age recorded as 0 (39 images) is treated as missing.

## Notes

* Training ran locally on an RTX 4050 laptop GPU rather than Colab. On Windows machines with
  Smart App Control, recent pandas/matplotlib/scikit-learn wheels may be blocked; the pinned versions
  in `requirements.txt` work.
* HAM10000 (Tschandl et al., 2018) is released for non-commercial research under CC BY-NC 4.0.

## AI-use disclosure

I used Claude (Anthropic) for brainstorming and for some help with coding. I checked the code, the
numbers and the report myself.
