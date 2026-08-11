# Quickstart: CGDL contrastive run + evaluation

Minimal instructions for building the coco10 dataset, running one pipeline
config with the `cgdl` (contrastive) prompt template, and evaluating it. For
the full multi-config ablation matrix and per-metric implementation details,
see [`docs/coco10_ablation_methods.md`](coco10_ablation_methods.md).

## 0. Build the dataset

Needed once, before any run. Builds `data/coco10/{train_all,val,val_masked,val_grids}`
from the real COCO train2017/val2017 pools at
`/media/NVME_8TB/abka03/Projects/vlm_space/local-evidence-vlm/data/raw/`
(hardcoded default source in both scripts below — no flag needed unless your
raw COCO data lives elsewhere).

```bash
conda activate xlvlms
cd /media/NVME_8TB/abka03/Projects/xl-vlms-rsml

# Step 1: pull the 10 categories (apple, banana, bird, cake, cat, cup, dog,
# donut, knife, orange) from COCO train2017 (-> data/coco10/train_all/) and
# val2017 (-> data/coco10/val/), plus manifest.json recording per-category
# image/annotation ids for step 2.
python preprocessing/build_coco10_dataset.py \
  --train-cap 300 \
  --test-cap 50 \
  --seed 42
#   --train-cap 20 --test-cap 5   # smaller pilot subset, for a fast smoke test

# Step 2: decode each val image's ground-truth COCO segmentation mask for its
# category, crop tight + padding -> data/coco10/val_masked/<category>/, then
# assemble val_masked/ into 2x2 grids -> data/coco10/val_grids/ (what
# IMAGE_ROOT points at below).
python preprocessing/build_coco10_masked_grids.py \
  --context-pixels 16 \
  --num-grids 50 \
  --grid-n 4 \
  --image-size 384 \
  --seed 42
#   --num-grids 10   # pilot
```

Both scripts are idempotent — re-running clears and rebuilds their own output
subdirectories rather than accumulating stale files. Output layout:

```
data/coco10/
  train_all/<file>.jpg          flat pool, all categories' train images (INPUT_DIR)
  val/<category>/<file>.jpg     per-category unmasked val images
  val_masked/<category>/<file>.jpg   ground-truth-mask-cropped val images
  val_grids/*.jpg                2x2 grids of val_masked images (IMAGE_ROOT below)
  manifest.json                  per-category train/test image+annotation ids
```

`val_masked/<category>/` (one class per folder, single object per image) is
also what the object-classification F1 metric in §3 needs — `val_grids/`
alone won't work for that metric. See
[`docs/coco10_ablation_methods.md`](coco10_ablation_methods.md) §1 for more
detail (e.g. building a smaller pilot subset).

## What `cgdl` is

`PROMPT_TEMPLATE=cgdl` sends a contrastive yes/no-style prompt per image:

> "Classify the image as either `[concept]` or No `[concept]` based on its
> content. Return only the predicted label."

`[concept]` is substituted with the tag/category name being probed. The
hidden state is extracted at the position where the tag word (or "No
`[concept]`") is generated (`save_hidden_states_for_token_of_interest`),
and decomposition defaults to `per_tag` (one SNMF call per tag). See
`docs/coco10_ablation_methods.md` §3-4 for how this differs from the
`non_contrastive`/`null` templates.

## 1. Setup

```bash
conda activate xlvlms
cd /media/NVME_8TB/abka03/Projects/xl-vlms-rsml
cp .env.example .env   # if you don't already have one; edit paths as needed
```

Make sure the dataset was built (§0) before running the pipeline. Any
dataset with the same layout (flat training pool + val grids) works, not
just coco10.

## 2. Run the CGDL pipeline

Canonical entry point is `scripts/run_full_pipeline.py`, configured via env
vars:

```bash
export INPUT_DIR=data/coco10/train_all
export IMAGE_ROOT=data/coco10/val_grids
export CONCEPTS_VOCAB=src/assets/coco10_vocab.txt
export PROMPT_TEMPLATE=cgdl
export CROP_MODE=langsam              # or none, or sliding_window
export DECOMP_METHODS=snmf
export CLEAN_EXAMPLE_RATIO=0.2
export PATCH_SIZE=160
export IMAGE_BUDGET=-1                # -1 = all training images per category
export EXPL_MAX_IMAGES=-1             # -1 = all val_grids images
export EXPL_PROMPT_MODE=mcq
export EXPL_CHOICES="apple,banana,bird,cake,cat,cup,dog,donut,knife,orange"
export OUTPUT_DIR=outputs/cgdl_run

python scripts/run_full_pipeline.py --output-dir "$OUTPUT_DIR" --decomp snmf
```

A ready-made version of this (coco10, `CROP_MODE=langsam`) is
`scripts/run_coco10_full.sh` — run `scripts/run_coco10_pilot.sh` first on a
small subset to sanity-check the chain before launching the full run.

This runs the whole 8-step pipeline: dataset inference → crop generation →
feature extraction → SNMF decomposition → VLM explanation → faithfulness
eval → grounding eval → plots. Each step skip-checks its own output, so
re-running the same `OUTPUT_DIR` resumes rather than redoing finished
steps.

Output layout:

```
$OUTPUT_DIR/
├── inference/       # concepts + crops
├── features/        # extracted hidden states
├── concept/snmf/    # decomposed concept bank
├── explanations/snmf/vlm_explanations.json
├── eval/snmf/        # evaluation CSVs/JSONs (see below)
└── plots/
```

## 3. Full evaluation for this run

Steps 6-7 above already produce all evaluation output automatically. The
commands below are for re-running or debugging one eval script standalone
without redoing the whole pipeline:

```bash
CFG=outputs/cgdl_run
METHOD=snmf
CONCEPT_PATH="$CFG/concept/$METHOD/combined_concept_${METHOD}_cr0.2_raw.pth"
EXPLANATIONS="$CFG/explanations/$METHOD/vlm_explanations.json"
EVAL_OUT="$CFG/eval/$METHOD"
```

**Faithfulness (insertion/deletion AUC)** — does the concept vector
causally drive the model's confidence in the token it generated?

```bash
python eval/concept_deletion_eval.py \
  --results_json "$EXPLANATIONS" \
  --concept_path "$CONCEPT_PATH" \
  --model_name google/gemma-3n-E4B-it \
  --layer_path model.language_model.norm \
  --mode token --num_points 70 \
  --out_dir "$EVAL_OUT" --device cuda:0 \
  --rank 1 --insertion
# repeat per --rank {1,2,3}, with/without --insertion, and with
# --order_mode random for the chance-level baseline

python eval/concept_curve_auc_eval.py \
  --out_dir "$EVAL_OUT" --top_n 3 --mode token \
  --output_prefix concept_curve_auc_token
# integrates the curves above into concept_curve_auc_token_table.csv
```

**Grounding (BERTScore / CLIPScore)** — does the concept's label/images
actually match what the model is describing?

```bash
python eval/clip_bert_score_eval.py \
  --json_path "$EXPLANATIONS" \
  --concept_path "$CONCEPT_PATH" \
  --max_k 3 --seed 42 \
  --out_dir "$EVAL_OUT" --output_prefix clip_bert_topk
# writes clip_bert_topk_table.csv
```

To force evaluation to redo after changing an eval script, delete the
`eval/` output and re-run the full pipeline command (step 2) — every other
step's output already exists, so only eval reruns:

```bash
rm -rf "$CFG/eval"
python scripts/run_full_pipeline.py --output-dir "$CFG" --decomp snmf
```

**Object-classification F1 (concept identity vs. ground truth)** — treats
the concept the pipeline assigns an image (its top-1, or top-1-or-2, ranked
concept from `vlm_explanations.json`) as a predicted class label, and scores
it against the image's true object class. Two metrics: top-1/top-2 accuracy
(plain hit rate) and macro F1 (per-class precision/recall, averaged over
classes — for top-2 this uses a containment criterion since two guesses are
allowed per image). Implemented in `scripts/detect_object_topk_f1.py`.

Ground truth is read from each image's **parent directory name**
(`.../<class>/<file>.jpg`), not a separate manifest — so this metric only
works on eval images laid out one class per folder, e.g.
`data/coco10/val_masked/<class>/`. It does **not** work on the `val_grids`
2×2 grid images used elsewhere in this doc, since a grid's four quadrants
aren't distinguishable in the explainer's per-image output JSON (only the
whole grid gets one predicted label). Point `IMAGE_ROOT` at a single-object
directory instead when you want this metric — e.g. rerun step 2's `EXPL_*`
config with `IMAGE_ROOT=data/coco10/val_masked`, or build a fixed subset
with `preprocessing/build_rebuttal_eval_set.py`.

`scripts/detect_object_topk_f1.py` as shipped is hardcoded to the 10-class
TGCL rebuttal ablation grid (`outputs/rebuttal_ablation_10class/`, iterating
every config from `scripts/run_rebuttal_ablation_10class.py`'s `build_grid()`).
If you've run that grid:

```bash
python scripts/detect_object_topk_f1.py
# writes outputs/rebuttal_ablation_10class/_report/object_detection_topk.csv
#   and .../object_detection_topk_summary.csv (mean/std over seeds)
```

For a single ad-hoc run (like `$CFG` above) with a per-class `IMAGE_ROOT`,
pass its `vlm_explanations.json` via `--explanations` instead of scanning
the ablation grid:

```bash
python scripts/detect_object_topk_f1.py \
  --explanations "$EXPLANATIONS" \
  --out "$EVAL_OUT/object_detection_topk.csv"
# --classes a,b,c to override the default coco10 10-class list if needed
```

Prints `n_images`, `top1_accuracy`, `top1_f1_macro`, `top2_accuracy`,
`top2_f1_macro` and writes them as a one-row CSV at `--out` (default:
next to the explanations file).

## More detail

For the full multi-config ablation grid (`scripts/run_ablation.py`), the
`CROP_MODE`/`DECOMP_STRATEGY` axes, and exactly what each metric measures
and where it's computed, see
[`docs/coco10_ablation_methods.md`](coco10_ablation_methods.md).
