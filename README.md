# Quickstart: CGDL contrastive run + evaluation

Minimal instructions for building the AU-AIR dataset, running one pipeline
config with the `cgdl` (contrastive) prompt template, and evaluating it. For
the full multi-config ablation matrix and per-metric implementation details,
see [`docs/coco10_ablation_methods.md`](coco10_ablation_methods.md) (the
mechanics documented there are dataset-agnostic despite the filename).

## 0. Build the dataset

Needed once, before any run. AU-AIR (`data/auair/auair2019data.zip` +
`auair2019annotations.zip`, 32,824 1920x1080 drone-camera frames, 8
traffic/transport categories: human, car, truck, van, motorbike, bicycle,
bus, trailer) is built into an object-centric crop layout. Its raw frames
are **not** object-centric the way ordinary photos are — a single frame can
hold up to 56 small objects at extreme distance — so **both** splits are
built by cropping each individual bbox instance out of its frame at
dataset-build time, with proportional context padding and a minimum-size
floor so no crop ends up illegibly tiny. One script does the whole thing
(selection, cropping, and grid assembly):

```bash
conda activate xlvlms
cd /media/NVME_8TB/abka03/Projects/xl-vlms-rsml

python preprocessing/build_auair_dataset.py \
  --train-cap 200 \
  --test-cap 50 \
  --seed 42
#   --train-cap 10 --test-cap 5 --num-grids 5   # pilot, for a fast smoke test
```

What it does:

1. Reads `annotations.json` straight out of `auair2019annotations.zip` (no
   need to unzip it yourself) and drops degenerate/near-invisible bboxes
   (`--min-bbox-size`, default 6px on a side).
2. Splits by **recording session** (parsed from the frame filename), not by
   random per-image split — AU-AIR frames are consecutive video, so a random
   split would leak near-duplicate frames between train and val. Two
   sessions are held out for val (hardcoded `VAL_SESSIONS` in the script,
   chosen to keep every category — including the rare ones, Bus/Motorbike —
   represented on both sides).
3. Per category, randomly samples up to `--train-cap` instances from the
   train-session pool, and takes the `--test-cap` *largest* instances (by
   bbox area) from the val-session pool, so val objects are legible — capped
   by whatever's available (Bus only has 33 val instances total).
4. For each selected instance, crops it from its source frame (decoded
   directly out of `auair2019data.zip`, once per frame even if it
   contributes multiple crops) with padding = `max(--min-context-pixels,
   --context-ratio * max(bbox_w, bbox_h))` per side, then — if the result is
   still under `--min-crop-size` (default 160px) — expands further,
   clamped to the frame's own bounds, so it never runs off the edge.
5. Assembles `val/` into 2x2 grids (`preprocessing/create_grids.py`).
6. Writes `src/assets/auair_vocab.txt` (the 8 category names, one per line).

Output layout:

```
data/auair/
  train_all/<category>_<frame-stem>_<i>.jpg   flat pool, all categories' train crops (INPUT_DIR)
  val/<category>/<frame-stem>_<i>.jpg          per-category val crops, one object per image
  val_grids/*.jpg                              2x2 grids of val crops (IMAGE_ROOT below)
  manifest.json                                per-category train/val instance list (frame name, bbox, output file)
src/assets/auair_vocab.txt                      human/car/truck/van/motorbike/bicycle/bus/trailer
```

Re-running the build script is idempotent — it clears `train_all/`, `val/`,
and `val_grids/` first rather than mixing stale crops in with new ones.

`val/<category>/` (one class per folder, single object per image) is also
what the object-classification F1 metric in §3 needs — `val_grids/` alone
won't work for that metric, since a grid's four quadrants aren't
distinguishable in the explainer's per-image output JSON.

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

### Tested hardware/software

- **GPU**: 4× NVIDIA RTX 4090 (24 GB each), driver 575.57.08. One config
  needs one GPU (`DEVICE=cuda:0`); the ablation grid
  (`run_ablation.py --devices cuda:0,cuda:1,...`) parallelizes across
  however many you point it at.
- **OS**: Ubuntu 22.04.5 LTS
- **Python**: 3.10 (conda env `xlvlms`)
- **PyTorch**: 2.12.1, CUDA 12.6 build (`+cu126`) — this is PyTorch's own
  bundled CUDA runtime; no system-wide CUDA Toolkit install is required
  (the system's own `nvcc` can be a completely different/older version,
  it's irrelevant to `torch.cuda`).
- 24 GB is comfortable at `BATCH_SIZE=16` for `google/gemma-3n-E4B-it` or
  `Qwen/Qwen2.5-VL-3B-Instruct`; raise/lower `BATCH_SIZE` to fit your card
  — see the OOM guidance in the eval section below if you hit
  `CUDA out of memory`.

### Create the environment

```bash
conda create -n xlvlms python=3.10
conda activate xlvlms
cd /media/NVME_8TB/abka03/Projects/xl-vlms-rsml

# PyTorch first, matched to your CUDA (this project was validated on cu126):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Install the libraries this pipeline actually needs

Core (always required — dataset build, feature extraction, decomposition,
explanation, evaluation):

```bash
pip install transformers huggingface_hub numpy Pillow tqdm scipy scikit-learn \
  matplotlib psutil inflect nltk python-dotenv pycocotools packaging requests spacy
python -m spacy download en_core_web_sm

pip install bert-score                                    # eval/clip_bert_score_eval.py
pip install git+https://github.com/openai/CLIP.git        # same file, + CLIP-scored crop ranking
pip install qwen-vl-utils                                 # only if using a Qwen2.5-VL model
```

Crop-mode-specific (`preprocessing/crops_to_json.py`'s `CROP_MODE`/`--detector`
lazily imports these — only install the one(s) you'll actually use;
`none`/`sliding_window` need neither):

```bash
# CROP_MODE=langsam (the example default in §2 below):
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git

# CROP_MODE=sam3: needs Meta's SAM3, not on PyPI — out of scope for this
# quickstart; only relevant if you deliberately opt into sam3 detection.
```

Not needed for this pipeline (installed in some environments for
unrelated `src/models/` classes or legacy eval paths, e.g. `pydicom` is
only used by the CheXagent model class): skip unless you hit an
`ImportError` for a feature you're deliberately using outside the `cgdl`
+ ablation workflow this doc covers.

Verify the install:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

**Known issue — `libstdc++`/ICU import crash on a fresh env**: on at least one
tested machine, a fresh `xlvlms` env hit
`ImportError: .../libstdc++.so.6: version 'CXXABI_1.3.15' not found
(required by .../libicui18n.so.78)` the first time anything imports
`nltk.corpus` (pulled in transitively by `src/metrics/`, which
`analyse_features.py` always imports). Root cause: `conda activate` on this
setup does not set `LD_LIBRARY_PATH`, so a pip-installed C extension
resolves `libstdc++` against the system copy (too old) instead of the
conda env's own newer one, even though the env has the right library
sitting right there. Fix — add this to your env activation (once):

```bash
conda activate xlvlms
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

If you hit the same `CXXABI_1.3.15` error anywhere else, this is almost
certainly the cause regardless of which step surfaces it first.

### Configure `.env`

```bash
cp .env.example .env   # if you don't already have one; edit paths as needed
```

Make sure the dataset was built (§0) before running the pipeline.

## 2. Run the CGDL pipeline

Canonical entry point is `scripts/run_full_pipeline.py`, configured via env
vars:

```bash
export INPUT_DIR=data/auair/train_all
export IMAGE_ROOT=data/auair/val_grids
export CONCEPTS_VOCAB=src/assets/auair_vocab.txt
export PROMPT_TEMPLATE=cgdl
export CROP_MODE=langsam              # or none, or sliding_window
export DECOMP_METHODS=snmf
export CLEAN_EXAMPLE_RATIO=0.2
export PATCH_SIZE=160
export IMAGE_BUDGET=-1                # -1 = all training images per category
export EXPL_MAX_IMAGES=-1             # -1 = all val_grids images
export EXPL_PROMPT_MODE=mcq
export EXPL_CHOICES="human,car,truck,van,motorbike,bicycle,bus,trailer"
export OUTPUT_DIR=outputs/auair_run

python scripts/run_full_pipeline.py --output-dir "$OUTPUT_DIR" --decomp snmf
```

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
CFG=outputs/auair_run
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
`data/auair/val/<class>/` (already this layout — no separate masking step
needed, unlike whole-image datasets). It does **not** work on the
`val_grids` 2×2 grid images used elsewhere in this doc, since a grid's four
quadrants aren't distinguishable in the explainer's per-image output JSON
(only the whole grid gets one predicted label). Point `IMAGE_ROOT` at
`data/auair/val` instead when you want this metric — i.e. rerun step 2's
`EXPL_*` config with `IMAGE_ROOT=data/auair/val` to get one
`vlm_explanations.json` entry per single-object image.

```bash
python scripts/detect_object_topk_f1.py \
  --explanations "$EXPLANATIONS" \
  --classes human,car,truck,van,motorbike,bicycle,bus,trailer \
  --out "$EVAL_OUT/object_detection_topk.csv"
```

Prints `n_images`, `top1_accuracy`, `top1_f1_macro`, `top2_accuracy`,
`top2_f1_macro` and writes them as a one-row CSV at `--out` (default:
next to the explanations file).

## More detail

For the full multi-config ablation grid (`scripts/run_ablation.py`), the
`CROP_MODE`/`DECOMP_STRATEGY` axes, and exactly what each metric measures
and where it's computed, see
[`docs/coco10_ablation_methods.md`](coco10_ablation_methods.md).
