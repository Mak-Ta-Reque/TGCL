#!/usr/bin/env python3
"""
Sanity-check for the AU-AIR dataset build (preprocessing/build_auair_dataset.py):
are the object crops actually big enough for the pipeline's VLM
(`VLM_MODEL` in .env) to recognize the object they contain?

AU-AIR frames hold dozens of tiny objects (median bbox ~2-3% of a
1920x1080 frame -- see build_auair_dataset.py's docstring), so unlike
coco10, every crop's *content* is a small, padded-out patch rather than a
naturally object-centric photo. This script:

  1. Reports crop-size statistics (raw bbox size from manifest.json, and
     actual saved crop file size) per category and split, so you can see
     how aggressively `--min-crop-size` padding is inflating tiny bboxes.
  2. Samples crops stratified by size, feeds each one to the VLM with the
     project's standard MCQ prompt (same template as
     inference/vlm_explainer_multibatch.py's single_object mcq mode) asking
     it to name the object from the 8 AU-AIR categories, and checks the
     answer against ground truth.
  3. Bins accuracy by crop-size bucket + prints a confusion matrix, so a
     size -> detectability relationship (if any) is visible directly.

Usage:
    python scripts/check_auair_crop_detectability.py
    python scripts/check_auair_crop_detectability.py --num-samples 200 --device cuda:0
    python scripts/check_auair_crop_detectability.py --auair-dir /path/to/data/auair --env-file /path/to/.env
"""
import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

AUAIR_CLASSES = ["human", "car", "truck", "van", "motorbike", "bicycle", "bus", "trailer"]

# Size buckets on the crop's shorter side, in pixels.
SIZE_BUCKETS = [(0, 160), (160, 250), (250, 400), (400, float("inf"))]


def bucket_label(short_side: int) -> str:
    for lo, hi in SIZE_BUCKETS:
        if lo <= short_side < hi:
            return f"{lo}-{int(hi) if hi != float('inf') else 'inf'}"
    return "unknown"


# ---------------------------------------------------------------------------
# Step 1: crop-size statistics
# ---------------------------------------------------------------------------
def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    values = sorted(values)
    return {
        "n": len(values),
        "min": values[0],
        "max": values[-1],
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
    }


def crop_size_stats(auair_dir: Path, manifest: dict) -> dict:
    """Bbox size (pre-crop, from manifest) vs saved file size (post-crop,
    read off disk) per category/split. File size reflects context padding +
    the --min-crop-size floor; bbox size is the raw AU-AIR annotation."""
    stats: Dict[str, dict] = {}
    for category in manifest["categories"]:
        cat_stats = {}
        for split in ("train", "val"):
            bbox_shorts, file_shorts = [], []
            for inst in manifest["data"][category][split]:
                bbox_shorts.append(min(inst["bbox"]["width"], inst["bbox"]["height"]))
                out_file = auair_dir / inst["output_file"]
                if out_file.exists():
                    with Image.open(out_file) as im:
                        file_shorts.append(min(im.size))
            cat_stats[split] = {
                "bbox_short_side": summarize(bbox_shorts),
                "saved_crop_short_side": summarize(file_shorts),
            }
        stats[category] = cat_stats
    return stats


def print_crop_size_stats(stats: dict) -> None:
    print("\n=== Crop size statistics (shorter side, px) ===")
    header = f"{'category':10s} {'split':5s} {'n':>4s} {'bbox min':>9s} {'bbox med':>9s} {'bbox max':>9s} {'saved med':>10s} {'saved min':>10s}"
    print(header)
    for category, per_split in stats.items():
        for split, s in per_split.items():
            b, f = s["bbox_short_side"], s["saved_crop_short_side"]
            if b.get("n", 0) == 0:
                continue
            print(
                f"{category:10s} {split:5s} {b['n']:4d} {b['min']:9.0f} {b['median']:9.0f} {b['max']:9.0f} "
                f"{f.get('median', float('nan')):10.0f} {f.get('min', float('nan')):10.0f}"
            )


# ---------------------------------------------------------------------------
# Step 2: sample selection stratified by saved crop size
# ---------------------------------------------------------------------------
def collect_val_pool(auair_dir: Path, manifest: dict) -> List[dict]:
    """One entry per val-split instance actually saved to disk: path, true
    label, and short side of the *saved* crop (what the VLM actually sees)."""
    pool = []
    for category in manifest["categories"]:
        for inst in manifest["data"][category]["val"]:
            out_file = auair_dir / inst["output_file"]
            if not out_file.exists():
                continue
            with Image.open(out_file) as im:
                short_side = min(im.size)
            pool.append({"path": out_file, "label": category, "short_side": short_side})
    return pool


def stratified_sample(pool: List[dict], num_samples: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    by_bucket: Dict[str, List[dict]] = defaultdict(list)
    for item in pool:
        by_bucket[bucket_label(item["short_side"])].append(item)
    for items in by_bucket.values():
        rng.shuffle(items)

    buckets = sorted(by_bucket.keys())
    per_bucket = max(1, num_samples // max(1, len(buckets)))
    sample = []
    for b in buckets:
        sample.extend(by_bucket[b][:per_bucket])
    # Top up from whatever's left (round-robin) if under target due to small buckets.
    leftovers = [it for b in buckets for it in by_bucket[b][per_bucket:]]
    rng.shuffle(leftovers)
    sample.extend(leftovers[: max(0, num_samples - len(sample))])
    rng.shuffle(sample)
    return sample[:num_samples]


# ---------------------------------------------------------------------------
# Step 3: VLM inference
# ---------------------------------------------------------------------------
def build_mcq_prompt(choices: List[str]) -> str:
    choice_str = ", ".join(choices)
    return (
        "This image shows a single object. Choose the single best-matching "
        f"label from this list: {choice_str}. "
        "Respond with exactly one label from that list — no other words, "
        "no repetition, no punctuation."
    )


def parse_prediction(raw_text: str, choices: List[str]) -> Optional[str]:
    text = raw_text.strip().lower()
    # Exact match first, then substring containment (model sometimes adds
    # punctuation/whitespace despite the prompt asking for a bare label).
    for c in choices:
        if text == c:
            return c
    for c in choices:
        if c in text:
            return c
    return None


def run_vlm_checks(
    model_class,
    samples: List[dict],
    choices: List[str],
    max_new_tokens: int,
    logger,
) -> List[dict]:
    import torch

    prompt = build_mcq_prompt(choices)
    tokenizer = model_class.get_tokenizer()
    model = model_class.get_model()
    results = []
    for i, item in enumerate(samples):
        try:
            inputs = model_class.preprocessor(
                instruction=prompt,
                image_file=str(item["path"]),
                response="",
                generation_mode=True,
            )
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as exc:
            logger.warning(f"[{i+1}/{len(samples)}] {item['path'].name}: generation failed: {exc}")
            raw_text = ""
        pred = parse_prediction(raw_text, choices)
        results.append({
            "path": str(item["path"]),
            "true_label": item["label"],
            "short_side": item["short_side"],
            "bucket": bucket_label(item["short_side"]),
            "raw_output": raw_text,
            "pred_label": pred,
            "correct": pred == item["label"],
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"  [{i+1}/{len(samples)}] running accuracy: "
                  f"{sum(r['correct'] for r in results) / len(results):.3f}")
    return results


# ---------------------------------------------------------------------------
# Step 4: reporting
# ---------------------------------------------------------------------------
def print_report(results: List[dict], choices: List[str]) -> None:
    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    n_no_pred = sum(r["pred_label"] is None for r in results)
    print("\n=== Overall detectability ===")
    print(f"n={n}  accuracy={n_correct/n:.3f}  unparseable_outputs={n_no_pred} ({n_no_pred/n:.1%})")

    print("\n=== Accuracy by crop-size bucket (saved crop's shorter side, px) ===")
    by_bucket: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_bucket[r["bucket"]].append(r)
    for b in sorted(by_bucket.keys(), key=lambda k: float(k.split("-")[0])):
        items = by_bucket[b]
        acc = sum(x["correct"] for x in items) / len(items)
        print(f"  {b:10s} n={len(items):4d}  accuracy={acc:.3f}")

    print("\n=== Accuracy by true category ===")
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["true_label"]].append(r)
    for cat in choices:
        items = by_cat.get(cat, [])
        if not items:
            continue
        acc = sum(x["correct"] for x in items) / len(items)
        med_size = statistics.median(x["short_side"] for x in items)
        print(f"  {cat:10s} n={len(items):4d}  accuracy={acc:.3f}  median_crop_short_side={med_size:.0f}px")

    print("\n=== Confusion (rows=true, cols=predicted; 'none'=unparseable) ===")
    cols = choices + ["none"]
    print(f"{'':10s} " + " ".join(f"{c[:6]:>6s}" for c in cols))
    for true_cat in choices:
        row = Counter()
        for r in by_cat.get(true_cat, []):
            row[r["pred_label"] or "none"] += 1
        print(f"{true_cat:10s} " + " ".join(f"{row.get(c, 0):6d}" for c in cols))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--auair-dir", default=str(ROOT_DIR / "data" / "auair"),
                         help="Dataset root produced by preprocessing/build_auair_dataset.py")
    parser.add_argument("--manifest", default=None, help="Defaults to <auair-dir>/manifest.json")
    parser.add_argument("--env-file", default=str(ROOT_DIR / ".env"),
                         help="Loaded for VLM_MODEL/DEVICE/HF_HOME defaults (CLI flags override).")
    parser.add_argument("--model", default=None, help="Override VLM_MODEL from .env")
    parser.add_argument("--device", default=None, help="Override DEVICE from .env")
    parser.add_argument("--num-samples", type=int, default=120,
                         help="Total val-set crops to run through the VLM, stratified by saved crop size.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--skip-vlm", action="store_true",
                         help="Only print crop-size statistics; skip loading the VLM / running inference.")
    parser.add_argument("--out", default=str(ROOT_DIR / "outputs" / "auair_crop_check" / "results.json"))
    args = parser.parse_args()

    if os.path.exists(args.env_file):
        try:
            from dotenv import load_dotenv
            load_dotenv(args.env_file)
        except ImportError:
            print(f"Warning: python-dotenv not installed, skipping {args.env_file}", file=sys.stderr)

    auair_dir = Path(args.auair_dir)
    manifest_path = Path(args.manifest) if args.manifest else auair_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    stats = crop_size_stats(auair_dir, manifest)
    print_crop_size_stats(stats)

    if args.skip_vlm:
        return

    model_name = args.model or os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    device_str = args.device or os.environ.get("DEVICE", "cuda:0")

    pool = collect_val_pool(auair_dir, manifest)
    if not pool:
        print(f"No val crops found under {auair_dir}; nothing to check.", file=sys.stderr)
        return
    samples = stratified_sample(pool, args.num_samples, args.seed)
    print(f"\nSampled {len(samples)} val crops (of {len(pool)} available), stratified by saved crop size.")

    from helpers.logger import setup_logger
    from models import get_model_class
    from device_utils import get_device_config

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=str(out_path.parent / "logs.log"))

    device_config = get_device_config(device_str)
    logger.info(f"Loading {model_name} on {device_config.raw} (primary={device_config.primary_device})")
    model_args = argparse.Namespace(local_files_only=args.local_files_only, cache_dir=args.cache_dir)
    model_class = get_model_class(
        model_name,
        processor_name=None,
        device=device_config.primary_device,
        logger=logger,
        args=model_args,
        device_config=device_config,
    )

    print(f"\n=== Running VLM ({model_name}) on {len(samples)} sampled crops ===")
    results = run_vlm_checks(model_class, samples, AUAIR_CLASSES, args.max_new_tokens, logger)

    print_report(results, AUAIR_CLASSES)

    with open(out_path, "w") as f:
        json.dump({
            "model": model_name,
            "device": device_str,
            "num_samples": len(results),
            "classes": AUAIR_CLASSES,
            "crop_size_stats": stats,
            "results": results,
        }, f, indent=2)
    print(f"\nWrote full results to {out_path}")


if __name__ == "__main__":
    main()
