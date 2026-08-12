#!/usr/bin/env python3
"""
Build an object-centric "auair8" dataset from the raw AU-AIR drone-video data,
analogous in output layout to coco10 (see build_coco10_dataset.py) but with a
crucial difference: AU-AIR source frames are 1920x1080 drone footage with
dozens of small objects per frame (median ~2-3, up to 56), so -- unlike COCO
photos, which are already reasonably object-centric -- both the train pool
*and* the val set are built by cropping each individual object instance out
of its frame, not by keeping whole frames. This keeps each dataset image to
one (or incidentally a couple of nearby) object rather than a whole cluttered
scene.

Source: data/auair/auair2019data.zip (images/*.jpg, 32,824 frames) +
data/auair/auair2019annotations.zip (annotations.json, one entry per frame
with a list of bboxes: {top, left, width, height, class}).

Categories (AU-AIR's own 8, index = "class" in each bbox):
    Human, Car, Truck, Van, Motorbike, Bicycle, Bus, Trailer
saved lowercase in the vocab file / folder names: human, car, truck, van,
motorbike, bicycle, bus, trailer.

Train/val split: AU-AIR has no official split, and frames are highly
correlated within a recording session (consecutive video frames), so a
random per-image split would leak near-duplicate frames across train/val.
Instead we split by *recording session* (parsed from the filename,
frame_<session-id>_<x|xx>_<idx>.jpg -- 12 sessions total): a fixed subset of
sessions is held out entirely for val, the rest go to train. Default held-out
sessions were chosen to include every category with a workable number of
instances (see docstring constant VAL_SESSIONS below).

Per category: up to TRAIN_CAP instances are randomly sampled from the train
sessions' pool; up to TEST_CAP instances are taken from the val sessions'
pool, largest-bbox-first (mirrors coco10's "prefer legible instances for val"
convention) -- capped by whatever's actually available (rarest class, Bus,
only has ~33-700 instances total).

Each selected bbox instance is cropped from its source frame with context
padding (proportional to bbox size, plus a pixel floor for tiny boxes) and,
if still smaller than MIN_CROP_SIZE after padding, expanded further (clamped
to the frame bounds) so no output image is left too small to be legible.

Output layout (mirrors data/coco10):
    data/auair/train_all/<category>_<frame-stem>_<i>.jpg   flat pool, all
                                                             categories' train
                                                             crops (INPUT_DIR)
    data/auair/val/<category>/<frame-stem>_<i>.jpg          per-category val
                                                             crops (one object
                                                             per image)
    data/auair/val_grids/*.jpg                              2x2 grids of val
                                                             crops (IMAGE_ROOT)
    data/auair/manifest.json                                full record: per
                                                             category train/val
                                                             instance list
                                                             (image name, bbox,
                                                             output file)

Usage:
    python preprocessing/build_auair_dataset.py
    python preprocessing/build_auair_dataset.py --train-cap 20 --test-cap 5 --num-grids 10  # pilot
"""

import argparse
import json
import random
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(Path(__file__).parent))
from create_grids import create_image_grids  # noqa: E402

DEFAULT_DATA_ZIP = ROOT_DIR / "data" / "auair" / "auair2019data.zip"
DEFAULT_ANNOTATIONS_ZIP = ROOT_DIR / "data" / "auair" / "auair2019annotations.zip"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "auair"
DEFAULT_VOCAB_PATH = ROOT_DIR / "src" / "assets" / "auair_vocab.txt"

# Recording sessions held out for val (date-id, camera-marker parsed from
# frame_<date-id>_<marker>_<idx>.jpg). Chosen to give every category,
# including the rarest (Motorbike, Bus), a non-trivial val pool -- see the
# module docstring / project data-exploration notes for per-session counts.
VAL_SESSIONS = {("20190905112522", "xx"), ("20190906150731", "xx")}

FRAME_NAME_RE = re.compile(r"frame_(\d+)_([a-z]+)_(\d+)\.jpg")


def session_of(image_name: str) -> Tuple[str, str]:
    m = FRAME_NAME_RE.match(image_name)
    if not m:
        raise ValueError(f"Unrecognized AU-AIR frame filename: {image_name}")
    return m.group(1), m.group(2)


def load_annotations(annotations_zip: Path) -> dict:
    with zipfile.ZipFile(annotations_zip) as z:
        with z.open("annotations.json") as f:
            return json.load(f)


def collect_instances(
    data: dict, val_sessions: set, min_bbox_size: int
) -> Tuple[Dict[str, Dict[str, List[dict]]], List[str]]:
    """Return {category: {"train": [...], "val": [...]}} of bbox instances,
    each {"image_name", "bbox", "area"}; degenerate/near-invisible bboxes
    (width/height <= 0, or smaller than min_bbox_size on either side) are
    dropped. Also returns the lowercase category name list, index-aligned
    with AU-AIR's own "class" ids."""
    categories = [c.lower() for c in data["categories"]]
    instances: Dict[str, Dict[str, List[dict]]] = {c: {"train": [], "val": []} for c in categories}

    n_degenerate = 0
    for entry in data["annotations"]:
        image_name = entry["image_name"]
        split = "val" if session_of(image_name) in val_sessions else "train"
        for bbox in entry["bbox"]:
            w, h = bbox["width"], bbox["height"]
            if w <= 0 or h <= 0 or min(w, h) < min_bbox_size:
                n_degenerate += 1
                continue
            category = categories[bbox["class"]]
            instances[category][split].append({
                "image_name": image_name,
                "bbox": {"top": bbox["top"], "left": bbox["left"], "width": w, "height": h},
                "area": w * h,
            })

    print(f"Dropped {n_degenerate} degenerate/too-small bboxes (< {min_bbox_size}px on a side).")
    return instances, categories


def select_per_category(
    instances: Dict[str, Dict[str, List[dict]]], seed: int, train_cap: int, test_cap: int
) -> Dict[str, Dict[str, List[dict]]]:
    rng = random.Random(seed)
    selected: Dict[str, Dict[str, List[dict]]] = {}
    for category, splits in instances.items():
        train_pool = list(splits["train"])
        rng.shuffle(train_pool)
        train_selected = train_pool[:train_cap]

        # Val: largest instances first, so the object is actually legible in
        # the eval crop (same rationale as coco10's mask-area sort).
        val_pool = sorted(splits["val"], key=lambda inst: inst["area"], reverse=True)
        val_selected = val_pool[:test_cap]

        selected[category] = {"train": train_selected, "val": val_selected}
    return selected


def compute_crop_box(
    bbox: dict, img_w: int, img_h: int,
    context_ratio: float, min_context_pixels: int, min_crop_size: int,
) -> Tuple[int, int, int, int]:
    left = max(0.0, float(bbox["left"]))
    top = max(0.0, float(bbox["top"]))
    right = min(float(img_w), left + bbox["width"])
    bottom = min(float(img_h), top + bbox["height"])
    w, h = right - left, bottom - top

    pad = max(min_context_pixels, context_ratio * max(w, h))
    x1, y1, x2, y2 = left - pad, top - pad, right + pad, bottom + pad

    def expand_to_min(a1: float, a2: float, min_size: int, limit: int) -> Tuple[float, float]:
        deficit = min_size - (a2 - a1)
        if deficit > 0:
            a1 -= deficit / 2
            a2 += deficit / 2
        return max(0.0, a1), min(float(limit), a2)

    x1, x2 = expand_to_min(x1, x2, min_crop_size, img_w)
    y1, y2 = expand_to_min(y1, y2, min_crop_size, img_h)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def build_crops(
    selected: Dict[str, Dict[str, List[dict]]],
    data_zip: Path,
    output_dir: Path,
    context_ratio: float,
    min_context_pixels: int,
    min_crop_size: int,
) -> None:
    train_all_dir = output_dir / "train_all"
    val_dir = output_dir / "val"
    if train_all_dir.exists():
        shutil.rmtree(train_all_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
    train_all_dir.mkdir(parents=True, exist_ok=True)
    for category in selected:
        (val_dir / category).mkdir(parents=True, exist_ok=True)

    # Group by source frame so each frame is decoded from the zip only once,
    # even if it contributes multiple selected instances (own or other cats).
    by_image: Dict[str, List[dict]] = defaultdict(list)
    for category, splits in selected.items():
        for split in ("train", "val"):
            for i, inst in enumerate(splits[split]):
                stem = Path(inst["image_name"]).stem
                if split == "train":
                    out_path = train_all_dir / f"{category}_{stem}_{i}.jpg"
                else:
                    out_path = val_dir / category / f"{stem}_{i}.jpg"
                by_image[inst["image_name"]].append({"category": category, "bbox": inst["bbox"], "out_path": out_path})
                inst["output_file"] = str(out_path.relative_to(output_dir))

    n_saved = 0
    with zipfile.ZipFile(data_zip) as z:
        for i, (image_name, crops) in enumerate(by_image.items(), 1):
            with z.open(f"images/{image_name}") as f:
                image = Image.open(BytesIO(f.read())).convert("RGB")
            img_w, img_h = image.size
            for c in crops:
                x1, y1, x2, y2 = compute_crop_box(
                    c["bbox"], img_w, img_h, context_ratio, min_context_pixels, min_crop_size
                )
                image.crop((x1, y1, x2, y2)).save(c["out_path"], quality=95)
                n_saved += 1
            if i % 500 == 0:
                print(f"  processed {i}/{len(by_image)} source frames, {n_saved} crops saved so far")

    print(f"Saved {n_saved} object crops from {len(by_image)} source frames.")


def write_vocab(categories: List[str], vocab_path: Path) -> None:
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w") as f:
        f.write("\n".join(categories) + "\n")
    print(f"Saved vocab: {vocab_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-cap", type=int, default=200, help="Max train instances per category")
    parser.add_argument("--test-cap", type=int, default=50, help="Max val instances per category")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-bbox-size", type=int, default=6, help="Drop bboxes smaller than this on either side (px)")
    parser.add_argument("--context-ratio", type=float, default=0.5, help="Padding added per side, as a fraction of the bbox's larger dimension")
    parser.add_argument("--min-context-pixels", type=int, default=20, help="Pixel floor for padding, so tiny bboxes still get real context")
    parser.add_argument("--min-crop-size", type=int, default=160, help="Minimum output crop side length (px); padding is expanded, clamped to frame bounds, to reach this")
    parser.add_argument("--num-grids", type=int, default=50)
    parser.add_argument("--grid-n", type=int, default=4, help="Images per grid (perfect square, default 2x2=4)")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--data-zip", default=str(DEFAULT_DATA_ZIP))
    parser.add_argument("--annotations-zip", default=str(DEFAULT_ANNOTATIONS_ZIP))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--vocab-path", default=str(DEFAULT_VOCAB_PATH))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_zip = Path(args.data_zip)
    annotations_zip = Path(args.annotations_zip)

    print("Loading annotations...")
    data = load_annotations(annotations_zip)

    print("Collecting bbox instances (train/val split by recording session)...")
    instances, categories = collect_instances(data, VAL_SESSIONS, args.min_bbox_size)

    print("Selecting per-category instances...")
    selected = select_per_category(instances, seed=args.seed, train_cap=args.train_cap, test_cap=args.test_cap)
    for category, splits in selected.items():
        print(f"  {category:10s}: train={len(splits['train']):4d}  val={len(splits['val']):3d}")

    print("\nCropping objects (this decodes each needed source frame once)...")
    build_crops(
        selected, data_zip, output_dir,
        context_ratio=args.context_ratio,
        min_context_pixels=args.min_context_pixels,
        min_crop_size=args.min_crop_size,
    )

    manifest = {
        "categories": categories,
        "seed": args.seed,
        "val_sessions": sorted(VAL_SESSIONS),
        "params": {
            "train_cap": args.train_cap,
            "test_cap": args.test_cap,
            "min_bbox_size": args.min_bbox_size,
            "context_ratio": args.context_ratio,
            "min_context_pixels": args.min_context_pixels,
            "min_crop_size": args.min_crop_size,
        },
        "data": {
            category: {"train": splits["train"], "val": splits["val"]}
            for category, splits in selected.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f)
    print(f"\nSaved manifest: {manifest_path}")

    write_vocab(categories, Path(args.vocab_path))

    val_dir = output_dir / "val"
    grids_dir = output_dir / "val_grids"
    if grids_dir.exists():
        shutil.rmtree(grids_dir)
    print(f"\nBuilding {args.num_grids} grids ({args.grid_n}-cell) from val crops...")
    random.seed(args.seed)
    create_image_grids(
        input_dir=str(val_dir),
        output_dir=str(grids_dir),
        n=args.grid_n,
        num_grids=args.num_grids,
        image_size=args.image_size,
    )

    print(f"\ntrain_all pool: {output_dir / 'train_all'}")
    print(f"val (per-category): {val_dir}")
    print(f"val_grids: {grids_dir}")


if __name__ == "__main__":
    main()
