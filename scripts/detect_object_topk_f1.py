#!/usr/bin/env python3
"""
Object-detection-via-concept-identity F1 for the 10-class rebuttal ablation.

Question this answers: independent of insertion/deletion faithfulness, does
the concept the pipeline actually assigns to an image (by cosine similarity,
via explanations/snmf/vlm_explanations.json's per_token_concepts.top_concepts,
already ranked by "similarity") correctly name that image's true object class?

For every image, exactly one generated token is stored (the explainer's
single-word MCQ answer), with up to 3 ranked candidate concepts, each with a
`concept_name` list (5 grounding words for that concept -- normally identical,
so the mode is taken defensively). Ground truth is read directly off the
image path's own parent directory name (.../val_masked_rebuttal10/<class>/...),
not a separate manifest -- self-contained, no cross-file lookup needed.

Two hit criteria:
  - top-1: does the rank-1 concept's name equal the true class?
  - top-2: does the rank-1 OR rank-2 concept's name equal the true class?

Two metrics per criterion:
  - accuracy: plain fraction of images that hit
  - macro F1: for top-1, standard sklearn multiclass F1 (one predicted label
    per image). For top-2, there's no single predicted label per image (two
    guesses allowed), so F1 is computed per class from a containment
    criterion: TP_c = true==c and c is among the (up to 2) guesses,
    FP_c = true!=c and c is among the guesses, FN_c = true==c and c is not
    among the guesses -- then macro-averaged over the 10 classes. This is
    the standard way multi-guess "top-k F1" is defined when the alternative
    (plain top-k accuracy) doesn't answer per-class precision/recall.

Writes (default, full-grid mode):
  outputs/rebuttal_ablation_10class/_report/object_detection_topk.csv
    per (condition, crop_mode, seed): n_images, top1/top2 accuracy + F1
  outputs/rebuttal_ablation_10class/_report/object_detection_topk_summary.csv
    per (condition, crop_mode): mean/std over seeds of the same 4 metrics

Usage:
    python scripts/detect_object_topk_f1.py
    # Score one arbitrary run's vlm_explanations.json instead of the ablation
    # grid (ground truth is read from each image path's own parent directory
    # name -- this needs a per-class IMAGE_ROOT layout, e.g. val_masked/<class>/,
    # not the multi-object val_grids/ layout used by the main cgdl pipeline):
    python scripts/detect_object_topk_f1.py \\
        --explanations outputs/cgdl_run/explanations/snmf/vlm_explanations.json
    # Override the class list if it's not coco10's 10 categories:
    python scripts/detect_object_topk_f1.py \\
        --explanations outputs/cgdl_run/explanations/snmf/vlm_explanations.json \\
        --classes apple,banana,cat
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from run_rebuttal_ablation_10class import build_grid, config_name  # noqa: E402
from posthoc_auc_curves import COND_ORDER  # noqa: E402

ABLATION_ROOT = ROOT_DIR / "outputs" / "rebuttal_ablation_10class"
METHOD = "snmf"
CLASSES = ["apple", "banana", "bird", "cake", "cat", "cup", "dog", "donut", "knife", "orange"]


def _concept_label(concept_name: Optional[List[str]]) -> Optional[str]:
    if not concept_name:
        return None
    return Counter(concept_name).most_common(1)[0][0]


def rows_from_explanations(path: Path) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Returns [(true_label, pred_rank1, pred_rank2), ...] from one vlm_explanations.json."""
    if not path.exists():
        return []
    with open(path) as f:
        d = json.load(f)
    out = []
    for item in d.get("results", []):
        true_label = Path(item["image_path"]).parent.name
        ptc = item.get("per_token_concepts") or []
        if not ptc:
            out.append((true_label, None, None))
            continue
        top_concepts = ptc[0].get("top_concepts") or []
        by_rank = {c["rank"]: c for c in top_concepts}
        pred1 = _concept_label(by_rank.get(1, {}).get("concept_name"))
        pred2 = _concept_label(by_rank.get(2, {}).get("concept_name"))
        out.append((true_label, pred1, pred2))
    return out


def load_predictions(config_name_str: str) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Returns [(true_label, pred_rank1, pred_rank2), ...] for one ablation-grid config."""
    path = ABLATION_ROOT / config_name_str / "explanations" / METHOD / "vlm_explanations.json"
    return rows_from_explanations(path)


def _top2_f1_macro(y_true: List[str], pred1: List[Optional[str]], pred2: List[Optional[str]],
                    classes: Optional[List[str]] = None) -> float:
    f1s = []
    for c in (classes or CLASSES):
        tp = fp = fn = 0
        for t, p1, p2 in zip(y_true, pred1, pred2):
            guessed = c in (p1, p2)
            if t == c:
                if guessed:
                    tp += 1
                else:
                    fn += 1
            elif guessed:
                fp += 1
        if tp + fn == 0:
            continue  # class not present in this config's eval set
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")


def compute_metrics(rows: List[Tuple[str, Optional[str], Optional[str]]],
                     classes: Optional[List[str]] = None) -> dict:
    classes = classes or CLASSES
    y_true = [r[0] for r in rows]
    pred1 = [r[1] for r in rows]
    pred2 = [r[2] for r in rows]

    top1_hits = [t == p1 for t, p1 in zip(y_true, pred1)]
    top2_hits = [h or (t == p2) for h, t, p2 in zip(top1_hits, y_true, pred2)]

    pred1_filled = [p if p is not None else "__none__" for p in pred1]
    top1_f1 = f1_score(y_true, pred1_filled, average="macro", labels=classes, zero_division=0)
    top2_f1 = _top2_f1_macro(y_true, pred1, pred2, classes=classes)

    return {
        "n_images": len(rows),
        "top1_accuracy": float(np.mean(top1_hits)),
        "top1_f1_macro": float(top1_f1),
        "top2_accuracy": float(np.mean(top2_hits)),
        "top2_f1_macro": float(top2_f1),
    }


def write_per_config_csv(rows: List[dict], out_path: Path) -> None:
    cols = ["condition", "crop_mode", "seed", "n_images",
            "top1_accuracy", "top1_f1_macro", "top2_accuracy", "top2_f1_macro"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])
    print(f"Wrote {out_path} ({len(rows)} rows)")


def write_summary_csv(rows: List[dict], out_path: Path) -> None:
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["condition"], r["crop_mode"])].append(r)

    summary = []
    for (cond, crop), grp in grouped.items():
        entry = {"condition": cond, "crop_mode": crop, "n_seeds": len(grp),
                  "n_images_total": sum(r["n_images"] for r in grp)}
        for metric in ["top1_accuracy", "top1_f1_macro", "top2_accuracy", "top2_f1_macro"]:
            vals = [r[metric] for r in grp]
            entry[f"{metric}_mean"] = float(np.mean(vals))
            entry[f"{metric}_std"] = float(np.std(vals)) if len(vals) > 1 else None
        summary.append(entry)
    summary.sort(key=lambda r: (COND_ORDER.index(r["condition"]) if r["condition"] in COND_ORDER else 99, r["crop_mode"]))

    cols = ["condition", "crop_mode", "n_seeds", "n_images_total",
            "top1_accuracy_mean", "top1_accuracy_std",
            "top1_f1_macro_mean", "top1_f1_macro_std",
            "top2_accuracy_mean", "top2_accuracy_std",
            "top2_f1_macro_mean", "top2_f1_macro_std"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in summary:
            w.writerow([r.get(c) for c in cols])
    print(f"Wrote {out_path} ({len(summary)} rows)")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--explanations", type=Path, default=None,
        help="Path to a single vlm_explanations.json to score, instead of scanning the "
             "full 10-class rebuttal ablation grid. Ground truth is read from each "
             "image path's own parent directory name, so IMAGE_ROOT must be a "
             "per-class layout (e.g. data/coco10/val_masked/<class>/), not val_grids/.",
    )
    parser.add_argument(
        "--classes", type=str, default=None,
        help="Comma-separated ground-truth class names for --explanations mode "
             "(default: the 10 coco10 categories baked into this script).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path for --explanations mode "
             "(default: <explanations_dir>/object_detection_topk_single.csv).",
    )
    return parser.parse_args()


def run_single(explanations: Path, classes: Optional[List[str]], out: Optional[Path]) -> None:
    rows = rows_from_explanations(explanations)
    if not rows:
        print(f"No results found in {explanations}")
        return
    metrics = compute_metrics(rows, classes=classes)
    out_path = out or explanations.parent / "object_detection_topk_single.csv"
    cols = ["n_images", "top1_accuracy", "top1_f1_macro", "top2_accuracy", "top2_f1_macro"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow([metrics[c] for c in cols])
    print(f"[{explanations}] n={metrics['n_images']} "
          f"top1_acc={metrics['top1_accuracy']:.3f} top1_f1={metrics['top1_f1_macro']:.3f} "
          f"top2_acc={metrics['top2_accuracy']:.3f} top2_f1={metrics['top2_f1_macro']:.3f}")
    print(f"Wrote {out_path}")


def main() -> None:
    args = parse_args()
    if args.explanations:
        classes = args.classes.split(",") if args.classes else None
        run_single(args.explanations, classes, args.out)
        return

    out_dir = ABLATION_ROOT / "_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_config_rows = []
    seen = set()
    for condition, crop_mode, seed in build_grid():
        name = config_name(condition, crop_mode, seed)
        if name in seen:
            continue
        seen.add(name)
        raw_rows = load_predictions(name)
        if not raw_rows:
            print(f"[{name}] SKIP: no vlm_explanations.json", flush=True)
            continue
        metrics = compute_metrics(raw_rows)
        per_config_rows.append({"condition": condition, "crop_mode": crop_mode, "seed": seed, **metrics})
        print(f"[{name}] n={metrics['n_images']} "
              f"top1_acc={metrics['top1_accuracy']:.3f} top1_f1={metrics['top1_f1_macro']:.3f} "
              f"top2_acc={metrics['top2_accuracy']:.3f} top2_f1={metrics['top2_f1_macro']:.3f}", flush=True)

    per_config_rows.sort(key=lambda r: (COND_ORDER.index(r["condition"]) if r["condition"] in COND_ORDER else 99,
                                         r["crop_mode"], r["seed"]))
    write_per_config_csv(per_config_rows, out_dir / "object_detection_topk.csv")
    write_summary_csv(per_config_rows, out_dir / "object_detection_topk_summary.csv")


if __name__ == "__main__":
    main()
