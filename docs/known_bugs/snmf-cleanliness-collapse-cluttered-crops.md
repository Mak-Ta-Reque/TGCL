# Some AU-AIR categories never produce a usable SNMF concept direction

**Status: open (limitation, not a simple code bug)**

## Summary

Even with every other fix applied (`tgcl` prompt pinned to the exact concept
word, `bus`/`bu` inflect bug fixed, vocab-priming `PROMPT`), some AU-AIR
categories' SNMF decomposition components consistently land in the negative
concept bank with `direction_positive_ratio = 0.0` -- not "close but under
threshold," a flat zero. Lowering `CLEAN_EXAMPLE_RATIO` (tried 0.8 -> 0.5)
does not help, because there's no partially-clean signal to threshold
against.

## Evidence

Loaded `outputs/auair_run/concept/snmf/combined_concept_snmf_cr0.5_raw.pth`
/ `combined_negative_concept_snmf_cr0.5_raw.pth` after a full pipeline run
(`INPUT_DIR=data/auair/train_all`, `IMAGE_BUDGET=2000`, all fixes applied):

- **Positive bank (usable): `motorbike` (x2), `bus`, `car`, `trailer`,
  `truck`, `van`** -- all `positive_ratio = 1.0`.
- **Negative bank (unusable): `bicycle` (x2), `human` (x2)**, plus one
  discarded component each from the categories that otherwise have a clean
  direction -- `positive_ratio = 0.0` across the board.

Running `scripts/detect_object_topk_f1.py` on the resulting
`vlm_explanations.json` (383 val images, all 8 classes):
`top1_accuracy=0.287`, but per-class: `motorbike` 96%, `truck` 98%, while
`bicycle`, `human`, `van`, `trailer` are all **0%** -- confusion matrix shows
most non-truck/motorbike vehicle images get predicted as `truck` regardless
of true label.

## Why (best current understanding)

`build_auair_dataset.py`'s own docstring already flags the root cause: AU-AIR
frames hold up to 56 objects at extreme distance, so crops aren't naturally
object-centric the way COCO photos are -- the padded context around a
`bicycle` or `human` bbox very often contains *other* AU-AIR objects (other
bikes, vehicles in traffic). The `tgcl` contrastive hidden state for
"[tag] or No [tag]" then reflects whichever object the model's attention
actually lands on, not necessarily the intended bbox -- for categories that
tend to appear in cluttered scenes, that signal never comes out clean enough
for SNMF to isolate a coherent direction, no matter how low the acceptance
threshold goes.

## Suggested fix (not yet implemented)

Isolate the target object more aggressively than padding-only cropping --
e.g. mask out *other* annotated bboxes that fall inside a crop's padded
region (blur/blackout, using AU-AIR's own per-frame annotations for every
other object), similar in spirit to how `coco10`'s `val_masked/` uses
ground-truth segmentation masks, just adapted for bbox-only annotations.
Not attempted yet -- flagging as the most promising next step, not a
completed fix.
