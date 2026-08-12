# `build_auair_dataset.py --min-crop-size` isn't always honored

**Status: open**

## Summary

`preprocessing/build_auair_dataset.py`'s `--min-crop-size` (default 160px)
is meant to guarantee every saved crop's shorter side is at least that many
pixels, expanding padding as needed. For bboxes near a frame edge, it
doesn't: the resulting crop can end up smaller than the configured floor.

## Evidence

`compute_crop_box`'s `expand_to_min()` (`preprocessing/build_auair_dataset.py:170-175`):

```python
def expand_to_min(a1: float, a2: float, min_size: int, limit: int) -> Tuple[float, float]:
    deficit = min_size - (a2 - a1)
    if deficit > 0:
        a1 -= deficit / 2
        a2 += deficit / 2
    return max(0.0, a1), min(float(limit), a2)
```

It splits the needed padding evenly across both sides (`deficit / 2` each),
then clamps each side independently to `[0, limit]` -- but never gives the
clamped-away padding back to the *other* side. A bbox near `x=0` or
`y=frame_height` gets its deficit only partially applied, so the final crop
can still be smaller than `min_size`.

Measured directly (`scripts/check_auair_crop_detectability.py`'s crop-size
stats, run against the real dataset): with `--min-crop-size 160`, several
categories' saved *train* crops had a minimum shorter side as low as
**102-116px** -- well under the configured floor.

## Suggested fix

In `expand_to_min`, after clamping, compute how much padding was actually
lost to the clamp and re-apply it to the opposite side (if that side still
has room), rather than treating each side independently.

## Note

This is *not* the main driver of the pipeline's low object-detectability --
a separate check (feeding sampled crops through the actual VLM) found
accuracy was roughly flat across crop-size buckets, including the smallest
one. This bug is real and worth fixing, but crop size is not the primary
explanation for downstream accuracy problems (see
`snmf-cleanliness-collapse-cluttered-crops.md` for the bigger driver).
