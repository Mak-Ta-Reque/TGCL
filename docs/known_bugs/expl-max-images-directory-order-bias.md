# `EXPL_MAX_IMAGES=N` (N < full val set) evaluates a directory-order slice, not a sample

**Status: open**

## Summary

`EXPL_MAX_IMAGES` / `--max_images` is documented as "first N images," which
is technically true but misleading: images are collected via `os.walk` and
then sorted alphabetically by full path (`inference/vlm_explainer_multibatch.py:823`),
so for a per-class directory layout (`data/auair/val/<class>/...`), "first N"
means *the first N images of the alphabetically-first class(es)* -- not a
representative sample across classes.

## Evidence

`inference/vlm_explainer_multibatch.py:812-823`:
```python
for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        ...
        collected.append(os.path.join(dirpath, fn))
collected.sort()
```
Then the cap (`inference/vlm_explainer_multibatch.py:841`):
```python
args.image = args.image[: args.max_images]
```
A plain slice of the sorted list -- no stratification by subfolder.

## Concretely hit this session

With AU-AIR's 8 classes (`human, car, truck, van, motorbike, bicycle, bus,
trailer`) and `EXPL_MAX_IMAGES=10`, every evaluated image came from
`bicycle/` (the alphabetically-first folder). Running
`scripts/detect_object_topk_f1.py` against that `vlm_explanations.json`
reported **top1_accuracy = 0.000** -- not because the pipeline is entirely
broken, but because `bicycle` happened to be one of the categories with no
usable concept direction at the time, and no other category was even
evaluated. The same config with `EXPL_MAX_IMAGES=-1` (all 383 val images)
later showed `top1_accuracy = 0.287`, `top1_f1_macro = 0.159` -- a very
different picture.

## Suggested fix

Either stratify the sample (take up to `N / num_classes` per subfolder) when
`--image_root` is a multi-subfolder layout, or at minimum shuffle with a
fixed seed before slicing so a small N is a random sample instead of a
directory-order prefix.

## Workaround today

Set `EXPL_MAX_IMAGES=-1` (or a value >= your full val set) for any run
you intend to compute accuracy/F1 from. Only use a small N for a quick
smoke test of the mechanics, never to estimate real performance.
