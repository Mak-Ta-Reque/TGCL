# `scripts/run_full_pipeline.py` skip-checks are existence-only, not config-aware

**Status: open**

## Summary

Every pipeline step "resumes" by checking whether its own output file
already exists -- never whether the *config that produced it* still matches
the current run. Change any env var (`IMAGE_ROOT`, `CROP_MODE`, `PROMPT`,
`CONCEPTS_VOCAB`, ...) and re-run against the same `OUTPUT_DIR`, and the
step whose output already exists just logs `Skip <step> (found ...)` and
reuses the old file untouched -- no warning that the new config was never
applied.

## Evidence

Confirmed at three separate steps by reading the code directly:

- Step 1 (`scripts/run_full_pipeline.py:548-556`):
  ```python
  csv_exists = config.objects_csv.exists()
  concept_map_exists = config.concept_map_json.exists()
  if csv_exists and concept_map_exists:
      logger.info(f"Skip Step 1 (found {config.objects_csv} and {config.concept_map_json})")
      ...
      return
  ```
- Step 2 (`scripts/run_full_pipeline.py:703`): `if config.crops_json.exists(): ... logger.info("Skip Step 2 ..."); return`
- Step 6 / explainer (`scripts/run_full_pipeline.py:1066-1069`):
  ```python
  out_json = out_dir / "vlm_explanations.json"
  if out_json.exists():
      logger.info(f"Skip Explainer ({method}) (found {out_json})")
      continue
  ```

None of these compare a hash/fingerprint of the relevant config (`IMAGE_ROOT`,
`PROMPT`, `CROP_MODE`, etc.) against what was used to produce the existing
file -- only `Path.exists()`.

## Concretely hit this session

1. **`README.md`'s own §3 instructions are broken by this.** It tells you to
   get the object-classification F1 metric working by "rerun step 2's
   `EXPL_*` config with `IMAGE_ROOT=data/auair/val`" against the same
   `OUTPUT_DIR` used for the main run. Since `vlm_explanations.json` already
   exists (from the `val_grids` run), step 6 just skips -- the new
   `IMAGE_ROOT` is silently ignored and you get the exact same (grid-based)
   file back. (README now carries a callout about this next to that
   instruction.)
2. **The `bu`/`bus` inflect bug (see `fixed-bugs.md`) looked "not fixed"
   after merging the actual fix**, because
   `outputs/auair_run/inference/concepts_to_images_auair_vocab_top8.json`
   already existed from a pre-fix run and `select_top_concepts.py` returns
   an existing file of the same computed name without regenerating it (a
   related but separate caching mechanism -- see
   `preprocessing/select_top_concepts.py:101-104`, `if filtered_path.exists():
   return filtered_path`, where the filename itself
   (`..._top{len(filtered_mapping)}.json`) is derived from *how many concepts
   survive the vocab filter*, not from a hash of the inputs). Worked around
   by manually deleting the stale file before re-running; the underlying
   design (cache key = output row count, not input hash) is still there.

## Suggested fix

Hash the relevant subset of `PipelineConfig` per step (or at minimum the
specific env vars that step reads) and store it alongside each output file;
skip only when both the output exists *and* the stored hash matches the
current config. Short of that, at least log a warning when skipping a step
whose config-relevant env vars differ from a previous recorded run.

## Workaround today

Before re-running any step with a changed config, delete that step's output
(or use a fresh `OUTPUT_DIR`) rather than trusting the resume logic to
notice the change.
