# Known bugs

Bugs and limitations found while auditing the AU-AIR/`tgcl` pipeline against
`README.md`'s documented setup/run instructions, plus earlier findings from
the same investigation. Each file has concrete evidence (file:line,
reproduction steps, actual output) rather than a guess.

## Open

| Bug | Impact |
|---|---|
| [`stale-step-skip-checks.md`](stale-step-skip-checks.md) | Changing config and re-running the pipeline against the same `OUTPUT_DIR` can silently reuse stale outputs instead of regenerating -- affects steps 1, 2, 6 (confirmed), likely others. |
| [`auair-min-crop-size-not-enforced.md`](auair-min-crop-size-not-enforced.md) | `build_auair_dataset.py --min-crop-size` isn't always honored near frame edges. |
| [`expl-max-images-directory-order-bias.md`](expl-max-images-directory-order-bias.md) | `EXPL_MAX_IMAGES=N` (N less than the full val set) always evaluates the same alphabetically-first class(es), not a representative sample. |
| [`snmf-cleanliness-collapse-cluttered-crops.md`](snmf-cleanliness-collapse-cluttered-crops.md) | Some AU-AIR categories (`bicycle`, `human`) never produce a usable SNMF concept direction even with all other fixes applied -- a crop-isolation limitation, not a simple bug. |

## Fixed

See [`fixed-bugs.md`](fixed-bugs.md) for bugs found and already merged into `main`.
