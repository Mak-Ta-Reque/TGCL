# License Compliance Report (project license: Apache-2.0)

Generated via the `license-compliance-check` skill (`.claude/skills/license-compliance-check/`),
covering the Schritt 1 (Erfassen) inventory across all four artefact
classes the framework distinguishes: **code packages**, **model weights**,
**datasets**, and **GitHub-sourced code** (git-installed, not on PyPI —
these need separate handling since `licensecheck` can't resolve them and
their PyPI metadata, when present at all, is frequently wrong/missing).

**Environment scanned**: `xlvlms-verify` — a from-scratch conda env built
strictly from `docs/cgdl_quickstart.md` §1's installation guide, i.e. only
what the *current, trimmed* codebase (`scripts/run_full_pipeline.py` +
`scripts/run_ablation.py`/`ablation_report.py` and their dependencies)
actually needs. This supersedes the previous scan, which was run against
an older environment that still carried demo/API/legacy-script cruft
since removed from the repo (156 packages now, vs. 203 before — the
diff is entirely dev/demo tooling no longer part of any code path, not a
change in what's actually shipped).

---

## A. Code packages (automated `licensecheck` scan)

156 packages, all previously classified against the same rule set
(re-verified: every package here is confirmed still installed in
`xlvlms-verify`; nothing new appeared vs. the prior scan — the trimmed
env is a strict subset). Full table:
[`license_report_packages.csv`](license_report_packages.csv) (was
`license_report.json` before this update — see note at the end).

**Summary**: 132 Fall 1 (permissive) · 4 Fall 2 (weakly protective) · 20
flagged Fall 3, of which **2 are classifier false positives** (confirmed
by reading `pip`'s own `License-Expression` metadata field directly,
which `licensecheck` doesn't parse):

- **False positive**: `typing-extensions` (PSF-2.0) — the script's
  substring matcher only catches the spelled-out "python software
  foundation" string, not the abbreviation "PSF-2.0". Genuinely Fall 1.
  (The other previously-flagged false positive, `audioop-lts`, is no
  longer in the environment at all — it was an unused transitive
  dependency.)
- **False positive**: `fsspec` — `pip show fsspec` reports
  `License-Expression: BSD-3-Clause` directly; `licensecheck` doesn't read
  that (newer, PEP 639) metadata field and falls back to "Unknown".
  Genuinely Fall 1. Mandatory (required by `huggingface_hub` for
  model download/caching).
- **13 NVIDIA CUDA runtime packages + `cuda-toolkit` + `cuda-bindings`**
  (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc.) — proprietary NVIDIA
  EULA (`cuda-bindings`'s own metadata confirms
  `License-Expression: LicenseRef-NVIDIA-SOFTWARE-LICENSE`). **All
  mandatory** — this is `torch`'s own CUDA runtime stack, pulled in
  automatically by `pip install torch --index-url .../cu126`; standard
  and unavoidable for any GPU-accelerated PyTorch project, not unique to
  this repo. Accepting this EULA is the cost of GPU acceleration itself.
- **`lang-sam`** — resolved in §D below (it's git-installed; PyPI has no
  metadata for it at all, hence "Unknown" here — verified Apache-2.0 at
  the actual source). **Only mandatory if `CROP_MODE=langsam`** — the
  example default in `docs/cgdl_quickstart.md` §2, but `CROP_MODE=none`
  or `sliding_window` (also documented, fully valid options) don't need
  it at all.

Fall 2 (weakly protective, Schritt 3 modular integration if kept):
`certifi`, `orjson`, `pyzmq`, `tqdm` — all dual/multi-licensed with
MPL-2.0/LGPL as one option among others.

---

## B. Model weights

Not covered by the package scan at all — the framework calls this out
explicitly (Known Gap #1/#3: model weights are their own artefact class,
and field-of-use restrictions are often shipped as a *separate* document
from the license text a scanner would read).

| Model | License | Fall | Notes |
|---|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` | **Qwen RESEARCH LICENSE AGREEMENT** (`qwen-research`) | **3** | Non-commercial only — see full escalation: [`escalation-qwen2.5-vl-3b.md`](escalation-qwen2.5-vl-3b.md) |
| `Qwen/Qwen2.5-VL-7B-Instruct` | **Apache-2.0** | 1 | Same-family drop-in alternative to the 3B model — recommended fix, see escalation doc |
| `Qwen/Qwen2.5-VL-72B-Instruct` | Custom "qwen" license (`license: other`) | 3 (unverified detail) | Different from `qwen-research`; not investigated further since the 7B Apache-2.0 option already resolves the need |
| `google/gemma-3n-E4B-it` | Custom "Gemma Terms of Use" (not OSI) | **Needs manual read** | Prohibited Use Policy checked — no explicit military exclusion found. Commercial-use grant wasn't unambiguous from an automated read of the terms; do not treat as cleared until a human reads the full agreement |

---

## C. Datasets

| Dataset | License | Fall | Notes |
|---|---|---|---|
| COCO 2017 (train2017/val2017 — source for `data/coco10/`) | **Mixed rights**: annotations under Creative Commons Attribution 4.0; the COCO Consortium does not own the underlying images, which remain subject to their original Flickr copyright holders' terms | **Unresolved — matches Known Gap #7** | This is exactly the "mixed-rights artefact" scenario SKILL.md flags as not folded into the Fall 1/2/3 tree. Direct verification of the exact Terms of Use wording at cocodataset.org failed (JS-rendered page, not fetchable) — the annotation license (CC BY 4.0) is well-established public fact, but the image-rights caveat should be independently confirmed before any external use/redistribution of `data/coco10/` beyond internal research. Not escalated as blocking since this project only *derives crops/masks for internal model probing*, not redistributing the source images — but flag to Tobias if `data/coco10/` or its derivatives are ever shared externally. |

No other datasets are used by the kept `cgdl` + ablation pipeline
(rebuttal-ablation-specific dataset variants were removed along with
those scripts).

---

## D. GitHub source code usage (git-installed, not on PyPI)

These are installed via `pip install git+https://...` per
`docs/cgdl_quickstart.md` §1 — `licensecheck`/PyPI metadata is unreliable
or entirely absent for these (confirmed: `lang-sam` shows "UNKNOWN" in §A
purely because PyPI has no record of it), so each was checked directly
against its GitHub repo's actual `LICENSE` file.

| Package | Repo | License (verified at source) | Fall | Used for |
|---|---|---|---|---|
| `clip` | `openai/CLIP` | MIT | 1 | `eval/clip_bert_score_eval.py`, CLIP-scored crop ranking in `preprocessing/crops_to_json.py` |
| `lang-sam` | `luca-medeiros/lang-segment-anything` | **Apache-2.0** | 1 | `CROP_MODE=langsam` detector, via `src/langsam_utils.py` |
| `sam-2` | `facebookresearch/sam2` (installed transitively by `lang-sam`) | Apache-2.0 (code) | 1 | Segmentation backend inside `lang-sam` — code license only; checkpoint weights auto-downloaded at runtime aren't separately verified here |
| GroundingDINO | `IDEA-Research/GroundingDINO` (installed transitively by `lang-sam`) | Apache-2.0 | 1 | Text-conditioned detection backend inside `lang-sam` |
| `language_evaluation` | `bckim92/language-evaluation` | MIT (in `LICENSE.md`, verified — an earlier check of a guessed `LICENSE` filename 404'd) | 1 | **Currently not on the pipeline's eager import path** — installed per the old README's instructions but `src/metrics/captioning_metrics.py` (the only kept-package consumer of `nltk`/metrics machinery) doesn't actually call into it. Keeping it listed for completeness since it's still installed; safe to drop from the install guide if confirmed unused. |

All four GitHub-sourced packages actually exercised by the documented
pipeline (`clip`, `lang-sam`, and its two transitive backends) are
**Apache-2.0 or MIT** — no Fall 2/3 findings in this category.

---

## Escalations open

1. **`Qwen/Qwen2.5-VL-3B-Instruct`** — Fall 3, non-commercial license.
   See [`escalation-qwen2.5-vl-3b.md`](escalation-qwen2.5-vl-3b.md).
2. **`google/gemma-3n-E4B-it`** — not blocked, but not yet confirmed
   clear either; needs a human read of the full Gemma Terms of Use
   before being treated as a compliant fallback for the item above.
3. **COCO image provenance** (§C) — flag if `data/coco10/` or derivatives
   are ever shared/redistributed outside this project; not blocking for
   internal-only use.

## Note on this update

The original automated scan produced a single flat report covering only
§A (package scan), against a broader environment that still had
demo/API/legacy-script dependencies since removed from the codebase. This
revision: (1) re-scoped §A to the 156 packages actually present in the
`xlvlms-verify` environment built from the current install guide —
raw data in [`license_report.json`](license_report.json) /
[`license_report_packages.csv`](license_report_packages.csv); (2) added
§B (model weights), §C (datasets), and §D (GitHub source code usage) as
new sections, none of which the automated scan can cover.
