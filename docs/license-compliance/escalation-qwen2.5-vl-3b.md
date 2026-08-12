# Escalation: `Qwen/Qwen2.5-VL-3B-Instruct` — Fall 3, non-commercial license

Prepared per the license-compliance-check skill's escalation format
(artefact + version, exact clause, why blocked, alternatives checked).
Escalating proactively per that skill's rule — not waiting to be asked.

## Artefact

- **Name / version**: `Qwen/Qwen2.5-VL-3B-Instruct` (Hugging Face model
  repo, current `main` revision as of 2026-08-11)
- **Where it's used in this project**: named as the default/example
  `VLM_MODEL` throughout `docs/cgdl_quickstart.md`, `.env.example`, and
  most of `scripts/`/`inference/` — not a peripheral dependency, it's the
  model the documented pipeline is written around (alongside
  `google/gemma-3n-E4B-it`, which is separately under review — see Known
  Gaps note below).

## Exact clause

License name (confirmed two independent ways — the repo's own `LICENSE`
file text, and Hugging Face's `cardData.license_name` API field, which
both agree): **"Qwen RESEARCH LICENSE AGREEMENT"** (`license_name:
qwen-research`).

> "You are granted a non-exclusive, worldwide, non-transferable and
> royalty-free limited license... to use, reproduce, distribute, copy,
> create derivative works of, and make modifications to the Materials
> **FOR NON-COMMERCIAL PURPOSES ONLY**."
>
> "If you are commercially using the Materials, you shall request a
> license from us."

Also requires "Built with Qwen" / "Improved using Qwen" attribution if
used to train or improve a distributed model. No explicit military-use
clause found (the restriction is commercial-use, not military-use), but
under the framework's non-negotiable requirement — **both** military and
commercial use must be permitted — a non-commercial-only grant is
automatic Fall 3 regardless of the military question, per Step 2 / Schritt
2 "military/commercial use excluded by any clause → Fall 3."

## Why this is blocked, not just flagged

This isn't a Fall 2 "weakly protective, use it modularly" case — the
license affirmatively prohibits commercial use outright, with no
structural-separation (Step 3) workaround available, since Step 3 only
addresses copyleft obligation scope, not a field-of-use prohibition.

## Alternatives already checked

- **`Qwen/Qwen2.5-VL-7B-Instruct`** — same model family/interface, larger
  size. Confirmed via Hugging Face API metadata: **`license: apache-2.0`**.
  This is a same-family, drop-in-compatible Option A substitute — no
  pipeline code changes needed beyond `VLM_MODEL`/`BATCH_SIZE` tuning for
  the larger model's memory footprint. **Recommended default fix** unless
  there's a hardware/latency reason the project specifically needs the 3B
  size.
- **`Qwen/Qwen2.5-VL-72B-Instruct`** — checked, also **not** Apache-2.0
  (`license_name: qwen`, a different custom Qwen license than the 3B's
  `qwen-research`). Not verified further since the 7B Apache-2.0 option
  already resolves this without needing the 72B's much larger footprint.
- **LLaVA-family models** — not checked in detail; already excluded a
  priori per this project's `KNOWN_FIELD_OF_USE_RISK` list (inherits
  Meta's Llama Acceptable Use Policy military exclusion).
- **`google/gemma-3n-E4B-it`** (the project's other documented default) —
  separately reviewed, no explicit military exclusion found in its
  Prohibited Use Policy, but commercial-use permission wasn't unambiguous
  from an automated read of the terms — flagged for a human read, not yet
  cleared as a confirmed-compliant fallback. See
  `docs/license-compliance/license_report.md` summary notes.

## Recommendation

Switch the documented/default `VLM_MODEL` from `Qwen/Qwen2.5-VL-3B-Instruct`
to `Qwen/Qwen2.5-VL-7B-Instruct` (Apache-2.0) across `.env.example`,
`docs/cgdl_quickstart.md`, and any script defaults, pending Tobias's
sign-off. Not made automatically as part of this check — this is a
project-direction decision, not just a doc fix.
