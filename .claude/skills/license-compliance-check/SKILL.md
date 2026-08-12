---
name: license-compliance-check
description: Use whenever the user asks Claude to check, evaluate, clear, or document the license of code, a dataset, an ML/VLM model, or any third-party artefact for this project — e.g. "is this license OK", "can we use this model/dataset", "check the license", "does this need Tobias", "update the SBOM", or when a new model/dataset candidate is introduced during model selection or fine-tuning work. Encodes the project's Schritt 1-5 / Fall 1-3 license-compliance framework (Cyberagentur) plus the RSML/Confluence "Further Ideas" extension, so every check is done consistently. Also use to flag when a candidate falls into a gap the framework doesn't clearly cover (model weights, fine-tuning inheritance, hidden AUPs, dataset-only alternatives) — see "Known Gaps" below.
---

# License Compliance Check

Codifies two source documents into one actionable procedure:
- **"Step-by-Step License Check"** (Cyberagentur, internal, German) → Schritt 1–5, Fall 1–3
- **"Further Ideas" (RSML Confluence)** → Step 1–7, extends the same framework with more detail on datasets/models, compatibility, and customer delivery

**Non-negotiable requirement for this project:** every artefact (code, dataset, model) must permit **both military and commercial use**, with no field-of-use exclusion. This is checked in *every* step below, not just once.

## Quick decision tree

```
1. ERFASSEN — identify every license touching the artefact (not just the headline one)
2. ANALYSIEREN — classify each license found
   ├─ Fall 1: ALL permissive (MIT, BSD, Apache-2.0, public domain)
   │          → usable, satisfy attribution/NOTICE obligations, done
   ├─ Fall 2: Weakly protective present (LGPL-2.1/3, MPL-2.0)
   │          ├─ Option A: permissive alternative exists → switch to it
   │          └─ Option B: no alternative → usable ONLY with modular/
   │             dynamic integration (Schritt 3) — never static-link/merge
   └─ Fall 3: Strongly/Network protective present (GPL, AGPL) OR
              unclear/non-standard/no license OR
              military/commercial use excluded by any clause
              → DO NOT USE. Escalate to Tobias proactively (see below).
3. Check field-of-use clause REGARDLESS of category above.
   Any exclusion of military or commercial use = automatic Fall 3 treatment,
   even if the base license is Apache/MIT (e.g. Meta's Acceptable Use Policy
   layered on top of an otherwise-permissive license text).
4. STRUKTUR — if Fall 2/Option B: separate directories, own LICENSE file per
   module, NOTICE file listing all licenses, dynamic import only.
5. DOKUMENTIEREN — update SBOM (component, version, license, source URL,
   hash), attach license texts, note any modifications made.
```

## Step-by-step procedure

### Step 1 — Erfassen / Capture
- Identify **every** artefact: source code, binaries, container images, scripts, docs, datasets, model weights, annotations.
- Find the license via README/model card/dataset card first; cross-check with a tool (e.g. Licensecheck) — do not rely on either source alone.
- **For a full dependency scan, use the bundled `scripts/generate_license_report.py`.** It wraps the `licensecheck` CLI (the tool named in Schritt 1) and maps every dependency onto this framework's Fall 1/2/3 categories in one pass — see "Automated dependency report" below.
- **Always resolve to the authoritative/original source, never a mirror or aggregator.** Mirrors (e.g. Roboflow re-hosts of academic datasets) have been found to mislabel licenses (VisDrone shown as CC BY 4.0 on Roboflow vs. the correct CC BY-NC-SA 3.0 from the original authors). Treat any license claim from a non-original host as unverified until confirmed at the source.
- Note "AND"/"OR" concatenations (dual/multi-licensing) — resolve to the most favorable *available* option, not the most restrictive.
- Record manual additions: internal forks, copy-pasted snippets, in-house code.
- Differentiate the artefact type explicitly: **software vs. dataset vs. model weights vs. annotations-only** — these can carry different licenses even when bundled together (e.g. annotations under a free license, underlying images still proprietary/Flickr-owned).
- Log everything toward a complete SBOM (component, version, license, source, hash).

### Step 2 — Analysieren / Classify
Use the category table below (from both documents) to classify each license found:

| Category | Examples | Can we use it? |
|---|---|---|
| Permissive | MIT, BSD-2, Apache-2.0, public domain | Yes — Fall 1 |
| Weakly Protective | LGPL-2.1/2.1+/3, MPL-2.0 | Conditionally — Fall 2 |
| Strongly Protective | GPL-2.0, GPL-3.0 | No (default) — Fall 3 |
| Network Protective | AGPL-3.0 | No (default) — Fall 3 |
| Non-OSI / Proprietary | SSPL, BSL-1.1, custom RAIL/OpenRAIL | No (default) — Fall 3 |

For **every** category, independently verify: *is military use excluded? is commercial use excluded?* A permissive base license does not guarantee this — check for a bolted-on Acceptable Use Policy, addendum, or "Responsible AI" clause (this is exactly how LLaVA/Llama-based models fail the framework despite an open-looking license).

- **Fall 1** (all permissive, no field-of-use exclusion): usable. Satisfy attribution/NOTICE/change-documentation duties. Done.
- **Fall 2** (weakly protective present):
  - *Option A:* Is there a permissive equivalent that fits the same purpose? If yes, prefer it.
  - *Option B:* No equivalent exists → usable, but **must** be dynamically/modularly integrated (Step 3). This is the same logic that should be applied when the "alternative" is a *dataset*, not just code — see Known Gaps below for how this project is currently extending it.
- **Fall 3** (strongly/network protective, non-OSI, unclear, or field-of-use excluded): avoid if at all possible. If unavoidable → **proactively escalate to Tobias** before proceeding.

### Step 3 — Sicherstellen der Lizenztrennung / Structural separation
Only relevant for Fall 2/Option B artefacts:
- Keep main code and weakly-protective modules in separate directories.
- One LICENSE file per license in use (e.g. `LICENSE_APACHE2.0`, `LICENSE_LGPLv3`), plus a top-level `NOTICE` file listing all licenses and copyrights.
- Integrate via **dynamic** import (`import`, `importlib.import_module()`), never static linking — dynamic linking acts as a "license firewall" that keeps obligations scoped to the module itself.
- Optional but recommended: maintain an SBOM (`sbom/`, SPDX or CycloneDX format).

### Step 4 — Compatibility handling
- **4a, if incompatible:** identify the conflict, then try (in order of preference): find a compatible substitute artefact → split into separate processes/microservices (aggregation ≠ license merging) → dual-licensing request to the rights holder → commercial license purchase (needs RSML PM approval) → if nothing works, halt use and escalate to Tobias.
- **4b, if compatible:** permissive-with-permissive is generally fine. When combining two license categories, the combined obligation is that of the *more restrictive* one. Document via NOTICE file + build-pipeline license bundling.
- Known compatibility pitfalls: Apache-2.0 code cannot be pulled into a GPL-2.0 project's outbound release without triggering GPL terms (direction matters); MPL-2.0 is explicitly designed to combine with GPL/LGPL/Apache-2.0.

### Step 5 — Documentation & final release
- Full re-scan before release.
- Manual dual-control check (second person verifies the license classification).
- Ship: license texts, NOTICE file, SBOM (signed, SPDX/CycloneDX), release tag + commit hash + scanner version.

## Automated dependency report

`scripts/generate_license_report.py` wraps the `licensecheck` CLI and produces the "list of used licenses per package incl. compatible/not-compatible flag" that Schritt 1 asks for, already sorted onto Fall 1/2/3:

```bash
pip install licensecheck --break-system-packages
python scripts/generate_license_report.py \
  -r requirements.txt -l Apache-2.0 \
  -o license_report.md --json-out license_report.json
```

- `-r` — one or more requirements.txt / pyproject.toml files to scan (default: `pyproject.toml` in cwd)
- `-l` — this project's own license, default `Apache-2.0` (the framework's permissive-by-default goal)
- Output: a Markdown table (package, version, license, framework category, Fall, `licensecheck` compat flag, field-of-use flag, required action) plus a summary count and a ready-to-use "Escalate to Tobias" list for every Fall 3 package.
- Adds a check `licensecheck` itself can't do: a curated, **extensible** list of packages/models known to carry a field-of-use restriction outside their base license text (`KNOWN_FIELD_OF_USE_RISK` in the script — currently seeded with the Llama/LLaVA AUP case and the non-commercial drone datasets from this project). Add to that dict whenever a new hidden-AUP case is found, per Known Gap #3.
- This only covers *packages resolvable via pip/PyPI metadata*. It does **not** replace the manual checks for datasets, model weights, mirrors-vs-original-source, or annotation/imagery splits — those still need the manual Step 1-2 review above.
- Anything the script marks Fall 3 (including "Unknown" license strings) still needs the same manual verification and Tobias escalation as a manually-found Fall 3 case — treat the report as a first pass, not a final answer.

## Escalation to Tobias
Escalate **proactively** (don't wait to be asked) whenever:
- An artefact lands in Fall 3 and no substitute/mitigation from Step 4a resolves it.
- Modular/dynamic separation (Step 3) isn't achievable for a Fall 2/Option B case.
- A license is unclear, missing, or conflicts between two stated sources (e.g. README says one thing, LICENSE file says another).
- **This project's live open case:** no compliant training dataset alternative exists for drone imagery (VisDrone/DOTA/xView all fail on the non-commercial clause). This is a Fall 2/Option A situation with no available Option A alternative at sufficient scale — flag as such when escalating, rather than defaulting to Fall 3 language, since it's a data-availability gap, not a rejected license.

When escalating, prepare: the artefact name/version, the exact license text or clause in question, why Step 2 classification is ambiguous or blocked, and what alternatives were already checked.

## Known Gaps — not clearly covered by the current framework
These came up doing real model/dataset evaluation on this project and are worth raising with Tobias/the framework owner for an explicit ruling, since the current documents don't give a definitive answer:

1. **Model weights as their own artefact class.** The framework says to "differentiate" model licenses from code licenses but doesn't say how the *check itself* should differ — e.g. whether a model card's license field is authoritative over a repo-level LICENSE file, or how to handle hybrid licenses like OpenRAIL/CreativeML that aren't in the Permissive/Weakly/Strongly/Non-OSI table at all.
2. **Fine-tuning inheritance.** If a permissively-licensed base model (e.g. Apache-2.0 Qwen2.5-VL) is fine-tuned on a dataset with a restrictive license, does that restriction attach to the resulting weights? Not addressed by either document, but directly relevant to any fine-tuning step in this project.
3. **Hidden Acceptable Use Policies.** Both documents say to "look out for" military/commercial exclusions, but neither flags that some vendors ship the restriction as a *separate document* (Meta's AUP) rather than inside the license text a scanner like Licensecheck would pick up. Worth adding as an explicit sub-step in Schritt 1/Step 1, not just a general warning.
4. **Mirror/aggregator provenance.** Confirmed pitfall (Roboflow mislabeling VisDrone) but not written into either document as a required verification step — recommend formalizing "verify at original source" as mandatory in Schritt 1.
5. **No compliant alternative for datasets specifically.** Step 2/Fall 2/Option A is written with code substitution in mind; it doesn't explicitly say what to do when the "alternative" search comes up empty for a *dataset* at the scale needed (this project's live blocker). Needs an explicit ruling on whether proprietary/self-collected data or rights-holder outreach is the expected path, and who approves it.
6. **Export control / dual-use regulation.** Neither document addresses non-license legal constraints (e.g. EU dual-use export controls) that may apply on top of licensing for military-relevant AI/drone work — currently out of scope for both documents entirely.
7. **Mixed-rights artefacts.** "Annotation-only" and "code-only" license splits are mentioned once (Step 2, Further Ideas) but aren't folded into the Fall 1/2/3 decision tree — currently unclear whether a dataset with free annotations but proprietary imagery should be treated as Fall 1 (annotations) with a carve-out, or Fall 3 overall.
8. **Escalation record-keeping.** "Reach out to Tobias proactively" is repeated several times but there's no defined format for what to bring him or how the decision gets logged for future SBOM/audit purposes.

Use this list as a checklist during escalation write-ups until the framework owner rules on each point explicitly.
