#!/usr/bin/env python3
"""
generate_license_report.py

Wraps the `licensecheck` CLI (the tool named in Schritt 1 of the
Cyberagentur "Step-by-Step License Check") and maps its output onto this
project's Schritt 2 / Fall 1-3 framework, adding the two checks the CLI
tool cannot do on its own:

  1. Field-of-use (military/commercial) risk flagging via a curated list
     of known Acceptable-Use-Policy / field-of-use-restricted packages,
     since AUPs live outside the license text and are invisible to any
     automated scanner (see Known Gap #3 in SKILL.md).
  2. Weakly-protective packages are explicitly called out as needing
     Schritt 3 (modular/dynamic integration), not just "compatible".

Requires: pip install licensecheck --break-system-packages

Usage:
  python generate_license_report.py -r requirements.txt -l Apache-2.0 \
      -o report.md --json-out report.json

  python generate_license_report.py            # defaults to pyproject.toml
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# --- Framework category mapping (Category table from both source docs) ---
# Matched case-insensitively against substrings of the license string
# licensecheck returns (which is often a concatenation, e.g.
# "Apache-2.0;; BSD-3-Clause").
CATEGORY_RULES = [
    # Order matters: "lgpl" and "agpl" both contain "gpl" as a substring,
    # so they must be matched BEFORE the plain-GPL rule or they'd be
    # misclassified as Strongly Protective.
    ("Network Protective", ["agpl"]),
    ("Weakly Protective", ["lgpl", "mpl", "mozilla public license", "eclipse public license", "epl"]),
    ("Strongly Protective", ["gpl-2", "gpl-3", "gplv2", "gplv3", "gnu general public license", "gpl v"]),
    ("Non-OSI / Proprietary", ["sspl", "bsl-1.1", "business source", "commons clause",
                               "creativeml", "openrail", "rail license", "responsible ai"]),
    ("Permissive", ["mit", "bsd", "apache", "isc", "public domain", "unlicense", "python software foundation", "zlib"]),
]

FALL_BY_CATEGORY = {
    "Permissive": 1,
    "Weakly Protective": 2,
    "Strongly Protective": 3,
    "Network Protective": 3,
    "Non-OSI / Proprietary": 3,
    "Unknown": 3,
}

ACTION_BY_FALL = {
    1: "OK - satisfy attribution/NOTICE duties",
    2: "Check for a permissive alternative (Option A). If none, dynamic/"
       "modular integration only (Schritt 3) - own directory + own LICENSE file.",
    3: "Do not use. Escalate to Tobias proactively.",
}

# Curated, extensible list of packages/models/orgs known to carry a
# field-of-use restriction (AUP, non-commercial or non-military clause)
# OUTSIDE their base license text. Add to this list whenever a new one
# is discovered during a check - this is exactly the Roboflow/Llama-AUP
# style pitfall the framework doesn't catch automatically.
KNOWN_FIELD_OF_USE_RISK = {
    "llama": "Meta Llama Acceptable Use Policy excludes military use",
    "meta-llama": "Meta Llama Acceptable Use Policy excludes military use",
    "llava": "Built on Llama - inherits Meta AUP military exclusion",
    "visdrone": "CC BY-NC-SA - non-commercial clause",
    "dota": "Non-commercial research-only terms in original release",
    "xview": "Non-commercial research-only terms in original release",
}


def classify(license_str: str) -> str:
    if not license_str or license_str.strip().upper() in {"UNKNOWN", ""}:
        return "Unknown"
    low = license_str.lower()
    for category, needles in CATEGORY_RULES:
        if any(n in low for n in needles):
            return category
    return "Unknown"


def field_of_use_flag(pkg_name: str) -> str:
    low = pkg_name.lower()
    for needle, note in KNOWN_FIELD_OF_USE_RISK.items():
        if needle in low:
            return note
    return ""


def run_licensecheck(requirements_paths, project_license, extra_args):
    cmd = ["licensecheck", "-f", "json", "-l", project_license]
    if requirements_paths:
        cmd += ["-r", *requirements_paths]
    cmd += extra_args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):  # 1 can just mean "incompatible found"
        sys.stderr.write(proc.stderr)
        sys.exit(f"licensecheck failed (exit {proc.returncode})")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr)
        sys.exit("Could not parse licensecheck JSON output")


def build_report(data):
    rows = []
    for pkg in data.get("packages", []):
        name = pkg.get("name", "")
        license_str = pkg.get("license", "") or "UNKNOWN"
        category = classify(license_str)
        fall = FALL_BY_CATEGORY[category]
        fou_flag = field_of_use_flag(name)
        if fou_flag:
            fall = 3  # field-of-use risk always forces Fall 3 regardless of base license
        rows.append({
            "name": name,
            "version": pkg.get("version", ""),
            "license": license_str,
            "category": category,
            "licensecheck_compat": pkg.get("licenseCompat"),
            "field_of_use_flag": fou_flag,
            "fall": fall,
            "action": ACTION_BY_FALL[fall],
        })
    rows.sort(key=lambda r: (-r["fall"], r["name"].lower()))
    return rows


def to_markdown(rows, project_license):
    lines = [
        f"# License Report (project license: {project_license})",
        "",
        "Generated via `licensecheck`, mapped onto the Schritt 2 / Fall 1-3 "
        "framework. Field-of-use flags are from a curated list and are NOT "
        "exhaustive - always verify at the original source for anything "
        "flagged Fall 3 or Unknown.",
        "",
        "| Package | Version | License | Category | Fall | licensecheck compat | Field-of-use flag | Action |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['version']} | {r['license']} | {r['category']} "
            f"| {r['fall']} | {r['licensecheck_compat']} | {r['field_of_use_flag']} | {r['action']} |"
        )

    n = len(rows)
    fall1 = sum(1 for r in rows if r["fall"] == 1)
    fall2 = sum(1 for r in rows if r["fall"] == 2)
    fall3 = sum(1 for r in rows if r["fall"] == 3)
    lines += [
        "",
        "## Summary",
        f"- Total packages: {n}",
        f"- Fall 1 (permissive, OK): {fall1}",
        f"- Fall 2 (weakly protective, needs Schritt 3): {fall2}",
        f"- Fall 3 (blocked / escalate to Tobias): {fall3}",
    ]
    if fall3:
        lines.append("")
        lines.append("### Escalate to Tobias")
        for r in rows:
            if r["fall"] == 3:
                reason = r["field_of_use_flag"] or r["category"]
                lines.append(f"- **{r['name']}** ({r['license']}) - {reason}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-r", "--requirements-paths", nargs="+", default=None,
                     help="requirements.txt / pyproject.toml paths (default: pyproject.toml in cwd)")
    ap.add_argument("-l", "--license", default="Apache-2.0", dest="project_license",
                     help="this project's license, default Apache-2.0 (permissive, per framework goal)")
    ap.add_argument("-o", "--out", default="license_report.md", help="markdown report output path")
    ap.add_argument("--json-out", default=None, help="optional: also write raw+classified JSON here")
    ap.add_argument("licensecheck_args", nargs=argparse.REMAINDER,
                     help="any extra args passed straight through to licensecheck, after --")
    args = ap.parse_args()

    extra = args.licensecheck_args
    if extra and extra[0] == "--":
        extra = extra[1:]

    data = run_licensecheck(args.requirements_paths, args.project_license, extra)
    rows = build_report(data)
    md = to_markdown(rows, args.project_license)

    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Wrote {args.out}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")

    fall3 = [r for r in rows if r["fall"] == 3]
    if fall3:
        print(f"\n{len(fall3)} package(s) need Tobias escalation:")
        for r in fall3:
            print(f"  - {r['name']} ({r['license']}) - {r['field_of_use_flag'] or r['category']}")


if __name__ == "__main__":
    main()
