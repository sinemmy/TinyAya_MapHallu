"""
Loader for consolidated experiment DataFrames.

Reads JSONL files from one or more run_experiments/output/{run_id}/ folders
and returns three DataFrames:

  mkqa_base  — MKQA base experiment records  (feeds Hallucination Rate, AMR, SCS/CLC)
  xnli_base  — XNLI base experiment records  (feeds CMDR)
  mkqa_pss   — MKQA PSS experiment records    (feeds PSS)

Each DataFrame carries `model` and `language` columns from the source records.

Usage:
    from load_data import load_run, load_runs

    # Single run
    mkqa, xnli, pss = load_run("run_experiments/output/global_20260321_172527")

    # Multiple runs (e.g. all four models)
    mkqa, xnli, pss = load_runs([
        "run_experiments/output/global_20260321_172527",
        "run_experiments/output/fire_20260323_135908",
    ])
"""

import json
from pathlib import Path

import pandas as pd

# ── Columns to keep (flat) ──────────────────────────────────────────────
# Heavy nested fields like full logprobs are dropped by default to keep
# the DataFrame manageable.  `response_text` is extracted from response.text.
_SCALAR_FIELDS = [
    "run_id",
    "dataset",
    "model",
    "language",
    "sample_id",
    "rep",
    "experiment_type",
    "variant_type",
    "prompt",
    "parsed_label",
    "gold_label",
    "is_correct",
    "sequence_probability",
]


def _parse_record(raw: dict) -> dict:
    """Flatten one JSONL record into a row-friendly dict."""
    row = {k: raw.get(k) for k in _SCALAR_FIELDS}

    # Extract nested response fields
    resp = raw.get("response") or {}
    row["response_text"] = resp.get("text")
    row["finish_reason"] = resp.get("finish_reason")
    usage = resp.get("usage") or {}
    row["input_tokens"] = usage.get("input_tokens")
    row["output_tokens"] = usage.get("output_tokens")

    # Gold answers: list[{type, text}] → list[str] of just the text values
    gold = raw.get("gold_answers")
    if gold and isinstance(gold, list):
        row["gold_answers"] = [g["text"] for g in gold if isinstance(g, dict) and "text" in g]
    else:
        row["gold_answers"] = gold  # None or already simple

    return row


def load_run(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all JSONL from a single run directory.

    Returns (mkqa_base, xnli_base, mkqa_pss) DataFrames.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    records = {"mkqa_base": [], "xnli_base": [], "mkqa_pss": []}

    for jsonl_path in sorted(run_dir.glob("*.jsonl")):
        name = jsonl_path.stem  # e.g. mkqa_tiny-aya-global_en_base
        parts = name.rsplit("_", 1)
        experiment = parts[-1] if len(parts) > 1 else ""
        dataset = name.split("_", 1)[0]

        if dataset == "mkqa" and experiment == "base":
            bucket = "mkqa_base"
        elif dataset == "xnli" and experiment == "base":
            bucket = "xnli_base"
        elif dataset == "mkqa" and experiment == "pss":
            bucket = "mkqa_pss"
        else:
            continue

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records[bucket].append(_parse_record(json.loads(line)))

    return (
        pd.DataFrame(records["mkqa_base"]),
        pd.DataFrame(records["xnli_base"]),
        pd.DataFrame(records["mkqa_pss"]),
    )


def load_runs(run_dirs: list[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and concatenate JSONL from multiple run directories.

    Returns (mkqa_base, xnli_base, mkqa_pss) DataFrames with all runs merged.
    """
    mkqa_parts, xnli_parts, pss_parts = [], [], []

    for d in run_dirs:
        m, x, p = load_run(d)
        mkqa_parts.append(m)
        xnli_parts.append(x)
        pss_parts.append(p)

    return (
        pd.concat(mkqa_parts, ignore_index=True) if mkqa_parts else pd.DataFrame(),
        pd.concat(xnli_parts, ignore_index=True) if xnli_parts else pd.DataFrame(),
        pd.concat(pss_parts, ignore_index=True) if pss_parts else pd.DataFrame(),
    )


def discover_runs(
    output_root: str | Path,
    include: set[str] | None = None,
    skip: set[str] | None = None,
) -> list[Path]:
    """Find valid run directories under output_root.

    Args:
        output_root: Parent directory containing run folders.
        include: If provided, *only* these directory names are considered.
        skip: Directory names to exclude (applied after include filter).
    """
    output_root = Path(output_root)
    skip = skip or set()
    candidates = sorted(
        d for d in output_root.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )
    if include is not None:
        candidates = [d for d in candidates if d.name in include]
    if skip:
        candidates = [d for d in candidates if d.name not in skip]
    return candidates


def load_all(
    output_root: str | Path,
    include: set[str] | None = None,
    skip: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Discover and load all runs under output_root.

    Args:
        output_root: Parent directory containing run folders.
        include: If provided, only load these directory names.
        skip: Directory names to exclude from discovery.
    """
    dirs = discover_runs(output_root, include=include, skip=skip)
    if not dirs:
        raise FileNotFoundError(f"No valid run directories found in {output_root}")
    print(f"Loading {len(dirs)} runs: {[d.name for d in dirs]}")
    return load_runs(dirs)
