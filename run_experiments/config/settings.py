"""
config/settings.py — Load, merge and freeze experiment configuration.

Priority (highest wins): CLI args → YAML config file → default.yaml
"""

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_DEFAULT_PATH = Path(__file__).parent / "default.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge *override* into *base* (non-destructive)."""
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def _parse_num_dataset_samples(value) -> int | None:
    """Parse num_dataset_samples: 'all' → None (meaning use all), otherwise int."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "all":
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run experiments")
    p.add_argument("--models", nargs="+", help="Cohere model IDs")
    p.add_argument("--languages", nargs="+", help="Language codes")
    p.add_argument("--datasets", nargs="+", choices=["xnli", "mkqa"], help="Datasets")
    p.add_argument("--experiments", nargs="+", choices=["base", "pss"], help="Experiment types")
    p.add_argument("--config", type=str, help="Path to YAML config override file")
    p.add_argument("--num-dataset-samples", type=str, help="Number of questions to draw per language, or 'all' for the full dataset")
    p.add_argument("--nreps", type=int, help="Number of repeated API calls per question (default: 1)")
    p.add_argument("--temperature", type=float, help="Sampling temperature")
    p.add_argument("--max-tokens", type=int, help="Max output tokens")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--output-dir", type=str, help="Base output directory")
    p.add_argument("--resume", type=str, metavar="RUN_ID", help="Resume a previous run by its run_id")
    return p


def load_config(argv: list[str] | None = None) -> dict:
    """
    Build a fully resolved config dict.

    For ``--resume``, the original config.json is loaded and CLI args are ignored.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- resume path: load original config, skip merging ---
    if args.resume:
        output_base = args.output_dir or str(Path(__file__).resolve().parent.parent / "output")
        config_path = Path(output_base) / args.resume / "config.json"
        if not config_path.exists():
            print(f"ERROR: cannot resume — {config_path} not found", file=sys.stderr)
            sys.exit(1)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["_resumed"] = True
        return cfg

    # --- normal path: defaults → yaml override → CLI ---
    cfg = _load_yaml(_DEFAULT_PATH)

    if args.config:
        override = _load_yaml(Path(args.config))
        cfg = _deep_merge(cfg, override)

    # CLI overrides (only non-None values)
    cli_map = {
        "models": args.models,
        "languages": args.languages,
        "datasets": args.datasets,
        "experiments": args.experiments,
        "num_dataset_samples": _parse_num_dataset_samples(args.num_dataset_samples) if args.num_dataset_samples is not None else None,
        "nreps": args.nreps,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    for k, v in cli_map.items():
        if v is not None:
            cfg[k] = v

    # Normalize num_dataset_samples from YAML (could be "all" string)
    cfg["num_dataset_samples"] = _parse_num_dataset_samples(cfg.get("num_dataset_samples", 300))

    # Generate run identity
    now = datetime.now(timezone.utc)
    cfg["start_timestamp"] = now.isoformat()
    cfg["run_id"] = now.strftime("%Y%m%d_%H%M%S")

    # Output dir
    output_base = args.output_dir or str(Path(__file__).resolve().parent.parent / "output")
    cfg["output_dir"] = str(Path(output_base) / cfg["run_id"])

    # Store CLI args for reproducibility
    cfg["cli_args"] = sys.argv[1:] if argv is None else list(argv)
    cfg["config_file"] = args.config

    cfg["_resumed"] = False
    return cfg


def save_config(cfg: dict, output_dir: str | Path) -> Path:
    """Write frozen config.json to the run output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return path
