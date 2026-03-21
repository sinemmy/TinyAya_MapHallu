#!/usr/bin/env python3
"""
run.py — CLI entry point for run_experiments.

Usage examples:
    # Fresh run (base only, small test)
    python run.py --models tiny-aya-global --languages en --datasets xnli --experiments base --num-samples 5

    # Full run with YAML config
    python run.py --config config/my_run.yaml

    # Resume a crashed run
    python run.py --resume 20260321_143000
"""

import sys
from pathlib import Path

# Ensure the run_experiments directory is on the path so imports work
# regardless of where the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import load_config, save_config
from runners.base_runner import run_base
from runners.pss_runner import run_pss


def main(argv: list[str] | None = None) -> None:
    cfg = load_config(argv)

    print(f"Run ID : {cfg['run_id']}")
    print(f"Models : {cfg['models']}")
    print(f"Langs  : {cfg['languages']}")
    print(f"Datasets: {cfg['datasets']}")
    print(f"Experiments: {cfg['experiments']}")
    print(f"Samples: {cfg['num_dataset_samples']}")
    print(f"Output : {cfg['output_dir']}")
    if cfg.get("_resumed"):
        print("(Resumed from existing run)")
    print()

    # Save / preserve config
    if not cfg.get("_resumed"):
        save_config(cfg, cfg["output_dir"])
        print(f"Config saved to {cfg['output_dir']}/config.json\n")

    # Run experiments in deterministic order
    if "base" in cfg["experiments"]:
        print("=" * 60)
        print("RUNNING BASE EXPERIMENTS")
        print("=" * 60)
        run_base(cfg)
        print()

    if "pss" in cfg["experiments"]:
        print("=" * 60)
        print("RUNNING PSS EXPERIMENTS (MKQA only)")
        print("=" * 60)
        run_pss(cfg)
        print()

    print("All experiments complete.")


if __name__ == "__main__":
    main()
