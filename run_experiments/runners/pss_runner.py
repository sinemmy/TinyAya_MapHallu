"""
runners/pss_runner.py — PSS variant data collection (MKQA only).

For each sample: generates base + 4 variants, queries model for each,
writes JSONL records with variant_type.
"""

import json
from pathlib import Path

from tqdm import tqdm

from model_client import query_model, calculate_sequence_probability
from data.load_mkqa import load_mkqa
from prompts.mkqa import build_mkqa_prompt, MKQA_RESPONSE_FORMAT
from prompts.variants import generate_variants
from evaluation.mkqa_eval import is_correct as mkqa_is_correct
from runners.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    is_unit_completed,
    mark_unit_completed,
    mark_in_progress,
    scan_completed_pss_keys,
)


def _jsonl_path(output_dir: Path, model: str, language: str) -> Path:
    safe_model = model.replace("/", "_")
    return output_dir / f"mkqa_{safe_model}_{language}_pss.jsonl"


def run_pss(cfg: dict) -> None:
    """Run PSS experiment (MKQA only) across configured models and languages."""
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ck = load_checkpoint(output_dir)

    # PSS only applies to MKQA
    if "mkqa" not in cfg["datasets"]:
        print("  [skip] PSS experiment requires mkqa dataset — not configured")
        return

    nreps = cfg.get("nreps", 1)
    data = load_mkqa(cfg["languages"], num_samples=cfg["num_dataset_samples"], seed=cfg["seed"])

    for model in cfg["models"]:
        for language in cfg["languages"]:
            if is_unit_completed(ck, "mkqa", model, language, "pss"):
                print(f"  [skip] mkqa/{model}/{language}/pss — already completed")
                continue

            samples = [s for s in data if s["language"] == language]
            if not samples:
                print(f"  [skip] mkqa/{model}/{language}/pss — no samples")
                mark_unit_completed(ck, "mkqa", model, language, "pss", 0)
                save_checkpoint(output_dir, ck)
                continue

            jpath = _jsonl_path(output_dir, model, language)
            done_keys = scan_completed_pss_keys(jpath)

            desc = f"mkqa/{model}/{language}/pss"
            n_written = len(done_keys)

            with open(jpath, "a", encoding="utf-8") as fh:
                for sample in tqdm(samples, desc=desc):
                    base_prompt = build_mkqa_prompt(
                        sample["prompt_fields"]["query"],
                        language=language,
                    )

                    # Build list: base + 4 variants
                    prompts_to_run = [{"variant_type": "base", "variant_prompt": base_prompt}]
                    prompts_to_run.extend(generate_variants(base_prompt))

                    for variant in prompts_to_run:
                        for rep in range(nreps):
                            pss_key = f"{sample['sample_id']}|{variant['variant_type']}|{rep}"
                            if pss_key in done_keys:
                                continue

                            try:
                                resp = query_model(
                                    variant["variant_prompt"],
                                    model=model,
                                    temperature=cfg["temperature"],
                                    max_tokens=cfg["max_tokens"],
                                    response_format=MKQA_RESPONSE_FORMAT,
                                )
                            except Exception as e:
                                print(f"  [error] sample {sample['sample_id']} "
                                      f"variant {variant['variant_type']} rep {rep}: {e}")
                                failure_record = {
                                    "run_id": cfg["run_id"],
                                    "dataset": "mkqa",
                                    "model": model,
                                    "language": language,
                                    "sample_id": sample["sample_id"],
                                    "rep": rep,
                                    "experiment_type": "pss",
                                    "variant_type": variant["variant_type"],
                                    "failed": True,
                                }
                                fh.write(json.dumps(failure_record, ensure_ascii=False) + "\n")
                                fh.flush()
                                continue

                            seq_prob = calculate_sequence_probability(resp.get("logprobs"))

                            record = {
                                "run_id": cfg["run_id"],
                                "dataset": "mkqa",
                                "model": model,
                                "language": language,
                                "sample_id": sample["sample_id"],
                                "rep": rep,
                                "experiment_type": "pss",
                                "variant_type": variant["variant_type"],
                                "prompt": variant["variant_prompt"],
                                "response": resp,
                                "parsed_label": None,
                                "gold_label": None,
                                "gold_answers": sample["gold_answers"],
                                "is_correct": mkqa_is_correct(resp["text"], sample["gold_answers"]),
                                "sequence_probability": seq_prob,
                            }

                            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                            fh.flush()
                            n_written += 1

                        if n_written % 50 == 0:
                            mark_in_progress(ck, "mkqa", model, language, "pss", n_written)
                            save_checkpoint(output_dir, ck)

            mark_unit_completed(ck, "mkqa", model, language, "pss", n_written)
            save_checkpoint(output_dir, ck)
            print(f"  [done] {desc} — {n_written} records")
