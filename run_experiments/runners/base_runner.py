"""
runners/base_runner.py — Core data collection loop for base experiments.

Deterministic order: dataset → model → language.
For each sample: build prompt → query model (logprobs=True) → tag correctness → write JSONL.
"""

import json
from pathlib import Path

from tqdm import tqdm

from model_client import query_model, calculate_sequence_probability
from data.load_xnli import load_xnli
from data.load_mkqa import load_mkqa
from prompts.xnli import build_xnli_prompt, XNLI_RESPONSE_FORMAT
from prompts.mkqa import build_mkqa_prompt, MKQA_RESPONSE_FORMAT
from evaluation.xnli_eval import is_correct as xnli_is_correct, parse_label
from evaluation.mkqa_eval import is_correct as mkqa_is_correct, parse_answer
from runners.checkpoint import (
    scan_completed_base_keys,
    load_checkpoint,
    save_checkpoint,
    is_unit_completed,
    mark_unit_completed,
    mark_in_progress,
)


def _jsonl_path(output_dir: Path, dataset: str, model: str, language: str) -> Path:
    safe_model = model.replace("/", "_")
    return output_dir / f"{dataset}_{safe_model}_{language}_base.jsonl"


def _load_data(dataset: str, languages: list[str], num_dataset_samples: int, seed: int) -> list[dict]:
    if dataset == "xnli":
        return load_xnli(languages, num_samples=num_dataset_samples, seed=seed)
    elif dataset == "mkqa":
        return load_mkqa(languages, num_samples=num_dataset_samples, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _build_prompt(dataset: str, sample: dict) -> str:
    if dataset == "xnli":
        return build_xnli_prompt(
            sample["prompt_fields"]["premise"],
            sample["prompt_fields"]["hypothesis"],
        )
    else:
        return build_mkqa_prompt(
            sample["prompt_fields"]["query"],
            language=sample["language"],
        )


def _get_response_format(dataset: str) -> dict:
    if dataset == "xnli":
        return XNLI_RESPONSE_FORMAT
    return MKQA_RESPONSE_FORMAT


def _evaluate(dataset: str, response_text: str, sample: dict) -> dict:
    """Return evaluation fields dict."""
    if dataset == "xnli":
        return {
            "parsed_label": parse_label(response_text),
            "gold_label": sample["gold_label"],
            "gold_answers": None,
            "is_correct": xnli_is_correct(response_text, sample["gold_label"]),
        }
    else:
        return {
            "parsed_label": None,
            "gold_label": None,
            "gold_answers": sample["gold_answers"],
            "is_correct": mkqa_is_correct(response_text, sample["gold_answers"]),
        }


def run_base(cfg: dict) -> None:
    """Run base experiment across all configured datasets, models, and languages."""
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ck = load_checkpoint(output_dir)

    nreps = cfg.get("nreps", 1)

    for dataset in cfg["datasets"]:
        # Load data once per dataset for all languages
        data = _load_data(dataset, cfg["languages"], cfg["num_dataset_samples"], cfg["seed"])

        for model in cfg["models"]:
            for language in cfg["languages"]:
                if is_unit_completed(ck, dataset, model, language, "base"):
                    print(f"  [skip] {dataset}/{model}/{language}/base — already completed")
                    continue

                # Filter data for this language
                samples = [s for s in data if s["language"] == language]
                if not samples:
                    print(f"  [skip] {dataset}/{model}/{language}/base — no samples")
                    mark_unit_completed(ck, dataset, model, language, "base", 0)
                    save_checkpoint(output_dir, ck)
                    continue

                jpath = _jsonl_path(output_dir, dataset, model, language)
                done_keys = scan_completed_base_keys(jpath)
                resp_fmt = _get_response_format(dataset)

                desc = f"{dataset}/{model}/{language}/base"
                n_written = len(done_keys)
                total = len(samples) * nreps

                with open(jpath, "a", encoding="utf-8") as fh:
                    for sample in tqdm(samples, desc=desc):
                        for rep in range(nreps):
                            record_key = f"{sample['sample_id']}|{rep}"
                            if record_key in done_keys:
                                continue

                            prompt = _build_prompt(dataset, sample)

                            try:
                                resp = query_model(
                                    prompt,
                                    model=model,
                                    temperature=cfg["temperature"],
                                    max_tokens=cfg["max_tokens"],
                                    response_format=resp_fmt,
                                )
                            except Exception as e:
                                print(f"  [error] sample {sample['sample_id']} rep {rep}: {e}")
                                failure_record = {
                                    "run_id": cfg["run_id"],
                                    "dataset": dataset,
                                    "model": model,
                                    "language": language,
                                    "sample_id": sample["sample_id"],
                                    "rep": rep,
                                    "experiment_type": "base",
                                    "variant_type": None,
                                    "failed": True,
                                }
                                fh.write(json.dumps(failure_record, ensure_ascii=False) + "\n")
                                fh.flush()
                                continue

                            eval_fields = _evaluate(dataset, resp["text"], sample)
                            seq_prob = calculate_sequence_probability(resp.get("logprobs"))

                            record = {
                                "run_id": cfg["run_id"],
                                "dataset": dataset,
                                "model": model,
                                "language": language,
                                "sample_id": sample["sample_id"],
                                "rep": rep,
                                "experiment_type": "base",
                                "variant_type": None,
                                "prompt": prompt,
                                "response": resp,
                                **eval_fields,
                                "sequence_probability": seq_prob,
                            }

                            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                            fh.flush()
                            n_written += 1

                        # Periodic checkpoint
                        if n_written % 50 == 0:
                            mark_in_progress(ck, dataset, model, language, "base", n_written)
                            save_checkpoint(output_dir, ck)

                mark_unit_completed(ck, dataset, model, language, "base", n_written)
                save_checkpoint(output_dir, ck)
                print(f"  [done] {desc} — {n_written} records")
