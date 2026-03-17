

import json
import argparse
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime

from model_client import AyaClient
from prompt_loader import load_mkqa, load_hallomtbench, build_prompt_for_language
from config.experiment_config import (
    MODELS,
    LANGUAGES,
    IN_DISTRIBUTION,
    SAMPLES_PER_PROMPT,
    TEMPERATURE,
    MAX_TOKENS,
    DATASET_SOURCE,
    N_PROMPTS,
    DATA_DIR,
)

DATA_DIR.mkdir(exist_ok=True)


def run_collection(prompts_df: pd.DataFrame, client: AyaClient) -> list[dict]:
    records = []

    total = sum(
        1
        for _ in prompts_df.itertuples()
        for model_key in MODELS
        for lang in LANGUAGES[model_key]
        if f"prompt_{lang}" in prompts_df.columns
    )
    done = 0

    for _, row in prompts_df.iterrows():
        prompt_id = row["prompt_id"]

        for model_key, model_name in MODELS.items():
            for lang in LANGUAGES[model_key]:
                prompt_col = f"prompt_{lang}"
                answer_col = f"answer_{lang}"

                # Skip if this language isn't present in the dataset row
                base_prompt = row.get(prompt_col)
                if not base_prompt or pd.isna(base_prompt):
                    continue

                ground_truth = row.get(answer_col)
                full_prompt  = build_prompt_for_language(str(base_prompt), lang)
                in_dist      = lang in IN_DISTRIBUTION.get(model_key, [])

                done += 1
                print(f"  [{done}/{total}] {model_key} | {lang} | {prompt_id}")

                responses = client.query(
                    model_name=model_name,
                    prompt=full_prompt,
                    n_samples=SAMPLES_PER_PROMPT,
                )

                for sample_idx, response in enumerate(responses):
                    records.append({
                        # Identifiers
                        "prompt_id":        prompt_id,
                        "model":            model_key,
                        "language":         lang,
                        "sample_idx":       sample_idx,
                        # Experimental axis
                        "in_distribution":  in_dist,
                        # Content
                        "prompt":           str(base_prompt),
                        "full_prompt":      full_prompt,
                        "response":         response,
                        "ground_truth":     str(ground_truth) if ground_truth else None,
                        # Hyperparameters — logged on every row for reproducibility
                        "temperature":      TEMPERATURE,
                        "max_tokens":       MAX_TOKENS,
                        "dataset":          row.get("source", DATASET_SOURCE),
                        "timestamp":        datetime.utcnow().isoformat(),
                    })

    return records


def save_outputs(records: list[dict]) -> None:
    config_str = json.dumps(
        {
            "models":      MODELS,
            "languages":   LANGUAGES,
            "temperature": TEMPERATURE,
            "samples":     SAMPLES_PER_PROMPT,
            "dataset":     DATASET_SOURCE,
        },
        sort_keys=True,
    )
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # raw JSON
    with open(DATA_DIR / "raw_outputs.json", "w") as f:
        json.dump(records, f, indent=2)

    # JSONL — one record per line
    with open(DATA_DIR / "sample_outputs.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # CSV
    pd.DataFrame(records).to_csv(DATA_DIR / "run_summary.csv", index=False)

    # Metadata
    meta = {
        "run_timestamp":      ts,
        "config_hash":        config_hash,
        "models":             MODELS,
        "languages":          LANGUAGES,
        "in_distribution":    IN_DISTRIBUTION,
        "temperature":        TEMPERATURE,
        "max_tokens":         MAX_TOKENS,
        "samples_per_prompt": SAMPLES_PER_PROMPT,
        "dataset":            DATASET_SOURCE,
        "n_records":          len(records),
    }
    with open(DATA_DIR / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {len(records)} records to {DATA_DIR}/")
    print(f"Config hash: {config_hash}  |  Timestamp: {ts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default=DATASET_SOURCE,
        choices=["mkqa", "hallomtbench"],
        help="Which dataset to load prompts from"
    )
    parser.add_argument(
        "--n", type=int, default=N_PROMPTS,
        help="Number of prompts to run"
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} ({args.n} prompts)...")
    all_langs = list(set(l for langs in LANGUAGES.values() for l in langs))
    if args.dataset == "mkqa":
        prompts_df = load_mkqa(languages=all_langs, n_samples=args.n)
    else:
        prompts_df = load_hallomtbench(n_samples=args.n)

    print("Initialising Cohere client...")
    client = AyaClient()

    print("Starting collection...\n")
    records = run_collection(prompts_df, client)
    save_outputs(records)