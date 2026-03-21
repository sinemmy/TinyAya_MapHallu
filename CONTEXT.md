# TinyAya_MapHallu Project Context

## What This Project Is
Research project studying multilingual hallucination in Cohere's TinyAya models. Three existing experiment types, building a unified runner.

## Existing Folders

### CMDR/ — Cross-lingual Model Disagreement Rate
- **Dataset**: XNLI (NLI classification: premise + hypothesis → entailment/neutral/contradiction)
- **Key files**: CMDR.py (pipeline), helpers.py (Cohere client with logprobs)
- **What it does**: Queries model in multiple languages, compares predicted labels + confidence (via logprobs) across language pairs
- **Extracts**: logprobs → sequence_probability, predicted label
- **Evaluation**: Gold label comparison (XNLI gold labels)

### hallucination-rate/ — Hallucination Rate via Gold Answer Match
- **Dataset**: MKQA (multilingual QA with gold answers)
- **Key files**: evaluate.py, metrics.py, model_client.py, data/load_mkqa.py
- **What it does**: Queries model, checks if response contains gold answer substring
- **Extracts**: response text, output tokens (NO logprobs currently)
- **Evaluation**: Substring containment — `is_correct = any(gold in response)`. Hallucination rate = 1 - accuracy
- **Limitation**: Hardcoded to compare exactly 2 models as a pair

### Prompt Sensitivity Score/ — PSS
- **Dataset**: MKQA (same as hallucination-rate)
- **Key files**: collect_data.py, prompt_variants.py, analysis/compute_pss.py, analysis/compute_pss_score.py
- **What it does**: Generates 4 prompt variants (paraphrase, instruction, context, short) per question, measures response stability
- **Extracts**: response text, optional logprobs (disabled by default)
- **Evaluation**: Semantic similarity, entity change rate, lexical overlap, response length variance → composite PSS score

### src/helpers.py — Shared Cohere Client
- Most complete version of `query_model()` with logprobs support
- `get_text_from_response()`, `get_logprobs_from_response()`
- Uses `cohere.ClientV2`, API key from .env

## What We're Building: run_experiments/
Unified data collection folder. See run_experiments/PLAN.md for full details.
- Runs all experiment types across configurable models, languages, datasets
- Always extracts logprobs (unlike current hallucination-rate and PSS)
- Stores full API response JSON
- Config tracked in config.json per run, checkpoint.json for resume
- Analysis deferred to a separate run_analysis/ folder (not built yet)

## Implementation Status
All code for run_experiments/ is written, syntax-checked, and import-verified. Remaining: dry-run test with a real API key.
See run_experiments/PLAN.md for full details and verification steps.

## All Models Use Cohere v2 Chat API
- API key in .env as COHERE_API or COHERE_API_KEY
- Models: tiny-aya-global, tiny-aya-fire, command-a-translate-08-2025 (configurable)
- logprobs=True returns token-level log probabilities
- response_format={"type": "json_object"} forces JSON output
- No multi-sample (n) parameter — 1 response per call

## Environment
- Uses `uv` for package management; activate with `source .venv/bin/activate`
- Key dependencies: `cohere`, `datasets` (HuggingFace), `pyyaml`, `tqdm`, `python-dotenv`

## CLI Quick Reference (run_experiments/)
```bash
python run.py --models tiny-aya-global tiny-aya-fire \
              --languages en fr ar \
              --datasets xnli mkqa \
              --experiments base pss \
              --num-dataset-samples 300 \
              --nreps 1
```
- `--num-dataset-samples`: integer or `all` (uses full dataset)
- `--nreps`: number of independent repeated API calls per question (default 1)
- `--resume {run_id}`: resume an interrupted run
- Config key in code/YAML: `num_dataset_samples` (int or None for all), `nreps` (int)
- Each JSONL record includes a `rep` field (0-indexed)
- No multi-sample (n) parameter — 1 response per call
