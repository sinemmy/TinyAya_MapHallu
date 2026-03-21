# TinyAya_MapHallu Project Context

## What This Project Is
Research project studying multilingual hallucination in Cohere's TinyAya models. Three experiment types exist as standalone folders; `run_experiments/` unifies them into a single data collection pipeline.

## Existing Experiment Folders (standalone, already completed)
- **CMDR/** — Cross-lingual Model Disagreement Rate. XNLI dataset, compares predicted NLI labels + logprob confidence across language pairs.
- **hallucination-rate/** — Gold answer substring match on MKQA. Hallucination rate = 1 - accuracy.
- **Prompt Sensitivity Score/** — MKQA with 4 prompt variants per question, measures response stability via semantic similarity and entity change rate.

## run_experiments/ — Unified Pipeline
Runs all 3 experiment types across configurable models, languages, datasets. Always extracts logprobs. Analysis deferred to future `run_analysis/`. **Pipeline fully tested and working** (2026-03-21). See `run_experiments/PLAN.md` for output schema, checkpointing, and public API reference.

### Bugs Fixed (2026-03-21)
- `run.py` referenced `cfg['num_samples']` instead of `cfg['num_dataset_samples']` → KeyError on startup. Fixed.
- `data/load_mkqa.py` mapped `zh` → `zh_cn` internally but returned `language: "zh_cn"` in samples, causing the runner to find 0 matches when filtering for `"zh"`. Fixed: loader now returns the caller's original language code.
- `model_client.py` had no retry logic → transient 403/429 errors caused skipped samples. Fixed: added exponential backoff (up to 5 retries, 2s base delay) for 403, 429, and 5xx errors.

### Known Behavior
- `hi` and `sw` are XNLI-only languages (not in MKQA) — correctly skipped for MKQA experiments.
- XNLI uses the `test` split (HuggingFace logs "Generating train split" but this is a misleading library message).

## Cohere v2 Chat API
- API key in `.env` as `COHERE_API` or `COHERE_API_KEY`
- Models: `tiny-aya-global`, `tiny-aya-fire`, `command-a-translate-08-2025` (configurable)
- `logprobs=True` returns token-level log probabilities
- `response_format={"type": "json_object"}` forces JSON output
- Returns 1 response per call (no multi-sample `n` parameter)

## Environment
- Uses `uv` for package management; activate with `source .venv/bin/activate`
- Key deps: `cohere`, `datasets` (HuggingFace), `pyyaml`, `tqdm`, `python-dotenv`

## CLI Quick Reference (run_experiments/)
```bash
python run.py --models tiny-aya-global tiny-aya-fire \
              --languages en fr ar \
              --datasets xnli mkqa \
              --experiments base pss \
              --num-dataset-samples 300 \
              --nreps 1
```
- `--num-dataset-samples`: integer or `all` (stored as `None` for all)
- `--nreps`: repeated independent API calls per question (default 1)
- `--resume {run_id}`: resume an interrupted run
