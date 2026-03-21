# TinyAya_MapHallu Project Context

## What This Project Is
Research project studying multilingual hallucination in Cohere's TinyAya models. Three experiment types exist as standalone folders; `run_experiments/` unifies them into a single data collection pipeline.

## Existing Experiment Folders (standalone, already completed)
- **CMDR/** — Cross-lingual Model Disagreement Rate. XNLI dataset, compares predicted NLI labels + logprob confidence across language pairs.
- **hallucination-rate/** — Gold answer substring match on MKQA. Hallucination rate = 1 - accuracy.
- **Prompt Sensitivity Score/** — MKQA with 4 prompt variants per question, measures response stability via semantic similarity and entity change rate.

## run_experiments/ — Unified Pipeline
Runs all 3 experiment types across configurable models, languages, datasets. Always extracts logprobs. Analysis deferred to future `run_analysis/`. All code is written and import-verified; remaining: dry-run with API key. See `run_experiments/PLAN.md` for output schema, checkpointing, and public API reference.

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
