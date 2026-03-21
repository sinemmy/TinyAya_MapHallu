# run_experiments/ — Unified Data Collection Pipeline

## What It Does
Unified pipeline that runs all 3 experiment types (CMDR, Hallucination Rate, PSS) across configurable models, languages, and datasets (XNLI + MKQA). Data collection only — analysis is deferred to a future `run_analysis/` folder.

## Key Design Decisions
- **Logprobs always extracted** (not just CMDR) — enables confidence analysis across all datasets
- **Full API response stored** as JSON (message, usage, logprobs, finish_reason)
- **Config tracked once** in `config.json` at run start — no per-record duplication
- **`run_id`** = start timestamp (e.g. `20260321_143000`); all output lives under `output/{run_id}/`
- **PSS is MKQA-only** (skipped for XNLI)
- **Both datasets use `response_format={"type": "json_object"}`** — XNLI: `{"label": "..."}`, MKQA: `{"answer": "..."}`
- **`model_client.py`** serialises Cohere logprob objects into plain dicts via `_logprobs_to_serialisable()`
- **`run.py`** uses `sys.path.insert(0, ...)` so imports work from any directory

## Run Types
1. **Base**: For each (model, language, dataset, sample) → query → save response + logprobs + gold comparison. Feeds CMDR and Hallucination Rate analysis.
2. **PSS**: For each (model, language, MKQA sample, variant) → query with prompt variant → save response + variant info. Feeds PSS analysis.

## Output Schema (`output/{run_id}/`)

**`config.json`** — frozen config snapshot (written once):
`run_id, start_timestamp, models, languages, datasets, experiments, num_dataset_samples, nreps, temperature, max_tokens, seed, cli_args, config_file_path`

**`checkpoint.json`** — completion state (updated per unit):
```json
{
  "completed": [{"dataset": "xnli", "model": "tiny-aya-global", "language": "en", "experiment": "base", "n_completed": 300}],
  "in_progress": {"dataset": "xnli", "model": "tiny-aya-global", "language": "fr", "experiment": "base", "n_completed": 142}
}
```

**`{dataset}_{model}_{language}_{experiment}.jsonl`** — one record per query:
`run_id, dataset, model, language, sample_id, rep, experiment_type, variant_type, prompt, response (full API JSON), parsed_label|null, gold_label|null, gold_answers|null, is_correct, sequence_probability`

## Checkpointing & Resume
- JSONL records flushed immediately; on resume, runner scans existing records to find completed `(sample_id, rep)` pairs (or `(sample_id, variant_type, rep)` for PSS)
- `checkpoint.json` tracks completed units; `--resume {run_id}` loads original config and skips finished work
- Deterministic iteration order: dataset → model → language → experiment → sample

## Implementation Status (2026-03-21)
**Pipeline fully tested and working.** Successfully completed test runs with real API key.

### Bugs Fixed
- `run.py`: `cfg['num_samples']` → `cfg['num_dataset_samples']` (KeyError fix)
- `data/load_mkqa.py`: Language alias mapping (`zh` → `zh_cn`) now preserves the caller's original language code in returned samples, so runner filtering works correctly
- `model_client.py`: Added exponential backoff retry (up to 5 attempts, 2s base delay) for 403, 429, and 5xx errors

### Known Behavior
- `hi` and `sw` are not in MKQA — correctly skipped for MKQA base and PSS experiments
- HuggingFace logs "Generating train split" for XNLI but the code uses `split="test"`
- `trust_remote_code` warning for MKQA is cosmetic (HuggingFace deprecated the flag)

### Public API Reference
| Module | Public API |
|---|---|
| `config.settings` | `load_config(argv)`, `save_config(cfg, output_dir)` |
| `data.load_xnli` | `load_xnli(languages, num_samples=300, seed=42)` — `None` = all |
| `data.load_mkqa` | `load_mkqa(languages, num_samples=500, seed=42)` — `None` = all |
| `model_client` | `query_model(prompt, model, ...)`, `calculate_sequence_probability(logprobs_data)` |
| `prompts.xnli` | `build_xnli_prompt(premise, hypothesis)`, `XNLI_RESPONSE_FORMAT` |
| `prompts.mkqa` | `build_mkqa_prompt(query, language)`, `MKQA_RESPONSE_FORMAT` |
| `prompts.variants` | `generate_variants(base_prompt)` |
| `evaluation.xnli_eval` | `is_correct(response_text, gold_label)`, `parse_label(response_text)` |
| `evaluation.mkqa_eval` | `is_correct(response_text, gold_answers)`, `parse_answer(response_text)` |
| `runners.checkpoint` | `load_checkpoint()`, `save_checkpoint()`, `is_unit_completed()`, `mark_unit_completed()`, `mark_in_progress()`, `scan_completed_base_keys()`, `scan_completed_pss_keys()` |
| `runners.base_runner` | `run_base(cfg)` |
| `runners.pss_runner` | `run_pss(cfg)` |
