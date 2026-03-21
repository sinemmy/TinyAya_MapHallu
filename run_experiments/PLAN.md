# Plan: Mega Run Experiments Folder

## TL;DR
Create `run_experiments/` — a unified data collection pipeline that runs all 3 experiment types (CMDR, Hallucination Rate, PSS) across configurable models, languages, and datasets (XNLI + MKQA). Extracts logprobs on every run. Analysis is deferred to a separate `run_analysis/` folder later. Config is fully tracked in output files. Work is splittable by model across 4 people.

## Context & Key Decisions
- **Datasets**: XNLI (NLI classification) and MKQA (open QA with gold answers)
- **Evaluation**: Gold-label comparison only (no LLM judge). XNLI: predicted label vs gold label. MKQA: substring containment against gold answer strings.
- **Logprobs**: Extracted on ALL runs (not just CMDR) to enable CMDR-style confidence analysis across datasets
- **Separation**: `run_experiments/` = data collection only. `run_analysis/` = metrics, aggregation, plotting (built later)
- **Splitting work**: By model — each person runs a subset of models across all languages/datasets
- **Config tracking**: `config.json` saved once at run start with all settings. No per-record duplication of temperature/max_tokens/timestamps.
- **Run identity**: `run_id` = start timestamp (e.g. `20260321_143000`). All files for a run live under `output/{run_id}/` and share this ID.
- **No per-prompt metadata**: temperature, max_tokens, timestamp are run-level only (in config.json). No latency_sec captured.
- **Full API response**: Store the full JSON response object from the API (not just text), so we have usage, logprobs, finish_reason, etc.
- **No multi-sample API**: Cohere v2 chat returns 1 response per call. Multiple samples require separate calls.
- **Checkpointing**: Crash-safe resume via checkpoint file + JSONL append. See Checkpointing section.

## Run Types
1. **Base Run**: For each (model, language, dataset, sample) → query model → save (response, logprobs, gold comparison). Serves both CMDR and Hallucination Rate analysis later.
2. **PSS Run**: For each (model, language, **MKQA only**, sample, variant) → query model with prompt variant → save (response, logprobs, variant info). Serves PSS analysis later. **PSS is skipped for XNLI** — can be added later if needed.

## Response Format
- **Both XNLI and MKQA use `response_format={"type": "json_object"}`** for consistency.
- XNLI prompt asks for JSON label: `{"label": "entailment"}`
- MKQA prompt asks for JSON answer: `{"answer": "Paris"}`
- This makes parsing uniform across datasets.

## Steps

### Phase 1: Scaffolding & Config
1. Create folder structure (see below)
2. **`config/settings.py`**: Config loader that merges YAML defaults → YAML file overrides → CLI args. Produces a frozen config dict. Generates a `run_id` (timestamp + short hash). *No dependencies.*
3. **`config/default.yaml`**: Default config values (temperature=0.3, max_tokens=512, num_dataset_samples=300, nreps=1, seed=42, experiments=[base, pss]). *Parallel with step 2.*

### Phase 2: Data & Model Client
4. **`data/load_xnli.py`**: Adapt from `CMDR/CMDR.py` `load_multilingual_data()`. Load XNLI via HuggingFace, return standardized format: `[{sample_id, language, prompt_fields: {premise, hypothesis}, gold_label}]`. *No dependencies.*
5. **`data/load_mkqa.py`**: Adapt from `hallucination-rate/data/load_mkqa.py`. Load MKQA, return standardized format: `[{sample_id, language, prompt_fields: {query}, gold_answers}]`. *Parallel with step 4.*
6. **`model_client.py`**: Unified Cohere client adapted from `src/helpers.py`. Single function `query_model(prompt, model, temperature, max_tokens, logprobs=True, response_format=None)` → returns the **full API response object** (JSON with message, usage, logprobs, finish_reason). Includes `calculate_sequence_probability()` from `CMDR/CMDR.py` as a separate utility. Always requests logprobs. *(parallel with 4-5)*

### Phase 3: Prompt Templates & Evaluation
7. **`prompts/xnli.py`**: Prompt template for XNLI NLI task. Adapted from `CMDR/CMDR.py` — takes premise + hypothesis, asks for JSON label. Uses `response_format="json_object"`. *Depends on step 4 for field names.*
8. **`prompts/mkqa.py`**: Prompt template for MKQA QA task. Asks for JSON answer `{"answer": "..."}`, with optional language instruction appended (adapted from PSS `config/experiment_config.py` LANGUAGE_INSTRUCTIONS). Uses `response_format="json_object"`. *Parallel with step 7.*
9. **`prompts/variants.py`**: PSS prompt variant generator. Adapted from `Prompt Sensitivity Score/prompt_variants.py` `generate_variants()`. *Parallel with steps 7-8.*
10. **`evaluation/xnli_eval.py`**: Parse NLI label from JSON response, compare to gold label → `is_correct` boolean. Adapted from `CMDR/CMDR.py` label extraction. *Parallel with steps 7-9.*
11. **`evaluation/mkqa_eval.py`**: Substring containment check of response against gold answers → `is_correct` boolean. Adapted from `hallucination-rate/metrics.py` `is_correct()`. *Parallel with step 10.*

### Phase 4: Runners
12. **`runners/base_runner.py`**: Core data collection loop. Deterministic order: dataset → model → language. For each sample: build prompt → query model (logprobs=True, gets full API response JSON) → compute sequence_probability → tag correctness → append JSONL record (flush immediately). Updates `checkpoint.json` after each (dataset, model, language, experiment) unit completes. On resume: scans existing JSONL for completed sample_ids and skips them. *Depends on steps 6-8, 10-11.*
13. **`runners/pss_runner.py`**: Same pattern, but **MKQA only** (PSS skipped for XNLI). Generates base + 4 variants per sample → writes JSONL records with variant_type. Same checkpoint/resume logic. *Depends on steps 6-9.*

### Phase 5: Entry Point
14. **`run.py`**: CLI entry point using argparse. Accepts `--models`, `--languages`, `--datasets`, `--experiments`, `--config` (YAML path), `--num-dataset-samples`, `--nreps`, `--output-dir`, `--resume {run_id}`. On fresh run: generates `run_id` from start timestamp, saves `config.json`, initializes `checkpoint.json`. On resume: loads existing config + checkpoint, skips completed work. *Depends on steps 2, 12-13.*

## Folder Structure
```
run_experiments/
├── README.md
├── requirements.txt
├── run.py                      # CLI entry point
├── config/
│   ├── __init__.py
│   ├── settings.py             # config loading & merging
│   └── default.yaml            # default values
├── data/
│   ├── __init__.py
│   ├── load_xnli.py            # XNLI loader (adapted from CMDR)
│   └── load_mkqa.py            # MKQA loader (adapted from hallucination-rate)
├── model_client.py             # unified Cohere client (always logprobs)
├── prompts/
│   ├── __init__.py
│   ├── xnli.py                 # NLI prompt template
│   ├── mkqa.py                 # QA prompt template
│   └── variants.py             # PSS variant generator
├── evaluation/
│   ├── __init__.py
│   ├── xnli_eval.py            # gold label comparison for NLI
│   └── mkqa_eval.py            # gold answer substring match
├── runners/
│   ├── __init__.py
│   ├── checkpoint.py           # checkpoint read/write, JSONL scan for resume
│   ├── base_runner.py          # standard data collection
│   └── pss_runner.py           # PSS variant collection
└── output/                     # created at runtime
```

## Output Schema

### Per-run directory: `output/{run_id}/`

**`config.json`** — frozen config snapshot (written once at run start):
```
run_id, start_timestamp, models, languages, datasets, experiments,
num_dataset_samples, nreps, temperature, max_tokens, seed, cli_args, config_file_path
```

**`checkpoint.json`** — tracks completion state (updated after each unit of work):
```json
{
  "run_id": "20260321_143000",
  "completed": [
    {"dataset": "xnli", "model": "tiny-aya-global", "language": "en", "experiment": "base", "n_completed": 300},
    {"dataset": "xnli", "model": "tiny-aya-global", "language": "en", "experiment": "pss", "n_completed": 300}
  ],
  "in_progress": {"dataset": "xnli", "model": "tiny-aya-global", "language": "fr", "experiment": "base", "n_completed": 142}
}
```

**`{dataset}_{model}_{language}_{experiment}.jsonl`** — one record per query:
```
run_id, dataset, model, language, sample_id, rep,
experiment_type ("base" | "pss"),
variant_type (null for base | "base" | "paraphrase" | "instruction" | "context" | "short"),
prompt,
response (full API response JSON: message, usage, logprobs, finish_reason),
parsed_label (for XNLI) | null,
gold_label (for XNLI) | null,
gold_answers (for MKQA) | null,
is_correct,
sequence_probability (exp(mean(logprobs)))
```

## Checkpointing & Resume

The system uses two-level checkpointing:

1. **JSONL append-safety**: Each record is flushed immediately after writing. On resume, the runner counts existing records in the JSONL file to know which sample_ids are done.

2. **`checkpoint.json`**: Updated after each (dataset, model, language, experiment) unit completes, or periodically mid-unit. Records `completed` list and `in_progress` state with count.

3. **Resume flow**: `python run.py --resume {run_id}` →
   - Load `config.json` from `output/{run_id}/` (uses original config, ignores new CLI args)
   - Load `checkpoint.json` to skip fully completed units
   - For in-progress units, scan JSONL to find last completed sample_id
   - Continue from next sample

4. **Execution order**: The runner iterates in a deterministic order: `for dataset → for model → for language → for experiment → for sample`. This ensures resume always picks up at the right spot.

## Source Files Referenced During Implementation
The code in run_experiments/ was adapted from these existing files (for historical reference only — all code is now self-contained):
- `CMDR/CMDR.py` — XNLI loading, sequence probability calculation, NLI prompt, label extraction
- `CMDR/helpers.py` — Cohere client with logprobs
- `hallucination-rate/data/load_mkqa.py` — MKQA loading
- `hallucination-rate/metrics.py` — gold answer substring check
- `Prompt Sensitivity Score/prompt_variants.py` — PSS variant generation
- `src/helpers.py` — Cohere v2 query_model with logprobs

## Verification
1. Run `python run.py --models tiny-aya-global --languages en --datasets xnli --experiments base --num-dataset-samples 5` → check output JSONL has full API response JSON with logprobs, sequence_probability populated, is_correct + gold_label set
2. Same with `--datasets mkqa` → verify gold_answers present and is_correct uses substring match, logprobs present
3. Run with `--experiments pss` → verify 5 records per sample (base + 4 variants) with variant_type field populated
4. Run with `--config custom.yaml` → verify `config.json` in output dir matches merged config exactly (no per-record temp/max_tokens)
5. Verify `config.json` has `start_timestamp` and `run_id` derived from it
6. Verify `checkpoint.json` is updated after each (dataset, model, language, experiment) unit
7. Resume test: kill mid-run, restart with `--resume {run_id}`, confirm it loads original config, skips completed units, and resumes mid-JSONL by scanning for completed sample_ids
8. Verify no per-prompt timestamps or latency in JSONL records

## Scope
**Included**: Data collection, logprobs extraction, gold-label tagging, config tracking, resume support, CLI + YAML config
**Excluded**: Aggregated metrics (CMDR rates, hallucination rates, PSS scores), plotting, cross-language/cross-dataset comparison — all deferred to `run_analysis/`

## Implementation Status (as of 2026-03-21)

### DONE — All files created, syntax-checked, import-verified:
- `config/__init__.py`, `config/settings.py`, `config/default.yaml`
- `data/__init__.py`, `data/load_xnli.py`, `data/load_mkqa.py`
- `model_client.py`
- `prompts/__init__.py`, `prompts/xnli.py`, `prompts/mkqa.py`, `prompts/variants.py`
- `evaluation/__init__.py`, `evaluation/xnli_eval.py`, `evaluation/mkqa_eval.py`
- `runners/__init__.py`, `runners/checkpoint.py`, `runners/base_runner.py`, `runners/pss_runner.py`
- `run.py`
- `requirements.txt`

### TODO:
1. **Dry run test**: Run with a real API key to verify end-to-end: `python run.py --models tiny-aya-global --languages en --datasets xnli --experiments base --num-dataset-samples 5`
2. **Review**: User should review generated code for correctness

### Key design notes:
- `runners/checkpoint.py` was added (not in original plan) — handles checkpoint.json read/write and JSONL scanning for resume
- PSS is MKQA-only (skipped for XNLI) — enforced in `pss_runner.py`
- Both XNLI and MKQA use `response_format={"type": "json_object"}` — XNLI asks for `{"label": "..."}`, MKQA for `{"answer": "..."}`
- `model_client.py` serialises Cohere logprob objects into plain dicts for JSON storage via `_logprobs_to_serialisable()`
- `run.py` uses `sys.path.insert(0, ...)` so imports work when invoked from any directory
- CLI flag `--num-dataset-samples` accepts an integer or `all`; stored in config as `num_dataset_samples` (int or None for all)
- CLI flag `--nreps` controls repeated independent API calls per question; stored in config as `nreps` (int, default 1)
- Each JSONL record includes a `rep` field (0-indexed); resume logic tracks `(sample_id, rep)` pairs for base and `(sample_id, variant_type, rep)` for PSS

### Actual function/class names (for import reference):
| Module | Public API |
|---|---|
| `config.settings` | `load_config(argv)`, `save_config(cfg, output_dir)` |
| `data.load_xnli` | `load_xnli(languages, num_samples=300, seed=42)` — `num_samples=None` means all |
| `data.load_mkqa` | `load_mkqa(languages, num_samples=500, seed=42)` — `num_samples=None` means all |
| `model_client` | `query_model(prompt, model, ...)`, `calculate_sequence_probability(logprobs_data)` |
| `prompts.xnli` | `build_xnli_prompt(premise, hypothesis)`, `XNLI_RESPONSE_FORMAT` |
| `prompts.mkqa` | `build_mkqa_prompt(query, language)`, `MKQA_RESPONSE_FORMAT` |
| `prompts.variants` | `generate_variants(base_prompt)` |
| `evaluation.xnli_eval` | `is_correct(response_text, gold_label)`, `parse_label(response_text)` |
| `evaluation.mkqa_eval` | `is_correct(response_text, gold_answers)`, `parse_answer(response_text)` |
| `runners.checkpoint` | `load_checkpoint()`, `save_checkpoint()`, `is_unit_completed()`, `mark_unit_completed()`, `mark_in_progress()`, `scan_completed_base_keys()`, `scan_completed_pss_keys()`, `scan_completed_sample_ids()` |
| `runners.base_runner` | `run_base(cfg)` |
| `runners.pss_runner` | `run_pss(cfg)` |
