# TinyAya MapHallu

Research project studying **multilingual hallucination** in Cohere's TinyAya models. Measures hallucination rates, cross-lingual model disagreement (CMDR), and prompt sensitivity (PSS) across languages and models.

## Quick Start

```bash
# 1. Clone & enter the repo
git clone https://github.com/<your-org>/TinyAya_MapHallu.git
cd TinyAya_MapHallu

# 2. Set up the environment (requires uv)
uv sync                          # installs all deps into .venv
source .venv/bin/activate

# 3. Add your Cohere API key
echo 'COHERE_API_KEY=your-key-here' > .env

# 4. Run an experiment
cd run_experiments
python run.py --models tiny-aya-global --languages en --datasets xnli --experiments base --num-dataset-samples 5
```

## Project Structure

| Folder | Purpose |
|---|---|
| `run_experiments/` | **Unified data collection pipeline** — runs all experiment types across models, languages, and datasets |
| `CMDR/` | Cross-lingual Model Disagreement Rate (legacy, standalone) |
| `hallucination-rate/` | Hallucination rate via gold answer match (legacy, standalone) |
| `Prompt Sensitivity Score/` | Prompt Sensitivity Score analysis (legacy, standalone) |
| `src/` | Shared helpers (Cohere client) |

## run_experiments/ — Main Pipeline

### What It Does

Runs configurable experiments against Cohere's chat API and saves full responses (with logprobs) as JSONL. Supports two experiment types:

- **base** — Query the model once per question, compare to gold labels. Works with both XNLI (NLI classification) and MKQA (open QA).
- **pss** — Prompt Sensitivity Score. Generates 4 prompt variants per question and queries each. MKQA only.

### CLI Flags

```
python run.py [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `--models` | Cohere model IDs (space-separated) | from config |
| `--languages` | Language codes (space-separated) | from config |
| `--datasets` | `xnli`, `mkqa`, or both | `xnli` |
| `--experiments` | `base`, `pss`, or both | `base pss` |
| `--num-dataset-samples` | Number of questions to draw per language, or `all` | `300` |
| `--nreps` | Repeated API calls per question (independent, stateless) | `1` |
| `--temperature` | Sampling temperature | `0.3` |
| `--max-tokens` | Max output tokens | `512` |
| `--seed` | Random seed for dataset sampling | `42` |
| `--config` | Path to a YAML config override file | — |
| `--output-dir` | Base output directory | `run_experiments/output/` |
| `--resume` | Resume a previous run by its `run_id` | — |

### Examples

```bash
# Full run for a single model — all languages, all samples, 5 reps
# (hi and sw are XNLI-only; zh auto-maps to zh_cn for MKQA)
python run.py --models tiny-aya-global \
              --languages ar de en ru th zh hi sw \
              --datasets xnli mkqa \
              --experiments base pss \
              --num-dataset-samples all \
              --nreps 5

# Quick full test : all languages , 1 model, nreps 
python run.py --models tiny-aya-global \
              --languages ar de en ru th zh hi sw \
              --datasets xnli mkqa \
              --experiments base pss \
              --num-dataset-samples 2 \
              --nreps 2

# Quick test: 1 model, 1 language, 5 samples
python run.py --models tiny-aya-global --languages en --datasets xnli --experiments base --num-dataset-samples 5

# Multiple models
python run.py --models tiny-aya-global tiny-aya-fire --languages en ar zh --datasets xnli mkqa --experiments base

# PSS with 3 repeated calls per question
python run.py --models tiny-aya-global --languages en --datasets mkqa --experiments pss --num-dataset-samples 50 --nreps 3

# Resume an interrupted run
python run.py --resume 20260321_143000
```

### Output

Each run creates `output/{run_id}/` containing:

- **`config.json`** — Frozen snapshot of all settings for this run
- **`checkpoint.json`** — Tracks which (dataset, model, language, experiment) units are complete (for resume)
- **`{dataset}_{model}_{language}_{experiment}.jsonl`** — One JSON record per API call, including:
  - Full API response (text, logprobs, usage, finish_reason)
  - Gold label / gold answers
  - `is_correct` evaluation
  - `sequence_probability` (derived from logprobs)
  - `rep` index (for nreps > 1)

### Checkpointing & Resume

The pipeline is crash-safe. If interrupted, re-run with `--resume {run_id}` and it picks up exactly where it left off by scanning existing JSONL records.

## Datasets

- **XNLI** — Natural Language Inference. Given a premise and hypothesis, classify as entailment / neutral / contradiction. Loaded via HuggingFace `datasets`.
- **MKQA** — Multilingual open-domain QA with gold answer strings. Evaluation uses normalized substring matching.

## Models

Configurable via `--models`. Tested with:
- `tiny-aya-global`
- `tiny-aya-fire`
- `command-a-translate-08-2025`

All use Cohere's v2 Chat API with `logprobs=True` and `response_format={"type": "json_object"}`.

## Environment Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
source .venv/bin/activate
```

Create a `.env` file in the project root with your Cohere API key:
```
COHERE_API_KEY=your-key-here
```

## License

See [LICENSE](LICENSE).
