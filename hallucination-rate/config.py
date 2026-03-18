"""
Configuration for TinyAya Global vs Water experiments.

Experiments:
  - Experiment 1: Out-of-region language (e.g. Arabic) — does Water degrade vs Global?
  - Experiment 2: Asia-Pacific / Europe languages — how do Global vs Water compare in-region?
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
COHERE_API_KEY = os.getenv("COHERE_API") or os.getenv("COHERE_API_KEY")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_GLOBAL = "command-a-translate-08-2025"
MODEL_WATER =  "tiny-aya-global"# "tiny-aya-base" cant call 
MODELS = [MODEL_GLOBAL, MODEL_WATER]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
MKQA_NUM_SAMPLES = 500
MKQA_SEED = 42

# ---------------------------------------------------------------------------
# Languages by region (TinyAya Water: "strongest for Asia-Pacific and Europe")
# ---------------------------------------------------------------------------
# Languages considered *outside* Asia-Pacific and Europe (for Experiment 1).
OUT_OF_REGION_LANGUAGES = ["ar", "he", "es"]  # Arabic, Hebrew, Spanish (Latin America)

# Default single language for Experiment 1 (out-of-region).
DEFAULT_OUT_OF_REGION_LANG = "ar"

# Asia-Pacific and Europe languages (for Experiment 2).
ASIA_PACIFIC_LANGUAGES = ["ja", "ko", "zh_cn", "zh_tw", "th", "vi", "km", "ms"]
EUROPE_LANGUAGES = ["de", "fr", "en", "it", "nl", "pl", "pt", "ru", "es", "tr", "da", "fi", "hu", "no", "sv"]
IN_REGION_LANGUAGES = ASIA_PACIFIC_LANGUAGES + EUROPE_LANGUAGES

# ---------------------------------------------------------------------------
# Paths (all under hallucination-rate/)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"
LOGS_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Per-experiment subdirs (created on first run)
def experiment_log_dir(experiment_name: str) -> Path:
    return LOGS_DIR / experiment_name

def experiment_results_dir(experiment_name: str) -> Path:
    return RESULTS_DIR / experiment_name

def experiment_plots_dir(experiment_name: str) -> Path:
    return PLOTS_DIR / experiment_name

# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------
TEMPERATURE = 0.3
MAX_TOKENS = 512
