
from pathlib import Path

# ── API ────────────────────────────────────────────────────────────────────────
COHERE_API_KEY_ENV = "COHERE_API_KEY"  

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS = {
    "global": "tiny-aya-global",
    "earth":  "tiny-aya-earth",
    "water":  "tiny-aya-water",
}

# Single language list used across ALL models — keeps comparison clean
LANGUAGES = ["en", "ar", "fr", "de", "zh_cn", "ja", "ko", "tr", "he"]

IN_DISTRIBUTION = {
    "global": ["en", "ar", "fr", "de", "zh_cn", "ja", "tr", "he", "ko"],
    "earth":  ["ar", "tr", "he"],
    "water":  ["zh_cn", "ja", "ko", "fr", "de"],
}


# ── Sampling ───────────────────────────────────────────────────────────────────
SAMPLES_PER_PROMPT = 3     # sample each prompt multiple times to separate
                           # language-induced variance from random sampling noise
TEMPERATURE        = 0.7   # fixed across ALL models and languages — never vary this
MAX_TOKENS         = 150

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET_SOURCE = "mkqa"    # "mkqa" | "hallomtbench"
N_PROMPTS      = 50        # start small, scale after pipeline is verified

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = BASE_DIR / "data"
ANALYSIS_DIR = BASE_DIR / "results"
PLOTS_DIR    = BASE_DIR / "plots"

# ── Rate limiting ──────────────────────────────────────────────────────────────
REQUEST_DELAY_S = 0.5   # seconds between API calls