
from pathlib import Path

# ── API ────────────────────────────────────────────────────────────────────────
COHERE_API_KEY_ENV = "COHERE_API_KEY"  

# ── Models ─────────────────────────────────────────────────────────────────────
# TODO: confirm exact model ID strings with Sinem once Cohere provides them
MODELS = {
    "global": "tiny-aya-global",
    "fire":   "tiny-aya-fire",
    "earth":  "tiny-aya-earth",
    "water":  "tiny-aya-water",
}

# ── Language coverage per model ────────────────────────────────────────────────
# Each variant is tested on its own target languages PLUS others it wasn't
# fine-tuned on — that cross-distribution gap is the core experimental axis.
LANGUAGES = {
    "global": ["en", "ar", "sw", "hi", "zh", "fr"],
    "earth":  ["ar", "sw", "hi", "zh", "fr"],    # in-dist: ar, sw + out-dist: hi, zh, fr
    "fire":   ["hi", "bn", "ar", "zh", "fr"],    # in-dist: hi, bn + out-dist: ar, zh, fr
    "water":  ["zh", "fr", "de", "ar", "hi"],    # in-dist: zh, fr, de + out-dist: ar, hi
}

# ── In-distribution languages per model ───────────────────────────────────────
IN_DISTRIBUTION = {
    "global": ["en", "ar", "sw", "hi", "bn", "zh", "fr", "de"],
    "earth":  ["ar", "sw"],
    "fire":   ["hi", "bn"],
    "water":  ["zh", "fr", "de"],
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
ANALYSIS_DIR = BASE_DIR / "analysis"
PLOTS_DIR    = BASE_DIR / "plots"

# ── Rate limiting ──────────────────────────────────────────────────────────────
REQUEST_DELAY_S = 0.5   # seconds between API calls