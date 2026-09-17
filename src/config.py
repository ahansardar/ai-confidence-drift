"""Shared paths and constants for the AI Confidence Drift project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"
REPORTS = ROOT / "reports"

for d in (DATA_RAW, DATA_PROCESSED, MODELS, RESULTS, FIGURES, METRICS, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

NEWSGROUPS_CATEGORIES = ["alt.atheism", "soc.religion.christian"]

# Corruption levels applied to each test document to generate a sequential
# confidence history per sample (0.0 = original text, 0.5 = half the words
# removed and the remainder shuffled).
CORRUPTION_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
