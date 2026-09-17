# AI Confidence Drift Detection Using Machine Learning

15-day AI/ML research project (PranavX Labs internship brief) investigating
whether **changes in an AI model's confidence over repeated predictions can
be detected and used to identify unreliable predictions**.

Research question: *Can Machine Learning detect significant changes in an
AI model's confidence and use those changes to identify potentially
unreliable predictions?*

See [`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md) for the full
research report and [`IMPLEMENTED.md`](IMPLEMENTED.md) for a section-by-section
status against the project brief.

## Approach

1. Train a real text classifier (TF-IDF + Logistic Regression) on a public,
   documented dataset (20 Newsgroups, `alt.atheism` vs `soc.religion.christian`).
2. Generate genuine confidence **sequences** by feeding each test document to
   the model repeatedly at increasing levels of controlled corruption (word
   deletion + shuffling) — every confidence value is a real model output, none
   are fabricated or LLM-assigned.
3. Engineer statistical features from these sequences (moving average,
   moving std, deltas, consecutive decreases, volatility, margin, historical
   per-class accuracy, ...).
4. Train a **confidence-drift detector** — a second ML model that predicts
   whether a given prediction is unreliable (incorrect) — and compare it
   under three controlled feature sets (Experiments A/B/C) using an identical
   document-level train/test split.
5. Run calibration analysis (ECE, Brier score, reliability diagram) and a
   dedicated high-confidence-error analysis.

## Project structure

```
src/
  config.py               shared paths/constants
  data_prep.py             1. data collection
  train_baseline.py        2. base classifier + its own metrics
  drift_simulation.py      3. confidence-sequence generation
  feature_engineering.py   4. drift feature engineering
  confidence_analysis.py   5. confidence-vs-accuracy binning, trend stats
  calibration.py           6. ECE / Brier score / reliability diagram
  error_analysis.py        7. high-confidence-error analysis
  drift_detector.py        8. Experiments A/B/C, drift-detector training
  visualize.py              9. report figures
  run_pipeline.py          runs all of the above in order
data/processed/            generated datasets (committed, reproducible)
models/                    trained baseline model + drift detector (joblib)
results/metrics/           every metric reported in the research report
results/figures/           reliability diagram, drift trajectories, etc.
reports/RESEARCH_REPORT.md full 18-section research report
```

`data/raw/` (the ~15 MB scikit-learn dataset cache) is not committed — it is
re-downloaded automatically the first time the pipeline runs.

## Running it

```bash
pip install -r requirements.txt
cd src
python run_pipeline.py
```

Everything is seeded (`RANDOM_STATE = 42` in `src/config.py`, plus a
SHA256-based deterministic per-document seed for the corruption sequences) —
re-running the pipeline reproduces identical metrics.

## Key findings (see the report for full detail and caveats)

- Base classifier: 81.1% accuracy, F1 0.837, ROC-AUC 0.892 on held-out data.
- Confidence is informative but imperfectly calibrated: ECE = 0.153, Brier
  score = 0.125. Accuracy rises from ~63% in the 40–60% confidence bin to
  ~93% in the 80–100% bin — confidence and correctness are correlated, but
  not perfectly (the very top bin does not monotonically have the highest
  accuracy on this small dataset).
- 6% of high-confidence (≥0.75) predictions are still wrong — including a
  96%-confidence prediction that was incorrect.
- Confidence-drift features give a real, measurable improvement in detecting
  unreliable predictions over a no-confidence baseline (F1 0.25 → 0.48,
  ROC-AUC 0.51 → 0.79), but the raw "total drift" of a sequence only weakly
  correlates with whether the final prediction is wrong — the *shape* of the
  sequence (volatility, moving stats) carries more signal than net drift
  alone.
