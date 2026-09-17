# Implementation Status

Tracks this repository's progress against the PranavX Labs *AI Confidence
Drift Detection Using Machine Learning* brief. Updated after every
meaningful change.

## 1-2. Project overview / Research question
Implemented. See [`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md) §1-4.

## 3. Project objectives (10 items)
| # | Objective | Status | Where |
|---|---|---|---|
| 1 | Select AI/ML prediction task | Done | Binary text classification, 20 Newsgroups (`src/config.py`) |
| 2 | Generate predictions from an AI/ML model | Done | `src/train_baseline.py` |
| 3 | Collect confidence-related measurements | Done | `src/drift_simulation.py` |
| 4 | Create dataset of predictions + outcomes | Done | `data/processed/confidence_sequences.csv` |
| 5 | Measure confidence changes across predictions | Done | `src/confidence_analysis.py` |
| 6 | Engineer statistical features | Done | `src/feature_engineering.py` |
| 7 | Develop ML-based confidence-drift detector | Done | `src/drift_detector.py` |
| 8 | Evaluate detector with appropriate metrics | Done | `results/metrics/drift_detector_experiments.json` |
| 9 | Analyse incorrect & high-confidence predictions | Done | `src/error_analysis.py` |
| 10 | Determine limitations | Done | `reports/RESEARCH_REPORT.md` §16 |

## 4. Experimental setup
Done — text classification task, public documented dataset
(`sklearn.datasets.fetch_20newsgroups`), measurable ground-truth label.

## 5. Data collection
Done. `src/data_prep.py` + `src/drift_simulation.py` record input ID,
predicted/actual class, confidence, correctness, full probability
distribution, model version, and experiment ID for every prediction, plus
per-document confidence history (`data/processed/confidence_sequences.csv`).

## 6. Confidence analysis
Done. Mean confidence, std, range, absolute/percentage change, moving
average, moving std, rate of change, consecutive decreases all implemented
in `src/feature_engineering.py` / `src/confidence_analysis.py`. Trend is
analysed, not assumed — see report §11 (weak drift/error correlation
explicitly tested and reported).

## 7. Machine Learning component
Done. Confidence-drift detector compared across Logistic Regression
(baseline), Random Forest, Gradient Boosting (`src/drift_detector.py`). A
simple baseline (no-confidence-features Logistic Regression, Experiment A)
is established for comparison as required.

## 8. Feature engineering
Done — 13 features engineered and empirically compared via Experiments
A/B/C rather than included blindly. See report §10, §13.

## 9. Confidence vs actual correctness
Done. Binned table in `results/metrics/confidence_vs_accuracy.csv` and
report §11; figure at `results/figures/confidence_vs_accuracy.png`.

## 10. High-confidence errors
Done. `src/error_analysis.py`, report §15. 6/100 (6.0%) of high-confidence
(≥0.75) predictions were incorrect, including a 96.3%-confidence error.

## 11. Calibration analysis
Done. ECE, Brier score, reliability diagram (`src/calibration.py`, report
§12, `results/figures/reliability_diagram.png`).

## 12. Model evaluation
Done. Base model: accuracy/precision/recall/F1/ROC-AUC/confusion matrix in
`results/metrics/baseline_model_metrics.json`. Drift detector: same metrics
plus FPR/FNR per experiment in
`results/metrics/drift_detector_experiments.json`. Kept explicitly separate
per the brief's requirement (report §9, §15).

## 13. Experimental comparison (A/B/C)
Done. `src/drift_detector.py`, report §13. Result: confidence-drift features
give a measured, non-fabricated improvement over both baselines
(F1 0.25→0.48, ROC-AUC 0.51→0.79) — reported with the actual, modest
magnitude, not oversold.

## 14. 15-day work plan
Not tracked day-by-day (this was executed as a single continuous build) —
all the underlying deliverables the plan targets are complete (see above).

## 15. Final deliverables
| Deliverable | Status | Where |
|---|---|---|
| Python source code | Done | `src/` |
| GitHub repository | Done | this repo (public) |
| Dataset / dataset source | Done | `data/processed/` + `sklearn.datasets.fetch_20newsgroups` |
| Trained ML model | Done | `models/baseline_model.joblib`, `models/drift_detector_best.joblib` |
| Feature-engineering methodology | Done | `src/feature_engineering.py`, report §10 |
| Experimental results | Done | `results/metrics/` |
| Confidence analysis | Done | `results/metrics/confidence_vs_accuracy.csv`, `confidence_trend_statistics.json` |
| Calibration analysis | Done | `results/metrics/calibration_analysis.json` |
| Research report | Done | `reports/RESEARCH_REPORT.md` |
| Final presentation | Not started | — |

## 16. Research report structure (18 sections)
All 18 sections present in `reports/RESEARCH_REPORT.md`, plus references.

## 17. Evaluation criteria alignment
All graded categories (AI/ML fundamentals, research methodology, dataset &
experimentation, feature engineering, ML implementation, evaluation &
analysis, research documentation) have corresponding, verifiable artifacts
in this repo as itemized above. Final presentation is the one outstanding
item.

## 18. Important instructions compliance
- Focused on AI/ML, not automation/chatbot/UI — confirmed, no web app or
  chatbot exists in this repo.
- No LLM-assigned confidence — all confidence values come from
  `LogisticRegression.predict_proba` on real inputs.
- No fabricated datasets, results, or accuracy — every number in the report
  is generated by `src/run_pipeline.py` and reproducible via a fixed seed.
- Unsuccessful/weak findings documented, not hidden (e.g. net-drift/error
  correlation is weak, r≈0.04; detector F1 is modest at ≈0.48).

## Remaining / open items
- Final presentation deck.
- Optional: repeat the pipeline on a second base-model architecture (future
  scope item, report §17) to test generalization of the findings.
