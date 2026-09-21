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
Done. Mean confidence, spread, range, absolute/percentage change, moving
statistics, rate of change and consecutive decreases are implemented in
`src/feature_engineering.py` and `src/confidence_analysis.py`. The latter
separately reports the correlation with becoming incorrect (0.090, 25
documents) and with any incorrect final prediction (0.266, 95 documents).

## 7. Machine Learning component
Done. A context-only baseline, a current-confidence model and a model adding
nine history features use the same document splits. Logistic Regression is
the controlled A/B/C comparison; Random Forest and Gradient Boosting are
also evaluated on C. The saved algorithm is chosen in grouped folds of the
training documents only (`results/metrics/algorithm_selection.json`).

## 8. Feature engineering
Done. C uses 13 inputs: two context, two current-confidence and nine history
features. `results/metrics/feature_ablation.json` checks three history groups
and an optional class-accuracy prior calculated from training labels only.

## 9. Confidence vs actual correctness
Done. Binned table in `results/metrics/confidence_vs_accuracy.csv` and
report §11; figure at `results/figures/confidence_vs_accuracy.png`.

## 10. High-confidence errors
Done. `src/error_analysis.py` and report §15 show that 6/100 (6.0%) of
high-confidence (≥0.75) predictions were incorrect. The detector's five-fold
out-of-fold check caught 0/6 at the uncorrupted step. This failure is
reported in `results/metrics/high_confidence_detector_analysis.json`.

## 11. Calibration analysis
Done. ECE 0.153, corrected binary Brier score 0.160 and a reliability
diagram (`src/calibration.py`, report §12).

## 12. Model evaluation
Done. Base model: accuracy/precision/recall/F1/ROC-AUC/confusion matrix in
`results/metrics/baseline_model_metrics.json`. Drift detector: same metrics
plus FPR/FNR per experiment in
`results/metrics/drift_detector_experiments.json`. Kept explicitly separate
per the brief's requirement (report §9, §15).

## 13. Experimental comparison (A/B/C)
Done. `src/drift_detector.py` and report §13 show the incremental results.
Pooled five-fold F1 is A 0.296, B 0.468, C 0.508; ROC-AUC is 0.509, 0.729,
0.763. B and C have matching context inputs, so the C-minus-B comparison
isolates added history features. The holdout gain is smaller (F1 0.447 to
0.457), and the report preserves that limitation.

## 14. 15-day work plan
The brief's 15-day schedule is a suggested work plan. This repository was
built as a continuous implementation, so it does not claim a fabricated
day-by-day timeline. The underlying deliverables are listed below.

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
| Final presentation | Done | `reports/presentation/AI_Confidence_Drift_Presentation.pptx` |

## 16. Research report structure (18 sections)
All 18 sections present in `reports/RESEARCH_REPORT.md`, plus references.

## 17. Evaluation criteria alignment
All graded categories (AI/ML fundamentals, research methodology, dataset &
experimentation, feature engineering, ML implementation, evaluation &
analysis, research documentation, final presentation) have corresponding,
verifiable artifacts in this repo as itemized above.

## 18. Important instructions compliance
- Focused on AI/ML, not automation/chatbot/UI — confirmed, no web app or
  chatbot exists in this repo.
- No LLM-assigned confidence — all confidence values come from
  `LogisticRegression.predict_proba` on real inputs.
- No fabricated datasets, results, or accuracy: reported numbers come from
  `src/run_pipeline.py` and fixed random seeds.
- Weak and unsuccessful findings are documented: drift-history improvement is
  modest, holdout performance varies, and the detector caught none of the
  six high-confidence errors in the out-of-fold check.

## Remaining / open items
- Further research: repeat the pipeline on another dataset or base model to
  test generalization. Improve detection of high-confidence errors at step 0.
