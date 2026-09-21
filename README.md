# AI Confidence Drift Detection Using Machine Learning

AI/ML research project following the PranavX Labs internship brief, investigating
whether **changes in an AI model's confidence over repeated predictions can
be detected and used to identify unreliable predictions**.

Research question: *Can Machine Learning detect significant changes in an
AI model's confidence and use those changes to identify potentially
unreliable predictions?*

See [`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md) for the full
research report, [`reports/presentation/AI_Confidence_Drift_Presentation.pptx`](reports/presentation/AI_Confidence_Drift_Presentation.pptx)
for the final presentation, and [`IMPLEMENTED.md`](IMPLEMENTED.md) for a
section-by-section status against the project brief.

## Approach

1. Train a real text classifier (TF-IDF + Logistic Regression) on a public,
   documented dataset (20 Newsgroups, `alt.atheism` vs `soc.religion.christian`).
2. Generate genuine confidence **sequences** by feeding each test document to
   the model repeatedly at increasing levels of controlled corruption (word
   deletion + shuffling) — every confidence value is a real model output, none
   are fabricated or LLM-assigned.
3. Engineer 13 model inputs from these sequences and their corruption level.
   The primary drift experiment adds nine history features to the same
   current-confidence baseline. A class-accuracy prior is tested separately
   using training labels only.
4. Train a **confidence-drift detector** — a second ML model that predicts
   whether a given prediction is unreliable (incorrect) — and compare it
   under three nested feature sets (Experiments A/B/C), with document-level
   splits and five-fold grouped validation.
5. Run calibration analysis (ECE, binary Brier score, reliability diagram)
   and check whether the detector actually catches high-confidence errors.
6. Fit a separate conservative review policy using out-of-fold predictions on
   the base-model training corpus. It routes high-confidence predictions in
   error-prone predicted classes to human review.
7. Train a sequence-level detector to judge the *original* prediction after
   six probes. Its base-model training sequences are out of fold, and it can
   score new documents without their labels.

## Project structure

```
src/
  config.py                  shared paths/constants
  data_prep.py               1. data collection
  train_baseline.py          2. base classifier + its own metrics
  drift_simulation.py        3. confidence-sequence generation
  feature_engineering.py     4. drift feature engineering
  confidence_analysis.py     5. confidence-vs-accuracy binning, trend stats
  calibration.py             6. ECE / Brier score / reliability diagram
  error_analysis.py          7. high-confidence-error analysis
  high_confidence_review.py  8. training-only review policy
  drift_detector.py          9. Experiments A/B/C, per-step detector
  sequence_error_detector.py 10. original-error detection after six probes
  visualize.py              11. report figures
  run_pipeline.py            runs all of the above in order
data/processed/              generated datasets (committed, reproducible)
models/                      trained models and review policy
results/metrics/             metrics reported in the research report
results/figures/             reliability diagram, drift trajectories, etc.
reports/RESEARCH_REPORT.md   full research report
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
SHA256-based deterministic per-document seed for the corruption sequences).
The model choice uses grouped folds on the detector training documents, so
the holdout labels do not select the saved model. Run the methodology checks
with `python -m unittest discover -s tests -v`.

## Key findings (see the report for full detail and caveats)

- Base classifier: 81.1% accuracy, F1 0.837, ROC-AUC 0.892 on held-out data.
- Confidence is informative but imperfectly calibrated: ECE = 0.153, binary
  Brier score = 0.160. Accuracy rises from 63.5% in the 40–60% confidence bin
  to 93.3% in the 80–100% bin.
- 6% of high-confidence (≥0.75) predictions are still wrong — including a
  96%-confidence prediction that was incorrect.
- On five document-grouped folds, adding current confidence to the context
  baseline raises unreliable-prediction F1 from 0.296 to 0.468. Adding nine
  history features raises it to 0.508; ROC-AUC moves from 0.729 to 0.763 for
  B versus C. The separate holdout gain is smaller (F1 0.447 to 0.457).
- Net drift correlates 0.090 with a correct prediction becoming incorrect and
  0.266 with any incorrect final prediction. The original per-step detector
  caught none of the six high-confidence errors at step 0, when no history
  exists. The new sequence-level detector flagged 6/6 after six probes and
  flagged 50 correct predictions among the 100 high-confidence cases. Its
  start-only ablation flagged 5/6 with 43 false alarms. The separate immediate
  review policy also covers 6/6, but sends 78 correct predictions to review.

`score_documents` in `src/sequence_error_detector.py` accepts a dataframe
with `input_id` and `text` and returns the
original prediction, confidence, error-risk score, and review flag. The saved
bundle is `models/drift_detector_best.joblib` (also saved as
`models/sequence_error_detector.joblib`). The earlier per-step model is kept
as `models/drift_detector_per_step.joblib` for the A/B/C experiment. The
primary detector runs six perturbed
inferences per document before returning a flag, so it is not a step-0 alert.
