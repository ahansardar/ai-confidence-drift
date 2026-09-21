"""Predict whether the original classification was wrong after six probes.

The per-step drift detector predicts correctness at each corruption level. It
cannot use a history at step 0. This detector asks a different question: after
collecting the six-step confidence sequence, was the original prediction
incorrect? Its training sequences come from base classifiers that did not see
the document they predict.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import CORRUPTION_LEVELS, DATA_PROCESSED, METRICS, MODELS, RANDOM_STATE
from drift_simulation import corrupt_text, stable_seed
from feature_engineering import build_features
from train_baseline import build_pipeline

HIGH_CONFIDENCE_THRESHOLD = 0.75
THRESHOLD_MARGIN = 0.75
START_FEATURES = ["start_confidence", "start_predicted_class", "word_count", "char_count"]


def predict_sequences(base_model, documents):
    """Probe each document with the same deterministic corruption schedule."""
    variants, metadata = [], []
    labeled = "label" in documents.columns
    for row in documents.itertuples():
        # The probes must depend on the document, not on a caller-assigned ID.
        # This also makes repeated scoring of the same text reproducible.
        seed = stable_seed(row.text)
        for step, level in enumerate(CORRUPTION_LEVELS):
            variants.append(corrupt_text(row.text, level, seed + step))
            metadata.append((row.input_id, int(row.label) if labeled else None, step, level))
    probabilities = base_model.predict_proba(variants)
    records = []
    for (doc, actual, step, level), probability in zip(metadata, probabilities):
        predicted = int(probability.argmax())
        record = {
            "input_id": doc, "step": step, "corruption_level": level,
            "predicted_class": predicted, "confidence": float(probability.max()),
            "proba_class0": float(probability[0]),
            "proba_class1": float(probability[1]),
        }
        if labeled:
            record.update({"actual_class": actual, "correct": predicted == actual})
        records.append(record)
    return pd.DataFrame(records)


def out_of_fold_sequences(documents, n_splits=5):
    """Build confidence histories without training the base model on each doc."""
    all_sequences = []
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for fit_idx, predict_idx in folds.split(documents.text, documents.label):
        model = build_pipeline().fit(
            documents.text.iloc[fit_idx], documents.label.iloc[fit_idx]
        )
        all_sequences.append(predict_sequences(model, documents.iloc[predict_idx]))
    return pd.concat(all_sequences, ignore_index=True)


def document_features(sequence_features, documents):
    """One feature vector and original-error target per complete sequence."""
    first = sequence_features[sequence_features.step == 0].set_index("input_id")
    last = sequence_features[sequence_features.step == len(CORRUPTION_LEVELS) - 1].set_index("input_id")
    frame = last.drop(columns=["actual_class", "correct", "unreliable"], errors="ignore").copy()
    frame["start_confidence"] = first.confidence
    frame["start_predicted_class"] = first.predicted_class
    frame["end_predicted_class"] = last.predicted_class
    frame["class_flips"] = sequence_features.sort_values("step").groupby("input_id").predicted_class.apply(
        lambda predictions: int((predictions.diff().fillna(0) != 0).sum())
    )
    text = documents.set_index("input_id").loc[frame.index].text
    frame["word_count"] = text.str.split().str.len()
    frame["char_count"] = text.str.len()
    frame["confidence_slope"] = (
        last.confidence - first.confidence
    ) / (len(CORRUPTION_LEVELS) - 1)
    frame["low_confidence"] = sequence_features.groupby("input_id").confidence.min()
    frame["high_confidence"] = sequence_features.groupby("input_id").confidence.max()
    numeric = frame.select_dtypes(include="number")
    target = (
        first.loc[numeric.index, "unreliable"].to_numpy().astype(int)
        if "unreliable" in first else None
    )
    high_confidence = (first.loc[numeric.index, "confidence"] >= HIGH_CONFIDENCE_THRESHOLD).to_numpy()
    return numeric, target, high_confidence


def score_documents(base_model, detector_bundle, documents):
    """Score new input_id/text rows without using or requiring true labels."""
    if not isinstance(documents, pd.DataFrame):
        raise TypeError("documents must be a pandas DataFrame")
    if not {"input_id", "text"}.issubset(documents.columns):
        raise ValueError("documents must contain input_id and text columns")
    if not documents.columns.is_unique:
        raise ValueError("documents must not contain duplicate column names")
    if documents.empty:
        raise ValueError("documents must contain at least one row")
    unlabeled = documents[["input_id", "text"]].copy()
    if not unlabeled.input_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("every input_id must be a nonempty string")
    if not unlabeled.input_id.is_unique:
        raise ValueError("input_id values must be unique")
    if not unlabeled.text.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("every text must be a nonempty string")
    if list(base_model.classes_) != [0, 1]:
        raise ValueError("base model must predict binary classes [0, 1]")
    if detector_bundle.get("required_corruption_levels") != CORRUPTION_LEVELS:
        raise ValueError("detector bundle requires a different probe schedule")
    if detector_bundle.get("probe_seed_method") != "sha256_text":
        raise ValueError("detector bundle requires a different probe seed method")
    sequences = predict_sequences(base_model, unlabeled)
    features = build_features(sequences)
    values, _, high_confidence = document_features(features, unlabeled)
    scores = detector_bundle["model"].predict_proba(
        values[detector_bundle["features"]]
    )[:, 1]
    initial = sequences[sequences.step == 0].set_index("input_id").loc[values.index]
    result = pd.DataFrame({
        "input_id": values.index,
        "predicted_class": initial.predicted_class.to_numpy(),
        "confidence": initial.confidence.to_numpy(),
        "original_error_score": scores,
        "flagged": high_confidence & (scores >= detector_bundle["threshold"]),
    }).reset_index(drop=True)
    return result.set_index("input_id").loc[unlabeled.input_id].reset_index()


def new_detector():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
    )


def training_threshold(features, target, high_confidence, n_splits=5):
    """Use only training folds to set a high-recall alert threshold."""
    scores = np.empty(len(target))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for fit_idx, predict_idx in folds.split(features, target):
        model = new_detector().fit(features.iloc[fit_idx], target[fit_idx])
        scores[predict_idx] = model.predict_proba(features.iloc[predict_idx])[:, 1]
    high_confidence_errors = high_confidence & target.astype(bool)
    if not high_confidence_errors.any():
        # With no high-confidence training errors there is no evidence for a
        # high-recall threshold. Never silently route every case to review.
        return 1.0, scores
    # Sparse positives make the least-risky observed error important. The
    # margin favors recall while keeping the threshold based on training data.
    return float(THRESHOLD_MARGIN * scores[high_confidence_errors].min()), scores


def high_confidence_metrics(target, high_confidence, scores, threshold):
    actual_errors = target[high_confidence].astype(bool)
    flagged = scores[high_confidence] >= threshold
    return {
        "n_high_confidence_predictions": int(high_confidence.sum()),
        "n_high_confidence_errors": int(actual_errors.sum()),
        "n_errors_caught": int((actual_errors & flagged).sum()),
        "n_correct_flagged": int((~actual_errors & flagged).sum()),
        "n_total_flagged": int(flagged.sum()),
        "error_recall": float((actual_errors & flagged).sum() / actual_errors.sum())
        if actual_errors.any() else None,
        "alert_precision": float((actual_errors & flagged).sum() / flagged.sum())
        if flagged.any() else None,
        "roc_auc": float(roc_auc_score(actual_errors, scores[high_confidence]))
        if len(np.unique(actual_errors)) == 2 else None,
    }


def nested_training_validation(documents):
    """Keep each validation document out of both base and detector fitting."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    for fit_idx, test_idx in outer.split(documents.text, documents.label):
        fit_docs = documents.iloc[fit_idx]
        test_docs = documents.iloc[test_idx]
        fit_features = build_features(out_of_fold_sequences(fit_docs, n_splits=3))
        x_fit, y_fit, high_fit = document_features(fit_features, fit_docs)
        threshold, _ = training_threshold(x_fit, y_fit, high_fit, n_splits=3)
        detector = new_detector().fit(x_fit, y_fit)
        base = build_pipeline().fit(fit_docs.text, fit_docs.label)
        test_features = build_features(predict_sequences(base, test_docs))
        x_test, y_test, high_test = document_features(test_features, test_docs)
        scores = detector.predict_proba(x_test[x_fit.columns])[:, 1]
        fold_results.append({
            "threshold": threshold,
            **high_confidence_metrics(y_test, high_test, scores, threshold),
        })
    totals = {
        key: sum(fold[key] for fold in fold_results)
        for key in ("n_high_confidence_predictions", "n_high_confidence_errors",
                    "n_errors_caught", "n_correct_flagged", "n_total_flagged")
    }
    return {
        "method": "five outer folds; base and detector models and threshold trained without each outer validation document",
        "totals": totals,
        "folds": fold_results,
    }


def main():
    train_docs = pd.read_csv(DATA_PROCESSED / "train.csv")
    test_docs = pd.read_csv(DATA_PROCESSED / "test.csv")
    train_features = build_features(out_of_fold_sequences(train_docs))
    x_train, y_train, high_train = document_features(train_features, train_docs)
    threshold, training_scores = training_threshold(x_train, y_train, high_train)
    model = new_detector().fit(x_train, y_train)
    bundle = {
        "model": model, "features": list(x_train.columns),
        "threshold": threshold, "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "required_corruption_levels": CORRUPTION_LEVELS,
        "probe_seed_method": "sha256_text",
        "score_calibrated": False,
        "target": "original step-0 prediction incorrect after six probes",
    }
    joblib.dump(bundle, MODELS / "sequence_error_detector.joblib")
    joblib.dump(bundle, MODELS / "drift_detector_best.joblib")

    # Recreate the test probes with the same content-derived seed used in
    # training and live scoring. The historical per-step experiment keeps its
    # own ID-seeded sequences in confidence_features.csv.
    base_model = joblib.load(MODELS / "baseline_model.joblib")
    test_features = build_features(predict_sequences(base_model, test_docs))
    x_test, y_test, high_test = document_features(test_features, test_docs)
    scores = model.predict_proba(x_test[x_train.columns])[:, 1]
    test_metrics = high_confidence_metrics(y_test, high_test, scores, threshold)
    test_metrics["all_document_roc_auc"] = float(roc_auc_score(y_test, scores))
    test_metrics["all_document_f1_at_0_5"] = float(f1_score(y_test, scores >= 0.5))

    start_threshold, _ = training_threshold(
        x_train[START_FEATURES], y_train, high_train
    )
    start_model = new_detector().fit(x_train[START_FEATURES], y_train)
    start_scores = start_model.predict_proba(x_test[START_FEATURES])[:, 1]
    start_metrics = high_confidence_metrics(y_test, high_test, start_scores, start_threshold)

    nested = nested_training_validation(train_docs)
    result = {
        "method": "OOF base-model sequences on training documents; six probes per document; Logistic Regression predicts original-step error",
        "threshold_method": "75% of the lowest high-confidence-error score in five meta-model training folds; no alerts if none observed",
        "threshold": threshold,
        "training_high_confidence_errors": int((high_train & y_train.astype(bool)).sum()),
        "training_at_threshold": high_confidence_metrics(y_train, high_train, training_scores, threshold),
        "nested_training_validation": nested,
        "base_test_set_evaluation": test_metrics,
        "start_only_ablation": start_metrics,
        "evaluation_limit": "developed after inspecting the six test errors; the same corpus cannot establish prospective performance",
    }
    with open(METRICS / "sequence_error_detector_analysis.json", "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({
        "threshold": threshold,
        "nested_training_validation_totals": nested["totals"],
        "base_test_set_evaluation": test_metrics,
        "start_only_ablation": start_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
