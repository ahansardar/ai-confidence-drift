"""Train and evaluate a detector for incorrect text classifications.

All comparisons use document-level splits. A and B share the same context
features, so their difference measures current confidence. C adds only
confidence history, so its difference from B measures the drift features.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from config import DATA_PROCESSED, METRICS, MODELS, RANDOM_STATE

CONTEXT_FEATURES = ["step", "corruption_level"]
CURRENT_FEATURES = ["confidence", "prediction_margin"]
DRIFT_GROUPS = {
    "change": ["prev_confidence", "confidence_diff", "confidence_pct_change", "rate_of_change"],
    "moving_statistics": [
        "moving_avg_confidence", "moving_std_confidence",
        "rolling_std_confidence", "confidence_range_so_far",
    ],
    "consecutive_drops": ["consecutive_decreases"],
}
DRIFT_FEATURES = [column for group in DRIFT_GROUPS.values() for column in group]
EXPERIMENTS = {
    "A_no_confidence_features": CONTEXT_FEATURES,
    "B_single_confidence_features": CONTEXT_FEATURES + CURRENT_FEATURES,
    "C_confidence_drift_features": CONTEXT_FEATURES + CURRENT_FEATURES + DRIFT_FEATURES,
}
ALGORITHMS = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
}


def evaluate(y_true, y_pred, y_proba) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": cm.tolist(),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def fit_detector(train_df, test_df, feature_cols, algorithm="logistic_regression"):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols])
    x_test = scaler.transform(test_df[feature_cols])
    model = ALGORITHMS[algorithm]()
    if algorithm == "gradient_boosting":
        counts = train_df["unreliable"].value_counts()
        class_weights = (len(train_df) / (2 * counts)).to_dict()
        model.fit(x_train, train_df["unreliable"],
                  sample_weight=train_df["unreliable"].map(class_weights).to_numpy())
    else:
        model.fit(x_train, train_df["unreliable"])
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = model.predict(x_test)
    return evaluate(test_df["unreliable"], predictions, probabilities), {
        "model": model, "scaler": scaler, "features": list(feature_cols),
        "algorithm": algorithm, "threshold": 0.5,
    }, probabilities


def add_training_only_class_accuracy(train_df, test_df):
    """Fit the optional class-accuracy prior on training documents only."""
    train_step0 = train_df[train_df["step"] == 0]
    by_class = train_step0.groupby("predicted_class")["correct"].mean().to_dict()
    fallback = float(train_step0["correct"].mean())
    train_copy, test_copy = train_df.copy(), test_df.copy()
    for frame in (train_copy, test_copy):
        frame["historical_accuracy_for_class"] = (
            frame["predicted_class"].map(by_class).fillna(fallback)
        )
    return train_copy, test_copy, {int(key): float(value) for key, value in by_class.items()}


def class_prior_and_group_ablation(train_df, test_df, full_c_metrics):
    results = {"full_C": full_c_metrics, "removed_groups": {}}
    for group_name, group_cols in DRIFT_GROUPS.items():
        retained = [column for column in EXPERIMENTS["C_confidence_drift_features"]
                    if column not in group_cols]
        metrics, _, _ = fit_detector(train_df, test_df, retained)
        results["removed_groups"][group_name] = metrics
    train_with_prior, test_with_prior, prior = add_training_only_class_accuracy(train_df, test_df)
    metrics, _, _ = fit_detector(
        train_with_prior, test_with_prior,
        EXPERIMENTS["C_confidence_drift_features"] + ["historical_accuracy_for_class"],
    )
    results["optional_training_only_class_accuracy"] = {
        "metrics": metrics, "train_step0_accuracy_by_predicted_class": prior,
    }
    return results


def select_algorithm_on_training(train_df):
    """Choose the saved C model without consulting the final holdout set."""
    selection = {}
    for algorithm in ALGORITHMS:
        fold_f1 = []
        for train_idx, validation_idx in GroupKFold(n_splits=3).split(
            train_df, groups=train_df["input_id"]
        ):
            metrics, _, _ = fit_detector(
                train_df.iloc[train_idx], train_df.iloc[validation_idx],
                EXPERIMENTS["C_confidence_drift_features"], algorithm,
            )
            fold_f1.append(metrics["f1"])
        selection[algorithm] = {
            "fold_f1": fold_f1,
            "mean_f1": float(np.mean(fold_f1)),
        }
    selected = max(selection, key=lambda name: selection[name]["mean_f1"])
    return selected, {
        "method": "three-fold document-grouped cross-validation on the training documents only",
        "selected_algorithm": selected,
        "algorithms": selection,
    }


def grouped_cross_validation(df):
    """Make one out-of-fold prediction per row, then inspect rare errors."""
    predictions = {name: np.empty(len(df)) for name in EXPERIMENTS}
    fold_metrics = {name: [] for name in EXPERIMENTS}
    for train_idx, test_idx in GroupKFold(n_splits=5).split(df, groups=df["input_id"]):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        for name, features in EXPERIMENTS.items():
            metrics, _, probabilities = fit_detector(train_df, test_df, features)
            predictions[name][test_idx] = probabilities
            fold_metrics[name].append(metrics)
    pooled_metrics = {
        name: evaluate(df["unreliable"], probabilities >= 0.5, probabilities)
        for name, probabilities in predictions.items()
    }
    c_probabilities = predictions["C_confidence_drift_features"]
    high_confidence = (df["step"].to_numpy() == 0) & (df["confidence"].to_numpy() >= 0.75)
    errors = df["unreliable"].to_numpy().astype(bool)
    flagged = c_probabilities >= 0.5
    high_confidence_results = {
        "evaluation": "five-fold document-grouped out-of-fold predictions on uncorrupted inputs",
        "threshold": 0.75,
        "n_predictions": int(high_confidence.sum()),
        "n_errors": int((high_confidence & errors).sum()),
        "n_errors_caught": int((high_confidence & errors & flagged).sum()),
        "n_false_alarms": int((high_confidence & ~errors & flagged).sum()),
        "n_total_flagged": int((high_confidence & flagged).sum()),
    }
    rng = np.random.default_rng(RANDOM_STATE)
    document_ids = df["input_id"].unique()
    row_indexes = {doc_id: np.flatnonzero(df["input_id"].to_numpy() == doc_id)
                   for doc_id in document_ids}
    target = df["unreliable"].to_numpy()
    b_probabilities = predictions["B_single_confidence_features"]
    f1_deltas, auc_deltas = [], []
    for _ in range(1000):
        sampled_ids = rng.choice(document_ids, size=len(document_ids), replace=True)
        rows = np.concatenate([row_indexes[doc_id] for doc_id in sampled_ids])
        sample_target = target[rows]
        if len(np.unique(sample_target)) < 2:
            continue
        f1_deltas.append(
            f1_score(sample_target, c_probabilities[rows] >= 0.5)
            - f1_score(sample_target, b_probabilities[rows] >= 0.5)
        )
        auc_deltas.append(
            roc_auc_score(sample_target, c_probabilities[rows])
            - roc_auc_score(sample_target, b_probabilities[rows])
        )
    uncertainty = {
        "method": "1000 paired document bootstrap resamples of out-of-fold predictions; training uncertainty excluded",
        "C_minus_B_f1_interval_95pct": np.quantile(f1_deltas, [0.025, 0.975]).tolist(),
        "C_minus_B_roc_auc_interval_95pct": np.quantile(auc_deltas, [0.025, 0.975]).tolist(),
    }
    return {"n_folds": 5, "fold_metrics": fold_metrics,
            "pooled_out_of_fold_metrics": pooled_metrics,
            "uncertainty": uncertainty}, high_confidence_results


def main():
    df = pd.read_csv(DATA_PROCESSED / "confidence_features.csv")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df["input_id"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    assert set(train_df["input_id"]).isdisjoint(test_df["input_id"])
    print(f"Detector split: {len(train_df)} training rows from {train_df['input_id'].nunique()} docs; "
          f"{len(test_df)} test rows from {test_df['input_id'].nunique()} docs")

    results, c_bundles = {}, {}
    for name, feature_cols in EXPERIMENTS.items():
        results[name] = {}
        algorithms = ALGORITHMS if name == "C_confidence_drift_features" else {
            "logistic_regression": ALGORITHMS["logistic_regression"]
        }
        for algorithm in algorithms:
            metrics, bundle, _ = fit_detector(train_df, test_df, feature_cols, algorithm)
            results[name][algorithm] = metrics
            if name == "C_confidence_drift_features":
                c_bundles[algorithm] = bundle
            print(f"{name} / {algorithm}: F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

    c_results = results["C_confidence_drift_features"]
    selected_algorithm, algorithm_selection = select_algorithm_on_training(train_df)
    best_bundle = c_bundles[selected_algorithm]
    best_bundle["evaluation_split"] = "GroupShuffleSplit(test_size=0.25, random_state=42)"
    best_bundle["selection_method"] = algorithm_selection["method"]
    joblib.dump(best_bundle, MODELS / "drift_detector_per_step.joblib")

    a = results["A_no_confidence_features"]["logistic_regression"]
    b = results["B_single_confidence_features"]["logistic_regression"]
    c = c_results["logistic_regression"]
    summary = {
        "comparison": "same task and document split; B adds only current confidence to A; C adds only history to B",
        "best_C_algorithm_by_holdout_f1": max(c_results, key=lambda name: c_results[name]["f1"]),
        "saved_model_algorithm_selected_on_training": selected_algorithm,
        "A_to_B_f1_delta": b["f1"] - a["f1"],
        "B_to_C_f1_delta": c["f1"] - b["f1"],
        "A_to_B_roc_auc_delta": b["roc_auc"] - a["roc_auc"],
        "B_to_C_roc_auc_delta": c["roc_auc"] - b["roc_auc"],
    }
    ablation = class_prior_and_group_ablation(train_df, test_df, c)
    cross_validation, high_confidence = grouped_cross_validation(df)
    for filename, data in (
        ("drift_detector_experiments.json", results),
        ("experiment_comparison_summary.json", summary),
        ("algorithm_selection.json", algorithm_selection),
        ("feature_ablation.json", ablation),
        ("drift_history_validation.json", cross_validation),
        ("high_confidence_detector_analysis.json", high_confidence),
    ):
        with open(METRICS / filename, "w") as stream:
            json.dump(data, stream, indent=2)
    print(json.dumps(summary, indent=2))
    print(json.dumps(high_confidence, indent=2))


if __name__ == "__main__":
    main()
