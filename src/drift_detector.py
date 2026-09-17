"""
Steps 8-9: Confidence-drift detector + controlled experimental comparison
(sections 7, 12, 13 of the brief).

The task: given the features available at a prediction step, classify
whether that prediction is UNRELIABLE (i.e. incorrect). Three feature sets
are compared under an identical train/test split and identical models:

  Experiment A - no confidence-based features at all (just step index /
                 corruption level: the naive "trust every prediction" world).
  Experiment B - single-point confidence features (current confidence +
                 prediction margin), no history.
  Experiment C - full confidence-drift features (moving average, moving std,
                 deltas, consecutive decreases, volatility, historical
                 per-class accuracy, ...).

Splitting is done at the document level (GroupShuffleSplit) so that steps
belonging to the same underlying document never appear in both train and
test - otherwise the drift detector could trivially memorise a document's
outcome from a leaked sibling row.
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from config import DATA_PROCESSED, METRICS, MODELS, RANDOM_STATE

EXPERIMENTS = {
    "A_no_confidence_features": ["step", "corruption_level"],
    "B_single_confidence_features": ["confidence", "prediction_margin"],
    "C_confidence_drift_features": [
        "confidence", "prev_confidence", "confidence_diff", "confidence_pct_change",
        "moving_avg_confidence", "moving_std_confidence", "rolling_std_confidence",
        "confidence_range_so_far", "rate_of_change", "consecutive_decreases",
        "prediction_margin", "historical_accuracy_for_class", "step", "corruption_level",
    ],
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
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": cm.tolist(),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def main():
    df = pd.read_csv(DATA_PROCESSED / "confidence_features.csv")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df["input_id"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    print(f"Train rows: {len(train_df)} ({train_df['input_id'].nunique()} docs), "
          f"Test rows: {len(test_df)} ({test_df['input_id'].nunique()} docs)")

    results = {}
    for exp_name, feature_cols in EXPERIMENTS.items():
        results[exp_name] = {}
        X_train, X_test = train_df[feature_cols], test_df[feature_cols]
        y_train, y_test = train_df["unreliable"], test_df["unreliable"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        algos = ALGORITHMS if exp_name == "C_confidence_drift_features" else {
            "logistic_regression": ALGORITHMS["logistic_regression"]
        }

        for algo_name, algo_fn in algos.items():
            model = algo_fn()
            if algo_name == "gradient_boosting":
                class_counts = y_train.value_counts()
                weight_per_class = (len(y_train) / (2 * class_counts)).to_dict()
                sample_weight = y_train.map(weight_per_class).values
                model.fit(X_train_s, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_proba = model.predict_proba(X_test_s)[:, 1]
            metrics = evaluate(y_test, y_pred, y_proba)
            results[exp_name][algo_name] = metrics
            print(f"[{exp_name} / {algo_name}] "
                  f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
                  f"roc_auc={metrics['roc_auc']:.4f}")

            if exp_name == "C_confidence_drift_features" and algo_name == "random_forest":
                joblib.dump(
                    {"model": model, "scaler": scaler, "features": feature_cols},
                    MODELS / "drift_detector_best.joblib",
                )

    with open(METRICS / "drift_detector_experiments.json", "w") as f:
        json.dump(results, f, indent=2)

    best_c = max(results["C_confidence_drift_features"].items(), key=lambda kv: kv[1]["f1"])
    baseline_a = results["A_no_confidence_features"]["logistic_regression"]
    improvement = {
        "baseline_A_f1": baseline_a["f1"],
        "best_C_algorithm": best_c[0],
        "best_C_f1": best_c[1]["f1"],
        "f1_improvement_over_baseline": best_c[1]["f1"] - baseline_a["f1"],
        "roc_auc_improvement_over_baseline": best_c[1]["roc_auc"] - baseline_a["roc_auc"],
    }
    with open(METRICS / "experiment_comparison_summary.json", "w") as f:
        json.dump(improvement, f, indent=2)
    print(json.dumps(improvement, indent=2))


if __name__ == "__main__":
    main()
