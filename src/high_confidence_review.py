"""Conservative review routing for high-confidence base-model predictions.

The drift detector has no history at step 0. This separate policy learns which
predicted classes have produced high-confidence errors from out-of-fold
predictions on the base-model training corpus. It then routes every matching
high-confidence prediction to human review. It is a coverage safeguard, not a
claim that those predictions are themselves incorrect.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import DATA_PROCESSED, METRICS, MODELS, RANDOM_STATE
from train_baseline import build_pipeline

HIGH_CONFIDENCE_THRESHOLD = 0.75


def policy_from_predictions(actual, predicted, confidence, threshold=HIGH_CONFIDENCE_THRESHOLD):
    """Choose review classes using training labels and out-of-fold outputs."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    confidence = np.asarray(confidence)
    high_confidence = confidence >= threshold
    errors = high_confidence & (predicted != actual)
    classes = sorted(int(value) for value in np.unique(predicted[errors]))
    # With no observed errors, there is no evidence to exempt any class from
    # review. Keep the policy conservative in small training folds.
    if not classes:
        classes = sorted(int(value) for value in np.unique(predicted))
    return {
        "high_confidence_threshold": float(threshold),
        "review_predicted_classes": classes,
        "selection_rule": "review classes with high-confidence out-of-fold errors; if none observed, review all predicted classes",
        "n_training_high_confidence": int(high_confidence.sum()),
        "n_training_high_confidence_errors": int(errors.sum()),
        "errors_by_predicted_class": {
            str(int(label)): int((errors & (predicted == label)).sum())
            for label in np.unique(predicted)
        },
    }


def fit_review_policy(train_df, n_splits=5):
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    probabilities = cross_val_predict(
        build_pipeline(), train_df["text"], train_df["label"],
        cv=folds, method="predict_proba",
    )
    return policy_from_predictions(
        train_df["label"], probabilities.argmax(axis=1), probabilities.max(axis=1)
    )


def review_flags(predicted, confidence, policy):
    """Apply the saved policy without consulting true labels."""
    return (
        (np.asarray(confidence) >= policy["high_confidence_threshold"])
        & np.isin(np.asarray(predicted), policy["review_predicted_classes"])
    )


def evaluate_flags(actual, predicted, confidence, policy):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    confidence = np.asarray(confidence)
    high_confidence = confidence >= policy["high_confidence_threshold"]
    errors = high_confidence & (predicted != actual)
    flags = review_flags(predicted, confidence, policy)
    return {
        "n_high_confidence_predictions": int(high_confidence.sum()),
        "n_high_confidence_errors": int(errors.sum()),
        "n_errors_routed_to_review": int((errors & flags).sum()),
        "n_correct_predictions_routed_to_review": int((~errors & flags).sum()),
        "n_total_routed_to_review": int(flags.sum()),
        "error_recall": float((errors & flags).sum() / errors.sum()) if errors.any() else None,
        "review_precision": float((errors & flags).sum() / flags.sum()) if flags.any() else None,
    }


def nested_training_validation(train_df):
    """Estimate the routing cost without reusing policy-selection labels."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aggregate = []
    fold_details = []
    for train_idx, test_idx in outer.split(train_df["text"], train_df["label"]):
        inner_train = train_df.iloc[train_idx]
        validation = train_df.iloc[test_idx]
        policy = fit_review_policy(inner_train, n_splits=3)
        base = build_pipeline().fit(inner_train["text"], inner_train["label"])
        probabilities = base.predict_proba(validation["text"])
        predicted = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        flags = review_flags(predicted, confidence, policy)
        fold_details.append({
            "review_predicted_classes": policy["review_predicted_classes"],
            **evaluate_flags(validation["label"], predicted, confidence, policy),
        })
        aggregate.append(pd.DataFrame({
            "actual": validation["label"].to_numpy(),
            "predicted": predicted,
            "confidence": confidence,
            "routed": flags,
        }))
    all_predictions = pd.concat(aggregate, ignore_index=True)
    high = all_predictions["confidence"] >= HIGH_CONFIDENCE_THRESHOLD
    errors = high & (all_predictions["predicted"] != all_predictions["actual"])
    flags = all_predictions["routed"].to_numpy()
    return {
        "method": "five outer folds; class policy selected in three inner folds on outer-training documents only",
        "n_high_confidence_predictions": int(high.sum()),
        "n_high_confidence_errors": int(errors.sum()),
        "n_errors_routed_to_review": int((errors & flags).sum()),
        "n_correct_predictions_routed_to_review": int((~errors & flags).sum()),
        "n_total_routed_to_review": int(flags.sum()),
        "folds": fold_details,
    }


def main():
    train_df = pd.read_csv(DATA_PROCESSED / "train.csv")
    step0 = pd.read_csv(DATA_PROCESSED / "confidence_sequences.csv")
    step0 = step0[step0["step"] == 0]
    policy = fit_review_policy(train_df)
    nested = nested_training_validation(train_df)
    test_result = evaluate_flags(
        step0["actual_class"], step0["predicted_class"], step0["confidence"], policy,
    )
    with open(MODELS / "high_confidence_review_policy.json", "w") as stream:
        json.dump(policy, stream, indent=2)
    result = {
        "purpose": "route high-confidence predictions to review; does not predict correctness",
        "policy_source": "out-of-fold predictions on the 1,315 base-model training documents",
        "policy": policy,
        "nested_training_validation": nested,
        "base_test_set_evaluation": test_result,
        "evaluation_limit": "policy designed after inspecting this test set's detector failure; confirm on a new corpus before deployment",
    }
    with open(METRICS / "high_confidence_review_analysis.json", "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"policy": policy, "nested_training_validation": {
        key: value for key, value in nested.items() if key != "folds"
    }, "base_test_set_evaluation": test_result}, indent=2))


if __name__ == "__main__":
    main()
