"""
Step 4: Engineer statistical features from confidence sequences (sections 6
and 8 of the brief).

Every feature is derived only from information available up to and including
the current step (no leakage from future steps or from the true label), since
the whole point of the detector is to flag unreliable predictions using only
what the model itself exposes at inference time.
"""
import json

import numpy as np
import pandas as pd

from config import DATA_PROCESSED

ROLLING_WINDOW = 3


def historical_accuracy_by_class(train_seq_step0: pd.DataFrame) -> dict:
    """Global accuracy of the base model per predicted class, measured on the
    step-0 (uncorrupted) test predictions - a static, non-leaking feature
    each row can be joined against on its own predicted_class."""
    return (
        train_seq_step0.groupby("predicted_class")["correct"]
        .mean()
        .to_dict()
    )


def build_features(seq_df: pd.DataFrame) -> pd.DataFrame:
    seq_df = seq_df.sort_values(["input_id", "step"]).reset_index(drop=True)

    step0 = seq_df[seq_df["step"] == 0]
    hist_acc = historical_accuracy_by_class(step0)

    rows = []
    for input_id, group in seq_df.groupby("input_id"):
        group = group.sort_values("step").reset_index(drop=True)
        confidences = group["confidence"].values
        margins = (group["proba_class0"] - group["proba_class1"]).abs().values

        prev_conf = np.nan
        consecutive_decreases = 0
        for i, row in group.iterrows():
            conf = row["confidence"]
            window = confidences[max(0, i - ROLLING_WINDOW + 1): i + 1]

            if i == 0:
                diff = 0.0
                pct_change = 0.0
            else:
                diff = conf - prev_conf
                pct_change = diff / prev_conf if prev_conf != 0 else 0.0

            if i > 0 and conf < prev_conf:
                consecutive_decreases += 1
            elif i > 0:
                consecutive_decreases = 0

            rows.append({
                "input_id": input_id,
                "step": int(row["step"]),
                "corruption_level": row["corruption_level"],
                "confidence": conf,
                "prev_confidence": conf if i == 0 else prev_conf,
                "confidence_diff": diff,
                "confidence_pct_change": pct_change,
                "moving_avg_confidence": confidences[: i + 1].mean(),
                "moving_std_confidence": confidences[: i + 1].std() if i > 0 else 0.0,
                "rolling_std_confidence": window.std() if len(window) > 1 else 0.0,
                "confidence_range_so_far": confidences[: i + 1].max() - confidences[: i + 1].min(),
                "rate_of_change": diff / max(row["step"], 1),
                "consecutive_decreases": consecutive_decreases,
                "prediction_margin": margins[i],
                "historical_accuracy_for_class": hist_acc.get(row["predicted_class"], 0.5),
                "predicted_class": int(row["predicted_class"]),
                "actual_class": int(row["actual_class"]),
                "correct": bool(row["correct"]),
                "unreliable": int(not row["correct"]),
            })
            prev_conf = conf

    return pd.DataFrame(rows)


def main():
    seq_df = pd.read_csv(DATA_PROCESSED / "confidence_sequences.csv")
    features_df = build_features(seq_df)
    features_df.to_csv(DATA_PROCESSED / "confidence_features.csv", index=False)

    summary = {
        "n_rows": len(features_df),
        "unreliable_rate": float(features_df["unreliable"].mean()),
        "feature_columns": [
            c for c in features_df.columns
            if c not in ("input_id", "predicted_class", "actual_class", "correct", "unreliable")
        ],
    }
    with open(DATA_PROCESSED / "confidence_features_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Built {len(features_df)} feature rows, "
          f"unreliable rate = {summary['unreliable_rate']:.4f}")
    print("Feature columns:", summary["feature_columns"])


if __name__ == "__main__":
    main()
