"""
Step 4: Engineer statistical features from confidence sequences (sections 6
and 8 of the brief).

Every model input here uses information available by the current step. Ground
truth is retained only to construct the target and evaluate the detector.
Class-level accuracy, which needs ground truth, is investigated separately in
drift_detector.py after the document-level train/test split.
"""
import json

import numpy as np
import pandas as pd

from config import DATA_PROCESSED

ROLLING_WINDOW = 3


def build_features(seq_df: pd.DataFrame) -> pd.DataFrame:
    seq_df = seq_df.sort_values(["input_id", "step"]).reset_index(drop=True)
    labeled = {"actual_class", "correct"}.issubset(seq_df.columns)

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

            features = {
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
                "predicted_class": int(row["predicted_class"]),
            }
            if labeled:
                features.update({
                    "actual_class": int(row["actual_class"]),
                    "correct": bool(row["correct"]),
                    "unreliable": int(not row["correct"]),
                })
            rows.append(features)
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
