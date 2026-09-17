"""
Step 7: High-confidence error analysis (section 10 of the brief).

Specifically examines cases of high confidence + incorrect prediction on the
step-0 (uncorrupted) test set - the cases that most starkly show the gap
between confidence and reliability.
"""
import json

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, METRICS

HIGH_CONFIDENCE_THRESHOLD = 0.75


def main():
    seq_df = pd.read_csv(DATA_PROCESSED / "confidence_sequences.csv")
    test_df = pd.read_csv(DATA_PROCESSED / "test.csv")
    step0 = seq_df[seq_df["step"] == 0].merge(
        test_df[["input_id", "text", "target_name"]], on="input_id", how="left"
    )

    high_conf = step0[step0["confidence"] >= HIGH_CONFIDENCE_THRESHOLD]
    high_conf_errors = high_conf[~high_conf["correct"]]

    seq_df_sorted = seq_df.sort_values(["input_id", "step"])
    doc_group = seq_df_sorted.groupby("input_id")
    became_unreliable = doc_group.apply(
        lambda g: (g["confidence"].iloc[0] >= HIGH_CONFIDENCE_THRESHOLD)
        and (not g["correct"].iloc[-1]),
        include_groups=False,
    )

    summary = {
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "n_high_confidence_predictions": int(len(high_conf)),
        "n_high_confidence_errors": int(len(high_conf_errors)),
        "high_confidence_error_rate": float(len(high_conf_errors) / len(high_conf)) if len(high_conf) else 0.0,
        "share_of_all_errors_that_are_high_confidence": float(
            len(high_conf_errors) / max(1, (~step0["correct"]).sum())
        ),
        "n_docs_high_conf_start_but_wrong_after_corruption": int(became_unreliable.sum()),
        "example_high_confidence_errors": high_conf_errors[
            ["input_id", "confidence", "predicted_class", "actual_class", "target_name"]
        ].head(10).to_dict(orient="records"),
    }

    with open(METRICS / "high_confidence_error_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
