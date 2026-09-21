"""
Step 5: Confidence vs actual correctness analysis (section 9 of the brief),
plus the confidence-trend statistics from section 6.

Uses the step-0 (uncorrupted) predictions, i.e. what the model actually
produces in normal deployment, for the confidence-vs-accuracy binning.
The full corruption sequence is used for the trend statistics.
"""
import json

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, METRICS

BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


def confidence_vs_accuracy(step0: pd.DataFrame) -> pd.DataFrame:
    step0 = step0.copy()
    step0["confidence_bin"] = pd.cut(step0["confidence"], bins=BINS, labels=BIN_LABELS, include_lowest=True)
    table = step0.groupby("confidence_bin", observed=False).agg(
        n_predictions=("correct", "size"),
        accuracy=("correct", "mean"),
    ).reset_index()
    return table


def trend_statistics(seq_df: pd.DataFrame) -> dict:
    per_doc = seq_df.sort_values(["input_id", "step"]).groupby("input_id")
    first = per_doc.first()
    last = per_doc.last()
    start_confidence = first["confidence"]
    end_confidence = last["confidence"]
    started_correct = first["correct"].astype(bool)
    ended_correct = last["correct"].astype(bool)
    delta = end_confidence - start_confidence
    return {
        "n_documents": int(delta.shape[0]),
        "mean_confidence_start": float(start_confidence.mean()),
        "mean_confidence_end": float(end_confidence.mean()),
        "mean_total_drift": float(delta.mean()),
        "pct_documents_confidence_decreased": float((delta < 0).mean()),
        "pct_documents_confidence_increased": float((delta > 0).mean()),
        "pct_documents_confidence_unchanged": float((delta == 0).mean()),
        "n_became_incorrect": int((started_correct & ~ended_correct).sum()),
        "n_final_incorrect": int((~ended_correct).sum()),
        "corr_drift_vs_became_incorrect": float(
            np.corrcoef(delta, (started_correct & ~ended_correct).astype(int))[0, 1]
        ),
        "corr_drift_vs_final_incorrect": float(
            np.corrcoef(delta, (~ended_correct).astype(int))[0, 1]
        ),
    }


def main():
    seq_df = pd.read_csv(DATA_PROCESSED / "confidence_sequences.csv")
    step0 = seq_df[seq_df["step"] == 0]

    conf_acc_table = confidence_vs_accuracy(step0)
    conf_acc_table.to_csv(METRICS / "confidence_vs_accuracy.csv", index=False)

    trend = trend_statistics(seq_df)
    with open(METRICS / "confidence_trend_statistics.json", "w") as f:
        json.dump(trend, f, indent=2)

    print(conf_acc_table.to_string(index=False))
    print(json.dumps(trend, indent=2))


if __name__ == "__main__":
    main()
