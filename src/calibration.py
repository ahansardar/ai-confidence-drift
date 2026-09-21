"""
Step 6: Calibration analysis (section 11 of the brief).

Expected Calibration Error, Brier score, and a reliability diagram, computed
on the step-0 (uncorrupted) test predictions.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from config import DATA_PROCESSED, FIGURES, METRICS

N_BINS = 10


def binary_brier_score(step0: pd.DataFrame) -> float:
    """Score class-1 probabilities against observed binary class labels."""
    return float(brier_score_loss(step0["actual_class"], step0["proba_class1"]))


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_records = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        weight = mask.sum() / len(confidences)
        ece += weight * abs(bin_acc - bin_conf)
        bin_records.append({
            "bin_range": f"{lo:.1f}-{hi:.1f}",
            "n": int(mask.sum()),
            "avg_confidence": float(bin_conf),
            "avg_accuracy": float(bin_acc),
        })
    return float(ece), bin_records


def plot_reliability_diagram(bin_records: list, path):
    confs = [b["avg_confidence"] for b in bin_records]
    accs = [b["avg_accuracy"] for b in bin_records]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(confs, accs, marker="o", color="#2563eb", label="Model")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability Diagram (base model, uncorrupted inputs)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    seq_df = pd.read_csv(DATA_PROCESSED / "confidence_sequences.csv")
    step0 = seq_df[seq_df["step"] == 0]

    confidences = step0["confidence"].values
    correct = step0["correct"].astype(int).values

    # Binary Brier score compares the probability of class 1 with the actual
    # class 0/1. Pairing correctness with the probability of the actual class
    # understates the penalty for incorrect, confident predictions.
    brier = binary_brier_score(step0)

    ece, bin_records = expected_calibration_error(confidences, correct)
    plot_reliability_diagram(bin_records, FIGURES / "reliability_diagram.png")

    results = {
        "expected_calibration_error": ece,
        "brier_score": float(brier),
        "n_bins": N_BINS,
        "bins": bin_records,
    }
    with open(METRICS / "calibration_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"ECE = {ece:.4f}, Brier score = {brier:.4f}")


if __name__ == "__main__":
    main()
