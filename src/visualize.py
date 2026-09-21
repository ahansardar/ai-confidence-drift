"""Step 10: Generate figures used in the research report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_PROCESSED, FIGURES, METRICS


def plot_confidence_vs_accuracy():
    table = pd.read_csv(METRICS / "confidence_vs_accuracy.csv")
    table = table.dropna()
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(table["confidence_bin"], table["n_predictions"], color="#93c5fd", label="# predictions")
    ax1.set_ylabel("Number of predictions")
    ax2 = ax1.twinx()
    ax2.plot(table["confidence_bin"], table["accuracy"], color="#dc2626", marker="o", linewidth=2, label="Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax1.set_xlabel("Confidence bin")
    ax1.set_title("Confidence vs Observed Accuracy (step-0 predictions)")
    fig.tight_layout()
    fig.savefig(FIGURES / "confidence_vs_accuracy.png", dpi=150)
    plt.close(fig)


def plot_example_drift_trajectories(n_examples: int = 12):
    seq_df = pd.read_csv(DATA_PROCESSED / "confidence_sequences.csv")
    doc_ids = seq_df["input_id"].unique()[:n_examples]

    fig, ax = plt.subplots(figsize=(7, 5))
    for doc_id in doc_ids:
        doc = seq_df[seq_df["input_id"] == doc_id].sort_values("step")
        color = "#dc2626" if not doc["correct"].iloc[-1] else "#16a34a"
        ax.plot(doc["corruption_level"], doc["confidence"], marker="o", alpha=0.6, color=color, linewidth=1)
    ax.set_xlabel("Corruption level")
    ax.set_ylabel("Confidence")
    ax.set_title("Example confidence-drift trajectories\n(green = correct at final step, red = incorrect)")
    fig.tight_layout()
    fig.savefig(FIGURES / "example_drift_trajectories.png", dpi=150)
    plt.close(fig)


def plot_experiment_comparison():
    import json
    with open(METRICS / "drift_detector_experiments.json") as f:
        results = json.load(f)

    labels, f1s, aucs = [], [], []
    for exp, algos in results.items():
        for algo, metrics in algos.items():
            labels.append(f"{exp.split('_')[0]}\n{algo}")
            f1s.append(metrics["f1"])
            aucs.append(metrics["roc_auc"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], f1s, width=0.4, label="F1", color="#2563eb")
    ax.bar([i + 0.2 for i in x], aucs, width=0.4, label="ROC-AUC", color="#f59e0b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Detector comparison on the 110-document holdout")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "experiment_comparison.png", dpi=150)
    plt.close(fig)


def main():
    plot_confidence_vs_accuracy()
    plot_example_drift_trajectories()
    plot_experiment_comparison()
    print("Figures written to", FIGURES)


if __name__ == "__main__":
    main()
