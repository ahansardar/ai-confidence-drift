"""
Step 3: Generate sequential confidence histories (Data Collection, section 5-6
of the brief).

Real confidence drift needs a *sequence* of predictions on related inputs.
Since the dataset has no natural repeated-measurement structure, each test
document is fed to the trained model multiple times at increasing levels of
controlled corruption (random word deletion + shuffling). This simulates the
brief's difficulty-increasing scenario (see its worked example: confidence
0.94 -> 0.59) without fabricating any confidence numbers - every value here
comes from an actual model inference call.

The ground-truth label is held fixed at the original document's label: we are
studying how the *same underlying sample* behaves as its signal degrades, not
relabelling it.
"""
import hashlib
import json
import random

import joblib
import pandas as pd

from config import CORRUPTION_LEVELS, DATA_PROCESSED, MODELS, RANDOM_STATE


def stable_seed(text: str) -> int:
    """Deterministic seed derived from a string, independent of Python's
    per-process hash randomization (unlike the builtin hash())."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**31)


def corrupt_text(text: str, level: float, seed: int) -> str:
    """Remove `level` fraction of words and shuffle what remains."""
    words = text.split()
    if not words or level <= 0:
        return text
    rng = random.Random(seed)
    n_keep = max(1, int(len(words) * (1 - level)))
    kept = rng.sample(words, n_keep)
    rng.shuffle(kept)
    return " ".join(kept)


def main():
    test_df = pd.read_csv(DATA_PROCESSED / "test.csv")
    pipeline = joblib.load(MODELS / "baseline_model.joblib")

    records = []
    for _, row in test_df.iterrows():
        doc_seed = stable_seed(row["input_id"])
        variants = [
            corrupt_text(row["text"], level, seed=doc_seed + i)
            for i, level in enumerate(CORRUPTION_LEVELS)
        ]
        proba = pipeline.predict_proba(variants)
        preds = proba.argmax(axis=1)
        confidences = proba.max(axis=1)

        for step, (level, pred, conf, p) in enumerate(
            zip(CORRUPTION_LEVELS, preds, confidences, proba)
        ):
            records.append({
                "input_id": row["input_id"],
                "experiment_id": "corruption_sequence_v1",
                "model_version": "tfidf_logreg_v1",
                "step": step,
                "corruption_level": level,
                "predicted_class": int(pred),
                "actual_class": int(row["label"]),
                "correct": bool(pred == row["label"]),
                "confidence": float(conf),
                "proba_class0": float(p[0]),
                "proba_class1": float(p[1]),
            })

    seq_df = pd.DataFrame(records)
    seq_df.to_csv(DATA_PROCESSED / "confidence_sequences.csv", index=False)

    summary = {
        "n_documents": test_df.shape[0],
        "steps_per_document": len(CORRUPTION_LEVELS),
        "corruption_levels": CORRUPTION_LEVELS,
        "n_total_predictions": len(seq_df),
        "accuracy_by_step": seq_df.groupby("step")["correct"].mean().to_dict(),
        "mean_confidence_by_step": seq_df.groupby("step")["confidence"].mean().to_dict(),
    }
    with open(DATA_PROCESSED / "confidence_sequences_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Generated {len(seq_df)} predictions across "
          f"{test_df.shape[0]} documents x {len(CORRUPTION_LEVELS)} steps")
    print("Mean confidence by step:", summary["mean_confidence_by_step"])
    print("Accuracy by step:", summary["accuracy_by_step"])


if __name__ == "__main__":
    main()
