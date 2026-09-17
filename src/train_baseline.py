"""
Step 2: Train the underlying prediction model (Experiment A baseline).

TF-IDF + Logistic Regression. This model's own metrics (accuracy, F1, ...)
are reported separately from confidence-drift-detector metrics, per the
brief's requirement to keep the two evaluations distinct.
"""
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from config import DATA_PROCESSED, METRICS, MODELS, RANDOM_STATE


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


def main():
    train_df = pd.read_csv(DATA_PROCESSED / "train.csv")
    test_df = pd.read_csv(DATA_PROCESSED / "test.csv")

    pipeline = build_pipeline()
    pipeline.fit(train_df["text"], train_df["label"])

    joblib.dump(pipeline, MODELS / "baseline_model.joblib")

    proba = pipeline.predict_proba(test_df["text"])
    preds = proba.argmax(axis=1)
    confidence = proba.max(axis=1)

    y_true = test_df["label"].values
    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "roc_auc": roc_auc_score(y_true, proba[:, 1]),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "classification_report": classification_report(y_true, preds, output_dict=True),
        "mean_confidence": float(confidence.mean()),
        "std_confidence": float(confidence.std()),
    }

    with open(METRICS / "baseline_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Baseline model accuracy={metrics['accuracy']:.4f} "
          f"f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}")
    print(f"Mean prediction confidence: {metrics['mean_confidence']:.4f} "
          f"(std {metrics['std_confidence']:.4f})")


if __name__ == "__main__":
    main()
