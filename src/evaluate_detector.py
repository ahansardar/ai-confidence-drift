"""Evaluate the frozen detector on a labeled corpus without retraining it.

The input is CSV or JSONL with unique input_id, text, and label (0 or 1).
For a prospective check, collect and label a new corpus before inspecting
detector scores. This script marks overlap with the development corpus.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, MODELS
from score_detector import read_documents
from sequence_error_detector import high_confidence_metrics, score_documents


def text_hashes(texts):
    return {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts}


def evaluate(documents, base_model, detector_bundle):
    if "label" not in documents.columns:
        raise ValueError("evaluation input must contain a label column")
    if not documents.label.map(lambda value: str(value) in {"0", "1"}).all():
        raise ValueError("labels must be 0 or 1, matching the base model classes")
    scores = score_documents(base_model, detector_bundle, documents)
    labels = documents.label.astype(int).to_numpy()
    errors = (scores.predicted_class.to_numpy() != labels).astype(int)
    high = (scores.confidence.to_numpy() >= detector_bundle["high_confidence_threshold"])
    known = pd.concat([
        pd.read_csv(DATA_PROCESSED / "train.csv", usecols=["text"]),
        pd.read_csv(DATA_PROCESSED / "test.csv", usecols=["text"]),
    ], ignore_index=True)
    existing = text_hashes(known.text)
    overlap = sum(hashlib.sha256(text.encode("utf-8")).hexdigest() in existing for text in documents.text)
    return {
        "n_documents": len(documents),
        "n_matching_development_texts": overlap,
        "no_text_overlap_with_development_corpus": overlap == 0,
        "base_model_accuracy": float(np.mean(1 - errors)),
        "threshold": detector_bundle["threshold"],
        "score_calibrated": detector_bundle["score_calibrated"],
        "high_confidence": high_confidence_metrics(
            errors, high, scores.original_error_score.to_numpy(), detector_bundle["threshold"]
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Labeled CSV or JSONL")
    parser.add_argument("--output", type=Path, help="Write summary JSON here instead of stdout")
    parser.add_argument("--model-dir", type=Path, default=MODELS)
    args = parser.parse_args(argv)
    try:
        documents = read_documents(args.input)
        base = joblib.load(args.model_dir / "baseline_model.joblib")
        detector = joblib.load(args.model_dir / "drift_detector_best.joblib")
        result = evaluate(documents, base, detector)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
