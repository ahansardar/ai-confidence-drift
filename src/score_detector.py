"""Score unlabeled documents with the saved six-probe detector.

Examples:
    python src/score_detector.py --text "A document to classify"
    python src/score_detector.py --input documents.jsonl --output scores.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from config import MODELS
from sequence_error_detector import score_documents


def read_documents(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"input_id": str}, keep_default_na=False)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        return pd.DataFrame(records)
    raise ValueError("input file must end in .csv or .jsonl")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="Text to score; repeat for more documents")
    source.add_argument("--input", type=Path, help="CSV or JSONL with unique input_id and text fields")
    parser.add_argument("--output", type=Path, help="Write JSONL here instead of stdout")
    parser.add_argument("--model-dir", type=Path, default=MODELS, help="Directory of trusted saved models")
    args = parser.parse_args(argv)

    try:
        documents = (
            pd.DataFrame({"input_id": [f"input_{i:05d}" for i in range(len(args.text))], "text": args.text})
            if args.text else read_documents(args.input)
        )
        base = joblib.load(args.model_dir / "baseline_model.joblib")
        detector = joblib.load(args.model_dir / "drift_detector_best.joblib")
        scored = score_documents(base, detector, documents)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))

    output = scored.to_json(orient="records", lines=True)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
