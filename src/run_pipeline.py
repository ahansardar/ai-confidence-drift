"""Runs the full AI Confidence Drift pipeline end-to-end, in order."""
import time

import calibration
import confidence_analysis
import data_prep
import drift_detector
import drift_simulation
import error_analysis
import feature_engineering
import high_confidence_review
import sequence_error_detector
import train_baseline
import visualize

STEPS = [
    ("1. Data collection", data_prep.main),
    ("2. Train baseline model", train_baseline.main),
    ("3. Generate confidence-drift sequences", drift_simulation.main),
    ("4. Feature engineering", feature_engineering.main),
    ("5. Confidence vs accuracy analysis", confidence_analysis.main),
    ("6. Calibration analysis", calibration.main),
    ("7. High-confidence error analysis", error_analysis.main),
    ("8. High-confidence review policy", high_confidence_review.main),
    ("9. Confidence-drift detector experiments", drift_detector.main),
    ("10. Sequence-level original-error detector", sequence_error_detector.main),
    ("11. Generate figures", visualize.main),
]


def main():
    for name, fn in STEPS:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        start = time.time()
        fn()
        print(f"({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
