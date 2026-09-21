"""Small checks for the errors that previously distorted the results."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration import binary_brier_score
from confidence_analysis import trend_statistics
from drift_detector import EXPERIMENTS, add_training_only_class_accuracy
from feature_engineering import build_features


class MethodologyTests(unittest.TestCase):
    def test_brier_penalizes_confident_wrong_prediction(self):
        step0 = pd.DataFrame({"actual_class": [1, 0], "proba_class1": [0.9, 0.9]})
        self.assertAlmostEqual(binary_brier_score(step0), (0.1**2 + 0.9**2) / 2)

    def test_drift_target_means_correct_to_incorrect(self):
        rows = []
        for doc, first_correct, last_correct, start, end in (
            ("a", True, False, 0.9, 0.6),
            ("b", False, True, 0.7, 0.8),
            ("c", True, True, 0.8, 0.75),
        ):
            rows.extend([
                {"input_id": doc, "step": 0, "correct": first_correct, "confidence": start},
                {"input_id": doc, "step": 1, "correct": last_correct, "confidence": end},
            ])
        result = trend_statistics(pd.DataFrame(rows))
        self.assertEqual(result["n_became_incorrect"], 1)
        self.assertEqual(result["n_final_incorrect"], 1)
        expected = np.corrcoef([-0.3, 0.1, -0.05], [1, 0, 0])[0, 1]
        self.assertAlmostEqual(result["corr_drift_vs_became_incorrect"], expected)

    def test_class_accuracy_ignores_holdout_labels(self):
        train = pd.DataFrame({
            "step": [0, 0, 0], "predicted_class": [0, 0, 1],
            "correct": [True, False, True],
        })
        holdout = pd.DataFrame({
            "step": [0, 0], "predicted_class": [0, 1],
            "correct": [False, False],
        })
        _, first, prior = add_training_only_class_accuracy(train, holdout)
        holdout["correct"] = True
        _, second, _ = add_training_only_class_accuracy(train, holdout)
        self.assertEqual(prior, {0: 0.5, 1: 1.0})
        self.assertEqual(first["historical_accuracy_for_class"].tolist(), [0.5, 1.0])
        self.assertEqual(first["historical_accuracy_for_class"].tolist(),
                         second["historical_accuracy_for_class"].tolist())

    def test_experiments_add_one_feature_family_at_a_time(self):
        a = set(EXPERIMENTS["A_no_confidence_features"])
        b = set(EXPERIMENTS["B_single_confidence_features"])
        c = set(EXPERIMENTS["C_confidence_drift_features"])
        self.assertEqual(b - a, {"confidence", "prediction_margin"})
        self.assertTrue(c > b)
        self.assertEqual(len(c), 13)
        self.assertNotIn("historical_accuracy_for_class", c)


if __name__ == "__main__":
    unittest.main()
