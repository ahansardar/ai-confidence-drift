"""Small checks for the errors that previously distorted the results."""
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration import binary_brier_score
from confidence_analysis import trend_statistics
from drift_detector import EXPERIMENTS, add_training_only_class_accuracy
from feature_engineering import build_features
from high_confidence_review import policy_from_predictions, review_flags, evaluate_flags
from sequence_error_detector import score_documents
from config import DATA_PROCESSED, MODELS


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

    def test_high_confidence_review_uses_training_errors_and_ignores_test_labels(self):
        policy = policy_from_predictions(
            actual=[0, 1, 1, 0], predicted=[1, 1, 0, 0],
            confidence=[0.9, 0.8, 0.55, 0.88],
        )
        self.assertEqual(policy["review_predicted_classes"], [1])
        predicted = [1, 1, 0]
        confidence = [0.96, 0.77, 0.99]
        self.assertEqual(review_flags(predicted, confidence, policy).tolist(), [True, True, False])
        first = evaluate_flags([0, 1, 0], predicted, confidence, policy)
        second = evaluate_flags([1, 0, 1], predicted, confidence, policy)
        self.assertEqual(first["n_total_routed_to_review"], second["n_total_routed_to_review"])

    def test_review_policy_does_not_exempt_classes_when_training_has_no_errors(self):
        policy = policy_from_predictions(
            actual=[0, 1], predicted=[0, 1], confidence=[0.9, 0.9],
        )
        self.assertEqual(policy["review_predicted_classes"], [0, 1])

    def test_sequence_detector_scores_unlabeled_documents(self):
        documents = pd.read_csv(DATA_PROCESSED / "test.csv").head(3)
        base = joblib.load(MODELS / "baseline_model.joblib")
        detector = joblib.load(MODELS / "drift_detector_best.joblib")
        self.assertEqual(detector["target"], "original step-0 prediction incorrect after six probes")
        scored = score_documents(base, detector, documents[["input_id", "text"]])
        self.assertEqual(len(scored), 3)
        self.assertEqual(scored["input_id"].tolist(), documents["input_id"].tolist())
        self.assertTrue(scored["original_error_risk"].between(0, 1).all())
        self.assertEqual(scored["flagged"].dtype, bool)


if __name__ == "__main__":
    unittest.main()
