import copy
import unittest

from blackbox.diagnosis import diagnose
from blackbox.explain import compare_runs, json_changes
from blackbox.features import extract_features
from blackbox.models.linear import train_ranker


def trace(stale=False):
    return {"run_id": "bad" if stale else "good", "created_at": "2026-10-03T10:00:00+00:00",
            "task_family": "finance", "prompt": "Convert a loan",
            "steps": [
                {"step_id": 1, "node_name": "planner", "node_type": "planner",
                 "parent_step_ids": [], "input": {"constraints": {"months": 12}},
                 "output": {"constraints": {"months": 12}}},
                {"step_id": 2, "node_name": "currency_rate", "node_type": "tool",
                 "parent_step_ids": [1], "input": {"from": "INR", "to": "USD"},
                 "output": {"rate": 0.0135 if stale else 0.012, "backup_rate": 0.012,
                            "as_of": "2025-10-02" if stale else "2026-10-03"}},
                {"step_id": 3, "node_name": "calculator", "node_type": "tool",
                 "parent_step_ids": [2], "input": {"rate": 0.0135 if stale else 0.012},
                 "output": {"value": 59.03 if stale else 52.47}},
                {"step_id": 4, "node_name": "final", "node_type": "llm",
                 "parent_step_ids": [3], "input": {}, "output": {"answer": 59.03 if stale else 52.47}}
            ]}


class DiagnosisTests(unittest.TestCase):
    def test_localizes_stale_source_without_reference(self):
        result = diagnose(trace(True))
        self.assertEqual(result["root_cause"]["step"], 2)
        self.assertGreater(result["p_fail"], 0.8)
        self.assertIn("data_age_days", [f["feature"] for f in result["evidence"]["factors"]])
        self.assertEqual(result["evidence"]["data_flow"], [2, 3, 4])
        self.assertEqual(result["model_status"], "heuristic_not_trained")

    def test_predictions_ignore_labels_and_outcomes(self):
        original = trace(True)
        altered = copy.deepcopy(original)
        altered.update(success=True, label_step=4, fault_type="magic", gold_answer=12345)
        altered["steps"][0]["input"].update(gold_answer=77, success=False, fault_type="nonsense")
        altered["steps"][0]["output"]["constraints"]["template_id"] = "held-out-template-metadata"
        altered["steps"][0]["input"]["checkpoint_id"] = "metadata-not-evidence"
        left, right = diagnose(original), diagnose(altered)
        for key in ("p_fail", "root_cause", "top_suspects", "evidence"):
            self.assertEqual(left[key], right[key])
        self.assertEqual(extract_features(original)["matrix"], extract_features(altered)["matrix"])

    def test_fresh_moving_value_agrees_with_backup(self):
        run = trace()
        run["steps"][1]["output"].update(rate=0.021, backup_rate=0.021)
        self.assertLess(diagnose(run)["p_fail"], 0.5)

    def test_reference_rejects_future_or_unrelated_tasks(self):
        healthy = trace()
        healthy["created_at"] = "2027-10-03"
        self.assertIsNone(diagnose(trace(True), [healthy])["evidence"]["nearest_success"])
        healthy["created_at"] = "2026-10-02"
        healthy["steps"][0]["input"]["constraints"]["months"] = 24
        self.assertIsNone(diagnose(trace(True), [healthy])["evidence"]["nearest_success"])

    def test_exact_historical_reference_first_divergence(self):
        result = diagnose(trace(True), [trace()])
        self.assertEqual(result["evidence"]["nearest_success"], "good")
        self.assertEqual(result["evidence"]["diverges_at"], 2)

    def test_empty_trace(self):
        self.assertIsNone(diagnose({"steps": []})["root_cause"])

    def test_alignment_handles_insertion(self):
        before = trace()
        after = copy.deepcopy(before)
        after["steps"].insert(1, {"step_id": 99, "node_name": "extra", "node_type": "retriever",
                                   "input": {}, "output": {"document": "helper"}})
        comparison = compare_runs(before, after)
        self.assertEqual(comparison["first_divergence"], 99)
        self.assertEqual([s["status"] for s in comparison["aligned_steps"]],
                         ["unchanged", "added", "unchanged", "unchanged", "unchanged"])

    def test_recursive_diff_and_outcome(self):
        changes = json_changes({"x": [1, {"y": 2}]}, {"x": [1, {"y": 3}], "z": True})
        self.assertEqual(changes[0]["path"], "$.x[1].y")
        before, after = trace(True), trace()
        before["success"], after["success"] = False, True
        comparison = compare_runs(before, after)
        self.assertEqual(comparison["first_divergence"], 2)
        self.assertTrue(comparison["outcome"]["changed"])

    def test_trainable_baseline_has_provenance_and_no_label_features(self):
        good, bad = trace(), trace(True)
        good.update(success=True, label_step=None, template_id="train-good")
        bad.update(success=False, label_step=2, template_id="train-bad", fault_type="stale_data")
        ranker = train_ranker([good, bad], epochs=20)
        self.assertEqual(ranker.predict(bad)[0]["step"], 2)
        self.assertFalse(ranker.artifact["provenance"]["references_used"])
        mutated = copy.deepcopy(bad)
        mutated.update(label_step=1, success=True, gold_answer=100)
        self.assertEqual(ranker.predict(bad), ranker.predict(mutated))


if __name__ == "__main__":
    unittest.main()
