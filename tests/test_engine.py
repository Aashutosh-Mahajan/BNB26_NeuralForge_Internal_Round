import copy
import tempfile
import unittest
from pathlib import Path

from blackbox import Engine, SandboxAgent, Store, wrap
from blackbox.engine import FAULT_CATALOG
from blackbox.llm import SandboxLLM
from blackbox.replay import wilson_interval

TIME = "2026-10-03T12:00:00+00:00"
PROMPT = "Convert INR 50,000 to USD and compute EMI for 12 months at 9%."

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "traces.db")
        self.engine = Engine(self.store, llm=SandboxLLM(0.0))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def good(self, family="finance", **params):
        return self.engine.run(PROMPT if family == "finance" else "Example task", family, {"frozen_at": TIME, **params})

    def test_four_families_record_checkpoints_and_zero_billed_tokens(self):
        for family in ("finance", "sql", "doc_qa", "math"):
            run = self.good(family)
            self.assertTrue(run["success"], family)
            # Offline calls spend estimated tokens but cost nothing.
            self.assertEqual(run["billed_tokens"], run["total_tokens"])
            self.assertEqual(run["cost_usd"], 0)
            self.assertTrue(run["tokens_estimated"])
            self.assertGreater(run["total_tokens"], 0)
            self.assertEqual(run["orchestrator"], "langgraph")
            self.assertEqual([s["llm_call"] for s in run["steps"]],
                             [True, True, False, False, False, False, False, False, True, True])
            self.assertEqual(len(run["steps"]), 10)
            for step in run["steps"]:
                self.assertEqual(self.store.get_checkpoint(step["checkpoint_id"]), step["state_after"])
                self.assertTrue(all(p < step["step_id"] for p in step["parent_step_ids"]))
        self.assertEqual(len(self.store.list_runs()), 4)

    def test_replay_changes_descendants_reuses_independent_branches(self):
        baseline = self.good()
        failed = self.engine.inject(baseline["run_id"], 3, "stale_data")
        original_snapshot = copy.deepcopy(self.store.get_run(failed["run_id"]))
        patch = self.engine.suggest_fix(failed["run_id"], 3)["options"][0]["patch"]
        replay = self.engine.replay(failed["run_id"], 3, patch, k=3)
        fixed = self.store.get_run(replay["run_id"])
        self.assertEqual(replay["passed"], 3)
        self.assertEqual(replay["checkpoint_steps"], [1, 2])
        self.assertEqual(replay["reused_steps"], [4, 7])
        self.assertEqual(replay["rerun_steps"], [3, 5, 6, 8, 9, 10])
        self.assertEqual(fixed["frozen_at"], TIME)
        self.assertEqual(fixed["final_answer"], baseline["final_answer"])
        self.assertEqual(self.store.get_run(failed["run_id"]), original_snapshot)
        self.assertEqual(fixed["parent_run_id"], failed["run_id"])
        self.assertEqual(replay["avoided_steps_pct"], 40.0)
        # Reasoner and final answer re-run; the cache serves them, so nothing is billed.
        self.assertEqual(replay["tokens_saved_pct"], 100.0)
        self.assertLess(replay["tokens_saved_pct_without_cache"], 100.0)
        self.assertFalse(replay["independent_trials"])
        self.assertEqual(replay["effective_sample_size"], 1)

    def test_fault_cannot_poison_the_clean_cache(self):
        baseline = self.good()
        bad = self.engine.inject(baseline["run_id"], 3, "wrong_tool_value")
        self.assertFalse(bad["success"])
        clean_again = self.good()
        self.assertEqual(clean_again["steps"][2]["output"], baseline["steps"][2]["output"])
        self.assertTrue(clean_again["success"])
        cached = self.store.get_cached(baseline["steps"][2]["cache_key"])
        self.assertEqual(cached["output"], baseline["steps"][2]["output"])

    def test_all_twelve_faults_have_effective_compatible_interventions(self):
        baseline = self.good()
        selected = {
            "wrong_tool_value": 3, "empty_result": 3, "tool_timeout": 3, "stale_data": 3,
            "retrieval_poisoning": 4, "wrong_tool_choice": 2, "wrong_arguments": 2,
            "dropped_constraint": 1, "hallucinated_fact": 9, "premature_final": 2,
            "memory_overwrite": 6, "loop_repetition": 2,
        }
        self.assertEqual(set(selected), {f["id"] for f in FAULT_CATALOG})
        for fault, step in selected.items():
            with self.subTest(fault=fault):
                run = self.engine.inject(baseline["run_id"], step, fault)
                self.assertFalse(run["success"])
                self.assertEqual(run["label_step"], step)
                fix = self.engine.suggest_fix(run["run_id"], step)["options"][0]["patch"]
                self.assertEqual(self.engine.replay(run["run_id"], step, fix, k=1)["passed"], 1)

    def test_reject_incompatible_fault_noop_patch_and_unimplemented_options(self):
        baseline = self.good()
        before = len(self.store.list_runs())
        with self.assertRaises(ValueError):
            self.engine.inject(baseline["run_id"], 4, "stale_data")
        with self.assertRaises(ValueError):
            self.engine.replay(baseline["run_id"], 3, {"output": baseline["steps"][2]["output"]})
        with self.assertRaises(ValueError):
            self.engine.replay(baseline["run_id"], 3, {"model": "pretend"})
        self.assertEqual(len(self.store.list_runs()), before)

    def test_ineffective_injection_not_retained(self):
        baseline = self.good("sql")
        # A date-util term has no influence on this SQL answer.
        before = len(self.store.list_runs())
        with self.assertRaisesRegex(ValueError, "does not change the outcome"):
            self.engine.inject(baseline["run_id"], 7, "wrong_tool_value")
        self.assertEqual(len(self.store.list_runs()), before)

    def test_wilson_interval_bounds(self):
        low, high = wilson_interval(5, 5)
        self.assertAlmostEqual(low, 0.5655175, places=6)
        self.assertAlmostEqual(high, 1.0, places=10)
        low, high = wilson_interval(0, 5)
        self.assertAlmostEqual(low, 0.0, places=10)
        self.assertAlmostEqual(high, 0.4344825, places=6)
        with self.assertRaises(ValueError):
            wilson_interval(1, 0)

    def test_live_callback_sees_persisted_checkpoint(self):
        seen = []
        def callback(step):
            saved = self.store.get_run(step["run_id"])
            self.assertEqual(saved["steps"][-1]["step_id"], step["step_id"])
            self.assertEqual(self.store.get_checkpoint(step["checkpoint_id"]), step["state_after"])
            seen.append(step["step_id"])
        run = self.engine.run(PROMPT, params={"frozen_at": TIME}, on_step=callback)
        self.assertEqual(seen, list(range(1, 11)))
        self.assertTrue(run["success"])

    def test_frozen_time_and_parameter_changes_have_different_cache_keys(self):
        one = self.good(amount=50000)
        two = self.good(amount=90000)
        self.assertNotEqual(one["steps"][2]["cache_key"], two["steps"][2]["cache_key"])
        old = self.engine.run(PROMPT, params={"frozen_at": "2025-10-03T12:00:00+00:00"})
        self.assertNotEqual(old["gold_answer"], one["gold_answer"])
        self.assertNotEqual(old["steps"][2]["cache_key"], one["steps"][2]["cache_key"])

    def test_sdk_supported_scope_is_explicit(self):
        recorded = wrap(SandboxAgent("math"), self.store).invoke("Buy 8 items at $12 with a 10% discount.")
        self.assertTrue(recorded["success"])
        self.assertEqual(recorded["final_answer"], 86.4)
        with self.assertRaises(TypeError):
            wrap(object(), self.store)


class InputBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine(":memory:", llm=SandboxLLM(0.0))

    def tearDown(self):
        self.engine.store.close()

    def test_arbitrary_json_router_patch_is_a_recorded_failure(self):
        baseline = self.engine.run(PROMPT, params={"frozen_at": TIME})
        for output in (None, 42, [], "currency_rate"):
            with self.subTest(output=output):
                replay = self.engine.replay(baseline["run_id"], 2, {"output": output}, k=1)
                recorded = self.engine.store.get_run(replay["run_id"])
                self.assertEqual(recorded["status"], "FAILED")
                self.assertEqual(len(recorded["steps"]), 10)
                self.assertTrue(recorded["steps"][2]["tool_error"])
                # Suggested fixes must tolerate a malformed upstream dependency.
                self.engine.suggest_fix(recorded["run_id"], 3)

    def test_malformed_retrieval_metadata_does_not_crash_recorder(self):
        baseline = self.engine.run("How many days do I have to return a purchase?", "doc_qa")
        for documents in (42, [7, None, "document"]):
            replay = self.engine.replay(baseline["run_id"], 3, {"output": {"documents": documents, "scores": 10, "as_of": {"date": TIME}}}, k=1)
            recorded = self.engine.store.get_run(replay["run_id"])
            self.assertEqual(recorded["status"], "FAILED")
            self.assertEqual(recorded["steps"][2]["retrieved_doc_ids"], [])
            self.assertEqual(recorded["steps"][2]["retrieval_scores"], [])
            self.assertIsNone(recorded["steps"][2]["as_of"])

    def test_huge_final_number_is_rejected_by_verifier_without_overflow(self):
        baseline = self.engine.run(PROMPT)
        replay = self.engine.replay(baseline["run_id"], 10, {"output": {"answer": 10 ** 400}}, k=1)
        self.assertEqual(replay["passed"], 0)
        self.assertEqual(self.engine.store.get_run(replay["run_id"])["status"], "FAILED")

    def test_non_json_and_non_finite_patches_fail_before_fork(self):
        baseline = self.engine.run(PROMPT)
        count = len(self.engine.store.list_runs())
        for value in ({"rate": float("nan")}, {"rate": float("inf")}, object()):
            with self.assertRaisesRegex(ValueError, "finite JSON"):
                self.engine.replay(baseline["run_id"], 3, {"output": value}, k=1)
        self.assertEqual(len(self.engine.store.list_runs()), count)

    def test_invalid_parameters_have_clear_errors_without_partial_runs(self):
        cases = [
            ("sql", {"region": 1}),
            ("finance", {"annual_rate": 10 ** 400}),
            ("math", {"template_id": []}),
            ("sql", {"quarter": "Q9"}),
            ("doc_qa", {"category": "spaceships"}),
            ("finance", ["not a parameter object"]),
        ]
        for family, params in cases:
            with self.subTest(family=family, params=str(params)[:60]):
                with self.assertRaises(ValueError):
                    self.engine.run("Example task", family, params)
        with self.assertRaises(ValueError):
            self.engine.run("Convert INR 50,000 for " + "9" * 400 + " months")
        self.assertEqual(self.engine.store.list_runs(), [])




class LLMProviderTests(unittest.TestCase):
    """The agent runs against any provider object; here a scripted fake stands in for GPT-6 Luna."""

    def test_chat_provider_records_tokens_cost_and_prompt_patches(self):
        import json
        from blackbox.llm.client import BaseLLM, LLMResult

        class ScriptedLLM(BaseLLM):
            provider, model, stochastic = "openai", "gpt-6-luna", True

            def __init__(self):
                self.calls = []

            def complete_json(self, system, user, **kwargs):
                self.calls.append((system, user, kwargs))
                if "planning node" in system:
                    data = {"constraints": {"amount": 50000, "months": 12, "annual_rate": 9,
                                            "base_currency": "INR", "target_currency": "USD"}}
                elif "router node" in system:
                    plan = json.loads(user.split("Plan: ", 1)[1])
                    data = {"tool": "currency_rate", "arguments": plan["constraints"]}
                elif "reasoning node" in system:
                    calc = json.loads(user.split("Calculator output: ", 1)[1].split(chr(10) + "Additional")[0])
                    data = {"answer": calc["value"], "rationale": "copied"}
                else:
                    data = {"answer": json.loads(user.split("Reasoner output: ", 1)[1])["answer"]}
                return LLMResult(data=data, text=json.dumps(data), provider="openai", model=kwargs.get("model") or self.model,
                                 tokens_in=200, tokens_out=50, cost_usd=0.000045, temperature=kwargs.get("temperature"),
                                 seed=kwargs.get("seed"))

        llm = ScriptedLLM()
        engine = Engine(":memory:", llm=llm)
        run = engine.run(PROMPT, params={"frozen_at": TIME}, expose_params=False)
        self.assertTrue(run["success"])
        self.assertEqual(run["billed_tokens"], 4 * 250)
        self.assertAlmostEqual(run["cost_usd"], 4 * 0.000045)
        self.assertEqual(run["llm_provider"], "openai")
        self.assertIn("planning node", run["steps"][0]["prompt"])
        replay = engine.replay(run["run_id"], 9, {"prompt": "Be precise.", "temperature": 0.5}, k=2)
        self.assertTrue(replay["independent_trials"])
        self.assertEqual(replay["passed"], 2)
        self.assertIn("Additional instruction: Be precise.", llm.calls[-1][1])
        with self.assertRaises(ValueError):
            engine.replay(run["run_id"], 3, {"prompt": "tools take no prompt"}, k=1)

    def test_budget_guard_and_pricing(self):
        import tempfile
        from blackbox.config import settings
        from blackbox.llm.usage import BudgetExceeded, UsageLedger
        cfg = settings()
        self.assertAlmostEqual(cfg.price_usd(1_000_000, 0), 0.10)
        self.assertAlmostEqual(cfg.price_usd(1_000_000, 1_000_000, cached_in=500_000), 0.05 + 0.005 + 0.50)
        with tempfile.TemporaryDirectory() as folder:
            ledger = UsageLedger(folder + "/usage.db")
            ledger.record(provider="openai", model="gpt-6-luna", purpose="agent", run_id="x", node="planner",
                          tokens_in=1, tokens_out=1, cached_in=0, reasoning=0, cost_usd=cfg.budget_usd + 1)
            with self.assertRaises(BudgetExceeded):
                ledger.check("openai")
            ledger.check("sandbox")


class DatasetAndLabelerTests(unittest.TestCase):
    def test_templates_parse_and_counterfactual_labels_match_simulation(self):
        import tempfile
        from blackbox.datagen import generate
        with tempfile.TemporaryDirectory() as folder:
            manifest = generate(folder + "/d.db", "sandbox", noise=0.15, tasks_per_family=12, seed=3)
            self.assertEqual(manifest["errors"], 0)
            self.assertGreater(manifest["natural_failures"], 0)
            self.assertGreater(manifest["injected"], manifest["clean"])
            self.assertEqual(manifest["labeler_agreement_with_simulation"], 1.0)
            self.assertIn("unseen_fault/injected", manifest["counts"])


class TelemetryTests(unittest.TestCase):
    def test_openinference_spans(self):
        from blackbox.recorder.otel import spans_for_run
        engine = Engine(":memory:", llm=SandboxLLM(0.0))
        spans = spans_for_run(engine.run(PROMPT, params={"frozen_at": TIME}))
        self.assertEqual(spans[0]["attributes"]["openinference.span.kind"], "AGENT")
        kinds = [s["attributes"]["openinference.span.kind"] for s in spans[1:]]
        self.assertEqual(kinds[:4], ["LLM", "LLM", "TOOL", "RETRIEVER"])
        self.assertIn("llm.token_count.prompt", spans[1]["attributes"])

    def test_wrap_records_an_external_langgraph(self):
        from typing import TypedDict
        from langgraph.graph import END, START, StateGraph

        class State(TypedDict, total=False):
            question: str
            plan: str
            answer: int

        graph = StateGraph(State)
        graph.add_node("planner", lambda s: {"plan": "add"})
        graph.add_node("calculator_tool", lambda s: {"answer": 4})
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "calculator_tool")
        graph.add_edge("calculator_tool", END)
        store = Store(":memory:")
        run = wrap(graph.compile(), store).invoke({"question": "2+2"}, gold_answer=4)
        self.assertEqual(run["status"], "PASSED")
        self.assertEqual([s["node_name"] for s in run["steps"]], ["planner", "calculator_tool"])
        self.assertEqual(store.get_checkpoint(run["steps"][1]["checkpoint_id"])["answer"], 4)


if __name__ == "__main__":
    unittest.main()
