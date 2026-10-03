"""Flight recorder and dependency-aware replay for deterministic sandbox DAGs."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import time
import uuid

from .agent.graph import execute, specification
from .agent.tasks import parameters, task_id, gold_answer, verify
from .agent.tools import parse_time
from .injector.catalog import FAULT_CATALOG, inject_output
from .replay.statistics import wilson_interval
from .storage import Store, canonical, content_key

def _now():
    return datetime.now(timezone.utc).isoformat()

def _id():
    return "R-" + uuid.uuid4().hex[:16]

class Engine:
    """A sandbox executor, not a LangGraph or live-LLM adapter."""
    def __init__(self, db_path="data/traces.db"):
        self.store = db_path if isinstance(db_path, Store) else Store(db_path)

    def run(self, prompt, task_family="finance", params=None, on_step=None, run_id=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A non-empty task prompt is required")
        if params is not None and not isinstance(params, dict):
            raise ValueError("params must be a JSON object")
        params = deepcopy(params or {})
        try:
            canonical(params)
        except (TypeError, ValueError) as exc:
            raise ValueError("Task parameters must contain finite JSON values") from exc
        frozen_at = parse_time(params.pop("frozen_at", _now())).isoformat()
        task_params = parameters(prompt, task_family, params)
        identity = run_id or _id()
        try:
            existing = self.store.get_run(identity)
        except KeyError:
            existing = None
        if existing and existing.get("status") != "QUEUED":
            raise ValueError("A completed or running run ID cannot be overwritten")
        run = self._metadata(identity, prompt, task_family, task_params, frozen_at)
        return self._execute(run, on_step=on_step)

    def _metadata(self, identity, prompt, family, params, frozen_at, parent=None):
        try:
            expected = gold_answer(family, params, frozen_at)
            canonical(expected)
        except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid sandbox task parameters: {exc}") from exc
        return {
            "run_id": identity, "task_id": task_id(prompt, family, params),
            "task_family": family, "template_id": params["template_id"],
            "prompt": prompt, "params": deepcopy(params), "status": "RUNNING",
            "success": None, "created_at": _now(), "frozen_at": frozen_at,
            "final_answer": None, "gold_answer": expected,
            "steps": [], "total_tokens": 0, "total_time": 0,
            "label_step": None, "fault_type": None, "label_method": None,
            "parent_run_id": parent, "llm_provider": "deterministic-sandbox",
            "execution_mode": "offline", "model": None,
        }

    def _execute(self, run, original=None, from_step=None, patch=None, on_step=None, require_failure=False):
        start = time.perf_counter()
        state = {}
        saved_checkpoints = []
        nodes = specification(run["task_family"])
        # Verify and actually load the original checkpoint before the intervention.
        restore_state = None
        if original is not None and from_step > 1:
            restore_state = self.store.get_checkpoint(original["steps"][from_step - 2]["checkpoint_id"])
        if not require_failure:
            self.store.save_run(run)
        for index, (name, kind, parents) in enumerate(nodes, start=1):
            if original is not None and index == from_step and restore_state is not None:
                state = deepcopy(restore_state)
            inputs = {"dependencies": {nodes[p - 1][0]: deepcopy(run["steps"][p - 1]["output"]) for p in parents}, "frozen_at": run["frozen_at"], "task_family": run["task_family"]}
            if not parents:
                inputs.update({"prompt": run["prompt"], "params": deepcopy(run["params"])})
            cache_key = content_key(name, inputs)
            old = original["steps"][index - 1] if original else None
            stamp = time.perf_counter()
            cache_hit = False
            actual_execution = False
            if old is not None and index < from_step:
                output, action = deepcopy(old["output"]), "checkpoint"
            elif old is not None and index == from_step:
                output, action = deepcopy(patch["output"]), "patched"
            elif old is not None and canonical(inputs) == canonical(old["input"]):
                output, action = deepcopy(old["output"]), "reused"
            else:
                output = self.store.get_cached(cache_key)
                cache_hit = output is not None
                action = "rerun" if original is not None else "executed"
                if not cache_hit:
                    actual_execution = True
                    try:
                        output = execute(index, run["task_family"], inputs)
                        canonical(output)
                    except (ArithmeticError, AttributeError, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
                        output = {"error": str(exc), "error_type": type(exc).__name__}
                    # Only unpatched actual responses enter the clean cache.
                    self.store.cache_output(cache_key, output)
            latency = (time.perf_counter() - stamp) * 1000 if actual_execution else 0.0
            before = deepcopy(state)
            state[name] = deepcopy(output)
            checkpoint = f"{run['run_id']}:{index}"
            output_dict = output if isinstance(output, dict) else {}
            documents = output_dict.get("documents", [])
            documents = documents if isinstance(documents, list) else []
            scores = output_dict.get("scores", [])
            scores = scores if isinstance(scores, list) else []
            step = {
                "run_id": run["run_id"], "step_id": index, "node_name": name, "node_type": kind,
                "parent_step_ids": parents, "input": inputs, "output": output,
                "prompt": run["prompt"] if kind == "planner" else None,
                "model": None, "temperature": None, "seed": None,
                "state_before": before, "state_after": deepcopy(state),
                "state_diff": {name: {"before": before.get(name), "after": deepcopy(output)}},
                "checkpoint_id": checkpoint, "cache_key": cache_key,
                "tokens_in": 0, "tokens_out": 0, "latency_ms": round(latency, 4),
                "retries": 0, "tool_error": bool(output_dict.get("error")),
                "as_of": output_dict.get("as_of") if isinstance(output_dict.get("as_of"), str) else None, "action": action, "cache_hit": cache_hit,
                "actual_execution": actual_execution,
                "retrieved_doc_ids": [d.get("id") for d in documents if isinstance(d, dict)],
                "retrieval_scores": scores,
            }
            run["steps"].append(step)
            saved_checkpoints.append((checkpoint, index, deepcopy(state)))
            if not require_failure:
                self.store.save_checkpoint(checkpoint, run["run_id"], index, state)
                self.store.save_run(run)
                if on_step:
                    on_step(deepcopy(step))
        final = run["steps"][-1]["output"]
        run["final_answer"] = final.get("answer") if isinstance(final, dict) else final
        run["success"] = verify(run["final_answer"], run["gold_answer"])
        run["status"] = "PASSED" if run["success"] else "FAILED"
        if require_failure:
            if run["success"]:
                raise ValueError("This intervention does not change the outcome; no failed run was retained")
            for checkpoint, index, checkpoint_state in saved_checkpoints:
                self.store.save_checkpoint(checkpoint, run["run_id"], index, checkpoint_state)
        # Wall time includes node work, cache access, checkpoints, and synchronous
        # recording callbacks. The final outcome-save below is not in this sample.
        run["total_time"] = round((time.perf_counter() - start) * 1000, 3)
        run["total_time_unit"] = "ms"
        self.store.save_run(run)
        return run

    def _step(self, run, step_id):
        if isinstance(step_id, bool) or not isinstance(step_id, int) or not 1 <= step_id <= len(run["steps"]):
            raise ValueError(f"step_id must be between 1 and {len(run['steps'])}")
        if run["status"] not in ("PASSED", "FAILED"):
            raise ValueError("Only a completed run can be forked")
        return run["steps"][step_id - 1]

    def _fork(self, original):
        run = self._metadata(_id(), original["prompt"], original["task_family"], original["params"], original["frozen_at"], original["run_id"])
        run["gold_answer"] = deepcopy(original["gold_answer"])
        return run

    def inject(self, run_id, step_id, fault_type):
        original = self.store.get_run(run_id)
        step = self._step(original, step_id)
        if not original["success"]:
            raise ValueError("Inject into a successful run so the intervention has a known outcome")
        output = inject_output(original, step, fault_type)
        if canonical(output) == canonical(step["output"]):
            raise ValueError("The intervention made no change")
        run = self._fork(original)
        run.update({"label_step": step_id, "fault_type": fault_type, "label_method": "injected", "injection": {"step_id": step_id, "fault_type": fault_type}})
        run = self._execute(run, original, step_id, {"output": output}, require_failure=True)
        run["new_run_id"] = run["run_id"]
        self.store.save_run(run)
        return run

    def replay(self, run_id, from_step, patch, k=5):
        original = self.store.get_run(run_id)
        step = self._step(original, from_step)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
            raise ValueError("k must be an integer between 1 and 20")
        if not isinstance(patch, dict) or set(patch) != {"output"}:
            raise ValueError("Offline sandbox supports a patch containing exactly 'output'; prompt/model/temperature patches require a live adapter")
        try:
            canonical(patch)
        except (TypeError, ValueError) as exc:
            raise ValueError("Patch output must contain finite JSON values") from exc
        if canonical(patch["output"]) == canonical(step["output"]):
            raise ValueError("Patch output is identical to the selected step; no intervention was made")
        variants = []
        replay_runs = []
        for number in range(k):
            run = self._fork(original)
            run["replay"] = {"from_step": from_step, "patch": deepcopy(patch), "variant": number + 1, "deterministic": True}
            result = self._execute(run, original, from_step, patch)
            replay_runs.append(result)
            variants.append({"run_id": result["run_id"], "success": result["success"], "status": result["status"], "final_answer": result["final_answer"], "tokens": 0, "deterministic": True})
        first = replay_runs[0]
        reused = [s["step_id"] for s in first["steps"] if s["action"] == "reused"]
        checkpoint = [s["step_id"] for s in first["steps"] if s["action"] == "checkpoint"]
        rerun = [s["step_id"] for s in first["steps"] if s["action"] in ("rerun", "patched")]
        actual = [s["step_id"] for s in first["steps"] if s["actual_execution"]]
        passed = sum(v["success"] for v in variants)
        count = len(first["steps"])
        return {
            "run_id": first["run_id"], "new_run_id": first["run_id"],
            "original_run_id": original["run_id"], "replay_run_ids": [r["run_id"] for r in replay_runs],
            "from_step": from_step, "reused_steps": reused, "checkpoint_steps": checkpoint,
            "rerun_steps": rerun, "actual_executed_steps": actual, "total_steps": count,
            "reused_count": len(reused) + len(checkpoint), "rerun_count": len(rerun),
            "avoided_steps_pct": round(100 * (len(reused) + len(checkpoint)) / count, 2),
            "tokens_saved_pct": None, "total_tokens": 0, "tokens_saved": 0,
            "token_savings_note": "No LLM calls were made; token savings are not measurable in this sandbox.",
            "passed": passed, "k": k, "success_rate": passed / k, "ci95": wilson_interval(passed, k),
            "deterministic": True, "independent_trials": False, "effective_sample_size": 1,
            "ci95_note": "Descriptive Wilson interval only. Variants are identical deterministic repeats, not independent stochastic evidence.",
            "verdict": "confirmed" if passed == k and not original["success"] else "rejected" if passed == 0 else "inconclusive",
            "verdict_text": "The patch restored the sandbox outcome." if passed == k and not original["success"] else "The patch did not restore the sandbox outcome." if passed == 0 else "The original run already passed.",
            "variants": variants, "frozen_at": original["frozen_at"],
        }

    def suggest_fix(self, run_id, step_id):
        run = self.store.get_run(run_id)
        step = self._step(run, step_id)
        options = []
        try:
            clean_output = execute(step_id, run["task_family"], step["input"])
            if canonical(clean_output) != canonical(step["output"]):
                options.append({"id": "retry", "label": "Retry sandbox node at the original timestamp", "source": "deterministic tool execution", "patch": {"output": clean_output}})
        except (ArithmeticError, AttributeError, KeyError, IndexError, TypeError, ValueError, RuntimeError):
            pass
        candidates = [r for r in self.store.list_runs() if r.get("task_id") == run["task_id"] and r.get("success") and len(r.get("steps", [])) >= step_id]
        for candidate in candidates:
            output = candidate["steps"][step_id - 1]["output"]
            if canonical(output) != canonical(step["output"]) and not any(canonical(o["patch"]["output"]) == canonical(output) for o in options):
                options.append({"id": "historical", "label": "Use matching successful task output", "source_run_id": candidate["run_id"], "patch": {"output": deepcopy(output)}})
                break
        return {"run_id": run_id, "step_id": step_id, "options": options, "frozen_at": run["frozen_at"], "note": "Candidate fixes are hypotheses; replay verifies the sandbox outcome."}
