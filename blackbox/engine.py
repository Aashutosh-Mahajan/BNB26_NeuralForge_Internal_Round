"""Flight recorder and dependency-aware checkpoint replay for the LangGraph agent."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import logging
import statistics
import time
import uuid

from .agent.graph import LLM_NODES, NodeContext, build_langgraph, execute, specification
from .agent.tasks import parameters, task_id, gold_answer, verify
from .agent.tools import backup_currency_rate, parse_time
from .config import settings
from .injector.catalog import FAULT_CATALOG, inject_output
from .llm.client import BaseLLM, SandboxLLM, get_llm
from .replay.statistics import wilson_interval
from .storage import Store, canonical, content_key

logger = logging.getLogger(__name__)
PATCH_FIELDS = {"output", "prompt", "model", "temperature"}
_COPY_FIELDS = ("tokens_in", "tokens_out", "cached_in", "reasoning", "cost_usd", "latency_ms", "model",
                "temperature", "seed", "prompt", "provider", "tokens_estimated", "llm_call")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _id():
    return "R-" + uuid.uuid4().hex[:16]


def resolve_llm(llm: BaseLLM | str | None = None) -> tuple[BaseLLM, str | None]:
    if isinstance(llm, BaseLLM):
        return llm, None
    try:
        return get_llm(llm), None
    except (RuntimeError, ValueError, ImportError) as exc:
        logger.warning("Falling back to the offline sandbox LLM: %s", exc)
        return SandboxLLM(0.0), str(exc)


class Engine:
    def __init__(self, db_path="data/traces.db", llm: BaseLLM | str | None = None, use_langgraph: bool = True):
        self.store = db_path if isinstance(db_path, Store) else Store(db_path)
        self.llm, self.llm_warning = resolve_llm(llm)
        self.use_langgraph = use_langgraph

    # ------------------------------------------------------------------ runs
    def run(self, prompt, task_family="finance", params=None, on_step=None, run_id=None, *,
            expose_params=True, seed=None, llm: BaseLLM | None = None, extra=None):
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
        context = {k: v for k, v in params.items() if k != "template_id"} if expose_params else {}
        task_params = parameters(prompt, task_family, params)
        identity = run_id or _id()
        try:
            existing = self.store.get_run(identity)
        except KeyError:
            existing = None
        if existing and existing.get("status") != "QUEUED":
            raise ValueError("A completed or running run ID cannot be overwritten")
        llm = llm or self.llm
        run = self._metadata(identity, prompt, task_family, task_params, frozen_at, llm=llm,
                             seed=settings().seed if seed is None else seed)
        run["context"] = context
        run.update(extra or {})
        return self._execute(run, llm=llm, on_step=on_step)

    def _metadata(self, identity, prompt, family, params, frozen_at, parent=None, llm=None, seed=7):
        try:
            expected = gold_answer(family, params, frozen_at)
            canonical(expected)
        except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid sandbox task parameters: {exc}") from exc
        llm = llm or self.llm
        return {
            "run_id": identity, "task_id": task_id(prompt, family, params),
            "task_family": family, "template_id": params["template_id"],
            "prompt": prompt, "params": deepcopy(params), "context": {}, "status": "RUNNING",
            "success": None, "created_at": _now(), "frozen_at": frozen_at,
            "final_answer": None, "gold_answer": expected,
            "steps": [], "total_tokens": 0, "billed_tokens": 0, "cost_usd": 0.0, "total_time": 0,
            "label_step": None, "fault_type": None, "label_method": None,
            "parent_run_id": parent, "llm_provider": llm.provider, "model": llm.model, "seed": seed,
            "execution_mode": "offline" if llm.provider == "sandbox" else "live-llm",
            "stochastic": llm.stochastic,
        }

    def _inputs(self, run, name, parents, nodes):
        inputs = {"dependencies": {nodes[p - 1][0]: deepcopy(run["steps"][p - 1]["output"]) for p in parents},
                  "frozen_at": run["frozen_at"], "task_family": run["task_family"]}
        if not parents:
            inputs = {"prompt": run["prompt"], "task_family": run["task_family"], "frozen_at": run["frozen_at"]}
            if run.get("context"):
                inputs["context"] = deepcopy(run["context"])
        if name in ("reasoner", "final_answer"):
            inputs["task"] = run["prompt"]
        return inputs

    def _execute(self, run, original=None, from_step=None, patch=None, on_step=None, require_failure=False,
                 llm=None, variant=0, purpose="agent"):
        llm = llm or self.llm
        start = time.perf_counter()
        nodes = specification(run["task_family"])
        saved_checkpoints = []
        state = {"value": {}}
        restore_state = None
        if original is not None and from_step > 1:
            restore_state = self.store.get_checkpoint(original["steps"][from_step - 2]["checkpoint_id"])
        if not require_failure:
            self.store.save_run(run)
        patch = patch or {}
        overrides = {k: patch[k] for k in ("prompt", "model", "temperature") if k in patch}

        def step_runner(index, name, kind, parents):
            if original is not None and index == from_step and restore_state is not None:
                state["value"] = deepcopy(restore_state)
            inputs = self._inputs(run, name, parents, nodes)
            is_llm = name in LLM_NODES
            node_overrides = overrides if (original is not None and index == from_step) else {}
            key_extra = {"llm": llm.label if is_llm else None, "overrides": node_overrides}
            if llm.stochastic:
                key_extra["seed"] = run["seed"] + variant
            cache_key = content_key(name, inputs, key_extra)
            old = original["steps"][index - 1] if original else None
            stamp = time.perf_counter()
            cache_hit, actual, meta = False, False, {"llm_call": is_llm}
            if old is not None and index < from_step:
                output, action = deepcopy(old["output"]), "checkpoint"
            elif old is not None and index == from_step and "output" in patch:
                output, action = deepcopy(patch["output"]), "patched"
            elif old is not None and index != from_step and canonical(inputs) == canonical(old["input"]):
                output, action = deepcopy(old["output"]), "reused"
            else:
                cached = self.store.get_cached(cache_key) if not node_overrides else None
                cache_hit = cached is not None
                action = "rerun" if original is not None else "executed"
                if cache_hit:
                    output, meta = cached.get("output"), {**cached.get("meta", {}), "cache_hit": True}
                else:
                    actual = True
                    ctx = NodeContext(llm=llm, seed=run["seed"], variant=variant, run_id=run["run_id"],
                                      purpose=purpose, overrides=node_overrides)
                    try:
                        output, meta = execute(index, run["task_family"], inputs, ctx)
                        canonical(output)
                    except (ArithmeticError, AttributeError, KeyError, IndexError, TypeError, ValueError,
                            RuntimeError, TimeoutError) as exc:
                        output = {"error": str(exc), "error_type": type(exc).__name__}
                        meta = {"llm_call": is_llm, "sim_noise": isinstance(exc, TimeoutError)}
                    if not output.get("error") if isinstance(output, dict) else True:
                        # Only clean, unpatched responses enter the content-addressed cache.
                        if not node_overrides:
                            self.store.cache_output(cache_key, {"output": output, "meta": meta})
            latency = (time.perf_counter() - stamp) * 1000 if actual else 0.0
            before = deepcopy(state["value"])
            state["value"][name] = deepcopy(output)
            checkpoint = f"{run['run_id']}:{index}"
            output_dict = output if isinstance(output, dict) else {}
            documents = output_dict.get("documents", [])
            documents = documents if isinstance(documents, list) else []
            scores = output_dict.get("scores", [])
            scores = scores if isinstance(scores, list) else []
            step = {
                "run_id": run["run_id"], "step_id": index, "node_name": name, "node_type": kind,
                "parent_step_ids": parents, "input": inputs, "output": output,
                "prompt": meta.get("prompt") if is_llm else None, "llm_call": is_llm,
                "provider": meta.get("provider") if is_llm else None,
                "model": meta.get("model") if is_llm else None, "temperature": meta.get("temperature"),
                "seed": meta.get("seed"),
                "state_before": before, "state_after": deepcopy(state["value"]),
                "state_diff": {name: {"before": before.get(name), "after": deepcopy(output)}},
                "checkpoint_id": checkpoint, "cache_key": cache_key,
                "tokens_in": int(meta.get("tokens_in", 0)), "tokens_out": int(meta.get("tokens_out", 0)),
                "cached_in": int(meta.get("cached_in", 0)), "reasoning": int(meta.get("reasoning", 0)),
                "cost_usd": float(meta.get("cost_usd", 0.0)), "tokens_estimated": bool(meta.get("estimated", False)),
                "latency_ms": round(latency, 4), "retries": 0,
                "tool_error": bool(output_dict.get("error")),
                "as_of": output_dict.get("as_of") if isinstance(output_dict.get("as_of"), str) else None,
                "action": action, "cache_hit": cache_hit, "actual_execution": actual,
                "sim_noise": bool(meta.get("sim_noise")),
                "retrieved_doc_ids": [d.get("id") for d in documents if isinstance(d, dict)],
                "retrieval_scores": scores,
            }
            if old is not None and action in ("checkpoint", "reused", "patched"):
                # The recorded response keeps its original measurements; nothing is billed now.
                for key in _COPY_FIELDS:
                    if key in old:
                        step[key] = deepcopy(old[key])
            billed = actual and not cache_hit
            step["billed_tokens"] = (step["tokens_in"] + step["tokens_out"]) if billed and is_llm else 0
            step["billed_cost_usd"] = step["cost_usd"] if billed else 0.0
            run["steps"].append(step)
            saved_checkpoints.append((checkpoint, index, deepcopy(state["value"])))
            if not require_failure:
                self.store.save_checkpoint(checkpoint, run["run_id"], index, state["value"])
                self.store.save_run(run)
                if on_step:
                    on_step(deepcopy(step))
            return output

        if self.use_langgraph:
            try:
                graph = build_langgraph(run["task_family"], step_runner)
            except ImportError:
                graph = None
        else:
            graph = None
        if graph is not None:
            graph.invoke({"outputs": {}})
            run["orchestrator"] = "langgraph"
        else:
            for index, (name, kind, parents) in enumerate(nodes, start=1):
                step_runner(index, name, kind, parents)
            run["orchestrator"] = "python-dag"
        final = run["steps"][-1]["output"]
        run["final_answer"] = final.get("answer") if isinstance(final, dict) else final
        run["success"] = verify(run["final_answer"], run["gold_answer"])
        run["status"] = "PASSED" if run["success"] else "FAILED"
        run["total_tokens"] = sum(s["tokens_in"] + s["tokens_out"] for s in run["steps"])
        run["billed_tokens"] = sum(s["billed_tokens"] for s in run["steps"])
        run["cost_usd"] = round(sum(s["billed_cost_usd"] for s in run["steps"]), 8)
        run["tokens_estimated"] = any(s.get("tokens_estimated") for s in run["steps"])
        if require_failure:
            if run["success"]:
                raise ValueError("This intervention does not change the outcome; no failed run was retained")
            for checkpoint, index, checkpoint_state in saved_checkpoints:
                self.store.save_checkpoint(checkpoint, run["run_id"], index, checkpoint_state)
        run["total_time"] = round((time.perf_counter() - start) * 1000, 3)
        run["total_time_unit"] = "ms"
        self.store.save_run(run)
        try:
            from .recorder.otel import export_run
            export_run(run)
        except Exception:  # Telemetry export must never break recording.
            logger.debug("OpenTelemetry export failed", exc_info=True)
        return run

    # ----------------------------------------------------------- interventions
    def _step(self, run, step_id):
        if isinstance(step_id, bool) or not isinstance(step_id, int) or not 1 <= step_id <= len(run["steps"]):
            raise ValueError(f"step_id must be between 1 and {len(run['steps'])}")
        if run["status"] not in ("PASSED", "FAILED"):
            raise ValueError("Only a completed run can be forked")
        return run["steps"][step_id - 1]

    def _fork(self, original, llm=None):
        run = self._metadata(_id(), original["prompt"], original["task_family"], original["params"],
                             original["frozen_at"], original["run_id"], llm=llm or self._llm_for(original),
                             seed=original.get("seed", settings().seed))
        run["gold_answer"] = deepcopy(original["gold_answer"])
        run["context"] = deepcopy(original.get("context", {}))
        for key in ("split", "dataset", "template_id"):
            if key in original:
                run[key] = original[key]
        return run

    def _llm_for(self, run):
        """Replays use the provider that produced the run when it is available."""
        provider = run.get("llm_provider", "sandbox")
        if provider == "deterministic-sandbox":
            provider = "sandbox"
        if provider == self.llm.provider and (provider != "sandbox" or run.get("model") == self.llm.model):
            return self.llm
        if provider == "sandbox":
            model = str(run.get("model") or "")
            noise = float(model.split("noise")[1]) if "noise" in model else 0.0
            return SandboxLLM(noise)
        llm, _ = resolve_llm(provider)
        return llm

    def inject(self, run_id, step_id, fault_type):
        original = self.store.get_run(run_id)
        step = self._step(original, step_id)
        if not original["success"]:
            raise ValueError("Inject into a successful run so the intervention has a known outcome")
        output = inject_output(original, step, fault_type)
        if canonical(output) == canonical(step["output"]):
            raise ValueError("The intervention made no change")
        llm = self._llm_for(original)
        run = self._fork(original, llm)
        run.update({"label_step": step_id, "fault_type": fault_type, "label_method": "injected",
                    "injection": {"step_id": step_id, "fault_type": fault_type}})
        run = self._execute(run, original, step_id, {"output": output}, require_failure=True, llm=llm, purpose="inject")
        run["new_run_id"] = run["run_id"]
        self.store.save_run(run)
        return run

    def _validate_patch(self, step, patch):
        if not isinstance(patch, dict) or not patch or not set(patch) <= PATCH_FIELDS:
            raise ValueError(f"Patch must contain 'output' or any of prompt/model/temperature")
        if "output" in patch and len(patch) > 1:
            raise ValueError("An output patch replaces the response; it cannot be combined with prompt/model/temperature")
        if "output" not in patch and not step.get("llm_call") and step["node_name"] not in LLM_NODES:
            raise ValueError("prompt/model/temperature patches apply to LLM nodes only; patch a tool's output instead")
        if "temperature" in patch and (isinstance(patch["temperature"], bool) or
                                       not isinstance(patch["temperature"], (int, float)) or not 0 <= patch["temperature"] <= 2):
            raise ValueError("temperature must be a number between 0 and 2")
        try:
            canonical(patch)
        except (TypeError, ValueError) as exc:
            raise ValueError("Patch output must contain finite JSON values") from exc
        if "output" in patch and canonical(patch["output"]) == canonical(step["output"]):
            raise ValueError("Patch output is identical to the selected step; no intervention was made")

    def replay(self, run_id, from_step, patch, k=5, purpose="replay", persist=True):
        original = self.store.get_run(run_id)
        step = self._step(original, from_step)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
            raise ValueError("k must be an integer between 1 and 20")
        self._validate_patch(step, patch)
        llm = self._llm_for(original)
        variants, replay_runs = [], []
        for number in range(k):
            run = self._fork(original, llm)
            run["replay"] = {"from_step": from_step, "patch": deepcopy(patch), "variant": number + 1,
                             "deterministic": not llm.stochastic}
            result = self._execute(run, original, from_step, patch, llm=llm, variant=number, purpose=purpose)
            replay_runs.append(result)
            variants.append({"run_id": result["run_id"], "success": result["success"], "status": result["status"],
                             "final_answer": result["final_answer"], "tokens": result["billed_tokens"],
                             "cost_usd": result["cost_usd"], "deterministic": not llm.stochastic})
        first = replay_runs[0]
        reused = [s["step_id"] for s in first["steps"] if s["action"] == "reused"]
        checkpoint = [s["step_id"] for s in first["steps"] if s["action"] == "checkpoint"]
        rerun = [s["step_id"] for s in first["steps"] if s["action"] in ("rerun", "patched")]
        actual = [s["step_id"] for s in first["steps"] if s["actual_execution"]]
        passed = sum(v["success"] for v in variants)
        count = len(first["steps"])
        full_tokens = sum(s.get("tokens_in", 0) + s.get("tokens_out", 0) for s in original["steps"])
        billed = first["billed_tokens"]
        full_latency = sum(s.get("latency_ms", 0) for s in original["steps"])
        spent_latency = sum(s["latency_ms"] for s in first["steps"] if s["actual_execution"])
        # Without the response cache, every re-run LLM step would be billed again.
        rerun_tokens = sum(s.get("tokens_in", 0) + s.get("tokens_out", 0) for s in first["steps"]
                           if s["action"] == "rerun")
        stochastic = llm.stochastic
        confirmed = passed / k >= 0.5 and not original["success"]
        return {
            "run_id": first["run_id"], "new_run_id": first["run_id"],
            "original_run_id": original["run_id"], "replay_run_ids": [r["run_id"] for r in replay_runs],
            "from_step": from_step, "patch_type": "output" if "output" in patch else "+".join(sorted(patch)),
            "reused_steps": reused, "checkpoint_steps": checkpoint,
            "rerun_steps": rerun, "actual_executed_steps": actual, "total_steps": count,
            "reused_count": len(reused) + len(checkpoint), "rerun_count": len(rerun),
            "avoided_steps_pct": round(100 * (len(reused) + len(checkpoint)) / count, 2),
            "full_run_tokens": full_tokens, "total_tokens": billed, "tokens_saved": max(0, full_tokens - billed),
            "tokens_saved_pct": round(100 * (1 - billed / full_tokens), 2) if full_tokens else None,
            "rerun_tokens_without_cache": rerun_tokens,
            "tokens_saved_pct_without_cache": round(100 * (1 - rerun_tokens / full_tokens), 2) if full_tokens else None,
            "tokens_estimated": bool(original.get("tokens_estimated") or first.get("tokens_estimated")),
            "token_savings_note": ("Token counts are estimated from prompt length (offline sandbox)."
                                   if first.get("tokens_estimated") else "Token counts reported by the provider."),
            "cost_usd": round(sum(r["cost_usd"] for r in replay_runs), 8),
            "time_saved_pct": round(100 * (1 - spent_latency / full_latency), 2) if full_latency > 0 else None,
            "passed": passed, "k": k, "success_rate": passed / k, "ci95": wilson_interval(passed, k),
            "deterministic": not stochastic, "independent_trials": stochastic,
            "effective_sample_size": k if stochastic else 1,
            "ci95_note": ("Wilson 95% interval over K independently sampled re-executions." if stochastic else
                          "Descriptive Wilson interval only. Variants are identical deterministic repeats."),
            "verdict": "confirmed" if confirmed else "rejected" if passed == 0 else "inconclusive",
            "verdict_text": ("The patch restored the outcome; the diagnosis is supported." if confirmed else
                             "The patch did not restore the outcome." if passed == 0 else
                             "The original run already passed." if original["success"] else
                             "Fewer than half of the variants passed."),
            "variants": variants, "frozen_at": original["frozen_at"], "provider": llm.provider, "model": llm.model,
        }

    # ------------------------------------------------------------ fix suggester
    def _clean_output(self, run, step_id):
        step = run["steps"][step_id - 1]
        llm = self._llm_for(run)
        ctx = NodeContext(llm=llm, seed=run.get("seed", 7), variant=101, run_id=run["run_id"], purpose="fix")
        output, _ = execute(step_id, run["task_family"], step["input"], ctx)
        return output

    def suggest_fix(self, run_id, step_id, successful_runs=None):
        run = self.store.get_run(run_id)
        step = self._step(run, step_id)
        options = []

        def add(option):
            if canonical(option["patch"]["output"]) != canonical(step["output"]) and not any(
                    canonical(o["patch"]["output"]) == canonical(option["patch"]["output"]) for o in options):
                options.append(option)

        try:
            add({"id": "retry", "label": "Retry the node at the original (frozen) time",
                 "source": "re-execution with a fresh sample" if step["node_name"] in LLM_NODES else "tool retry",
                 "patch": {"output": self._clean_output(run, step_id)}})
        except (ArithmeticError, AttributeError, KeyError, IndexError, TypeError, ValueError, RuntimeError, TimeoutError):
            pass
        output = step["output"] if isinstance(step["output"], dict) else {}
        if step["node_name"] == "currency_rate":
            route = step["input"].get("dependencies", {}).get("router") if isinstance(step.get("input"), dict) else None
            args = route.get("arguments") if isinstance(route, dict) else None
            args = args if isinstance(args, dict) else {}
            try:
                backup = backup_currency_rate(args.get("base_currency", "INR"), args.get("target_currency", "USD"), run["frozen_at"])
                add({"id": "backup", "label": "Use the backup FX source", "source": backup["source"], "patch": {"output": backup}})
            except (KeyError, ValueError):
                pass
        population = successful_runs if successful_runs is not None else [r for r in self.store.list_runs(limit=2000) if r.get("success")]
        same_node = [r["steps"][step_id - 1]["output"] for r in population
                     if r.get("task_family") == run["task_family"] and len(r.get("steps", [])) >= step_id
                     and isinstance(r["steps"][step_id - 1].get("output"), dict)]
        numeric_keys = [k for k, v in output.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if same_node and numeric_keys and step["node_name"] in ("currency_rate", "formula_search", "policy_reference"):
            patched = deepcopy(output)
            for key in numeric_keys:
                values = [o[key] for o in same_node if isinstance(o.get(key), (int, float)) and not isinstance(o.get(key), bool)]
                if values:
                    patched[key] = statistics.median(values)
            if "as_of" in patched:
                patched["as_of"] = run["frozen_at"]
            if "version" in patched:
                patched["version"] = "current"
            add({"id": "historical", "label": "Use the rolling median of successful runs", "source": f"{len(same_node)} successful runs",
                 "patch": {"output": patched}})
        for candidate in population:
            if candidate.get("task_id") == run["task_id"] and len(candidate.get("steps", [])) >= step_id:
                add({"id": "historical_match", "label": "Use the matching successful run's output",
                     "source_run_id": candidate["run_id"], "patch": {"output": deepcopy(candidate["steps"][step_id - 1]["output"])}})
                break
        return {"run_id": run_id, "step_id": step_id, "options": options, "frozen_at": run["frozen_at"],
                "note": "Candidate fixes are hypotheses; replay verifies the outcome. You can also edit the patch by hand."}


__all__ = ["Engine", "FAULT_CATALOG", "resolve_llm"]
