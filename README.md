# Black Box — an evidence-driven debugger for AI agents

Black Box records every step of an agent run, ranks the step most likely to have started a failure, shows the recorded evidence, and tests alternative fixes from saved checkpoints while keeping unaffected work. Repairs are judged by independent acceptance checks, never by the explanation.

> Record → detect → rank suspects → explain → propose alternatives → test them from a checkpoint → verify independently → compare.

It does **not** read a model's private reasoning, it does not repair every agent automatically, and a passing repair *supports* a suspect rather than proving a unique cause.

## Which models are where

| What | Model |
|---|---|
| Benchmarks and training data (final, in `data/metrics.json`) | **GPT-6 Luna** agent runs ($0.65 total OpenAI spend, including the Who&When judge) |
| Live demo agent (optional) | **gpt-5.4-nano** via `LLM_PROVIDER=openai` — default is the free offline sandbox |
| Natural failures, hybrid judge | **qwen2.5:3b** on local Ollama ($0) |
| Diagnosis | Trained Transformer + LightGBM + autoencoder ensemble on the local GPU (no API) |

OpenAI models that reject `temperature` (GPT-6 Luna does) run at their default; reproducibility comes from the response cache and K-variant replays with a Wilson interval, not from temperature 0.

## What is built

| Area | Implementation | Code |
|---|---|---|
| Agent | LangGraph agent, 10 nodes, LLM planner/router/reasoner/final, 4 task families, 48 templates | `blackbox/agent/` |
| Recorder | Inputs, outputs, state, dependencies, checkpoint per step, content cache with expiry for changing data, bounded retries, side-effect type per tool, secret redaction, recording overhead measured separately, schema v2 | `engine.py`, `storage.py` |
| External agents | `blackbox.wrap(graph)` for any compiled LangGraph: native checkpoints, reducer-correct state, superstep dependencies (parallel branches), crashes as ERROR incidents with the last good checkpoint, **fork** (patch a node, LangGraph re-runs what follows, independent branches kept) and **resume** | `recorder/sdk.py`, `examples/expense_agent.py` |
| Claude Code | Hook events (http hooks) recorded as runs; record + diagnose only | `integrations/` |
| MCP | `list_failed_runs`, `diagnose_run`, `explain_run`, `test_fix`, `failure_stats` for Claude/Cursor/Codex | `mcp_server.py` |
| Diagnosis | 412-dim step features (MiniLM, NLI, 20 numeric), ensemble 0.6/0.3/0.1, sliding windows for runs over 24 steps, "not sure" below a validated threshold | `diagnosis.py`, `models/` |
| Acceptance checks | Independent pass/fail/unknown: answer, errors, quote freshness, source agreement, current policy, arguments vs plan | `verifier.py` |
| Alternatives | Strategy catalogue with kind (tool re-execution, LLM re-run, argument repair, recorded-output substitution, ask the user), rationale, hard/soft preconditions; several candidates tested from one checkpoint and compared with total experiment cost; recorded vs fresh replay | `strategies.py`, `engine.test_alternatives` |
| Live recovery | Prefix-only checks after each step; repair before dependants run (max 2 attempts) or escalate | `live.py` |
| Incident memory | Strategies that passed in similar tested incidents (same family, step and failed checks) | `memory.py` |
| Review queue | Confirm/correct a suspect in the UI → export → retrain into a new folder → promote only if not worse | `review_queue.py` |
| Reports | Markdown incident report; failure exported as a pytest regression test | `report.py`, `regression.py` |
| Explainer | Template narrative by default; optional local Ollama narrative. A QLoRA fine-tuning script for Kaggle exists but **no fine-tuned explainer has been trained or deployed** | `explain/narrator.py`, `scripts/kaggle_qlora_explainer.py` |
| Labels | Counterfactual labeller with a careful re-run fixer that never sees the expected answer, skips steps whose inputs are already broken, records label confidence | `labeler/` |
| Dashboard | Runs, incident workspace (checks, suspect, tape or dependency graph, evidence, alternatives, branch comparison, review, export), Live (live recovery, planted faults, integrations), Break it, Replay, Compare, Evaluation | `dashboard/` |

## Measured results

All numbers are from files in `data/` and carry their sample counts. Targets came from the project plan, not from the problem statement.

**GPT-6 Luna runs (`data/metrics.json`, final)** — 2,918 runs; test split 322 failed runs.

| Result | Value |
|---|---|
| Top-1 / top-3 localization (unseen prompt templates) | 100% / 100% |
| Fault types withheld from training | 92.2% top-1 (Transformer alone 33%, autoencoder alone 91%) |
| Trained on the sandbox only, tested on GPT-6 Luna runs | 96.1% top-1 (2,118 failures) |
| Failure detection AUROC | 0.9997 |
| GPT-6 Luna as judge on the same runs (100-run sample) | 86% whole trace, 72% step by step → Black Box 1.16× (2× target **not met**) |
| Tokens saved by partial replay | 63% from checkpoints alone (real tokens) |
| Diagnosis time | 27 ms per run (RTX 3050) |

**Local, zero-cost experiments**

| Result | Value | File |
|---|---|---|
| qwen2.5:3b natural failures (nothing planted; 107 of 160 runs failed; 88 labelled) | 98.9% top-1, AUROC 0.90. 87 of 88 origins are the router dropping arguments, so this is real but not diverse | `metrics_natural.json` |
| Hybrid judge on GPT-6 Luna test runs (100) | Black Box 100%, qwen alone 27%, Black Box top-3 + qwen 73% — the small judge hurts | `hybrid_judge.json` |
| Workflow never seen in training (expense agent, 125 failures) | 23% top-1 (rules 40%, random 12%), AUROC 0.58 — **weak transfer to new workflows** | `metrics_heldout_workflow.json` |
| Live recovery (88 faults planted mid-run, 96 clean runs) | 96.6% repaired before dependants ran; 0% false interruptions | `metrics_live.json` |
| Who&When public benchmark (181 failures, 22.3 steps on average) | Zero-shot ensemble 4.4% (random 11.6%); trained on Who&When (5-fold CV) 29.8%; GPT-6 Luna judge 37.6% | `benchmark_whowhen*.json` |

Calibration, abstention, confidence intervals, PR-AUC and the false-alarm rate on odd-but-correct runs are in `metrics_extra.json` and `metrics_hardneg.json` and on the Evaluation screen.

**Honest reading.** On its own workflow family Black Box is near-perfect, fast and free, generalises to new prompts, held-out fault types and other agent models, and proves its suspects by replay. A frontier LLM judge is close behind on short traces and ahead on long free-text conversations, and diagnosis does not transfer to a brand-new workflow without labelled runs from it. The live-recovery validators were written knowing the fault catalogue, so they are not "unseen" in a learning sense.

## Run

```powershell
python -m venv .venv            # Python 3.10-3.12
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env     # LLM_PROVIDER=sandbox: offline and free
cd dashboard; npm ci; npm run build; cd ..
.\.venv\Scripts\python.exe -m uvicorn blackbox.api.app:app --port 8010 --env-file .env
```

Open http://localhost:8010. For a clean demo database: `$env:BLACKBOX_DB="data/demo.db"` before starting (it seeds runs, including an older quote so a "recent cached quote" fix is tested and rejected).

The dashboard opens on the product homepage. Choose **Open workspace** for the execution overview, or **Record a run** to begin recording. Workspace sections have hash URLs (for example `/#runs`, `/#replay`, and `/#evaluation`), and individual investigations can be opened directly with `/#run/<run_id>`. A failed-run investigation shows its acceptance checks, the leading suspect, and an **Alternatives** panel where candidates are tested from the same checkpoint and compared; **Explore alternatives** opens the replay workbench for a custom patch. Suggestions stay labelled as untested until a replay checks their outcome. See `UI_AUDIT.md` for the interface review.

Wrap your own LangGraph agent:

```python
import blackbox
agent = blackbox.wrap(graph, "data/traces.db", name="my-agent", check=lambda state: state["answer"] == expected)
run = agent.invoke({"request": "..."})          # recorded with native checkpoints
fixed = agent.fork(run["run_id"], step_id=2, output={"fx": {...}})   # LangGraph re-runs what follows
```

`python examples/expense_agent.py` shows record → silent failure → diagnosis → fork → crash → resume.

## Zero-credit rule

`data/metrics.json` and `data/models/` are final (backups: `data/metrics_gpt6luna_final.json`, `data/models_final/`). Do **not** run `datagen --provider openai`, any `--judge openai`, or training into `data/models`. New experiments use the sandbox or Ollama and write to their own files. `.env` caps OpenAI spend at $0.50 counted from `LLM_BUDGET_START`.

## Commands (all local, $0)

```powershell
python -m blackbox.eval_natural --dataset data/dataset_ollama.db        # natural failures, existing models
python -m blackbox.hybrid --whowhen                                      # hybrid judge with local qwen
python -m blackbox.eval_extra                                            # calibration, abstention, CIs, baselines
python -m blackbox.hard_negatives                                        # false alarms on odd-but-correct runs
python -m blackbox.benchmarks.heldout_workflow                           # workflow never seen in training
python -m blackbox.live_eval                                             # live recovery
python -m blackbox.review_queue status                                   # reviewed training queue
python -m blackbox.mcp_server                                            # MCP tools (server must be running)
python -m pytest
```

## Tests

`python -m pytest` — 49 tests covering recording, replay modes, retries, redaction, acceptance checks, alternatives (pass and reject), live recovery, LangGraph fork/resume with parallel branches, the incident API, Claude Code ingestion and the original engine/API behaviour.
