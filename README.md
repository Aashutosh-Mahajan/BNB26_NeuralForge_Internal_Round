# Black Box — a flight recorder for AI agents

Black Box records every step of an agent run, uses trained models to find the step that caused a failure, explains why, and proves the diagnosis by replaying the run from that step with a fix (re-running only the affected steps).

## What is implemented (PRD map)

| PRD | Status | Where |
| --- | --- | --- |
| FR-1 Test agent | LangGraph agent, 10 nodes (planner, router, tool, retriever, transform, memory, date_util, calculator, reasoner, final). Planner/router/reasoner/final are **LLM calls**; 4 task families with 12 natural-language templates each; independent verifier | `blackbox/agent/` |
| FR-2 Tools | Mocked, deterministic: time-varying `currency_rate` (+ backup feed), calculator, SQLite `sql_query`, `schema_lookup`, policy `doc_search` (current + archived docs), `date_util` | `agent/tools.py` |
| FR-3 Recorder | Span per step, checkpoint after every node, content-addressed cache `hash(node+inputs[+model+seed])`, OpenTelemetry/OpenInference export, `blackbox.wrap()` for the built-in agent **or any compiled LangGraph** | `engine.py`, `recorder/` |
| FR-4 Fault injector | 12 fault types; types 10-12 held out | `injector/catalog.py` |
| FR-5 Counterfactual labeler | Fix each step in turn (careful re-run or GPT fixer with the gold answer), replay K=3, label = earliest step with ≥2/3 passes | `labeler/` |
| FR-6 Features | 412-dim step vector: MiniLM (384) + node one-hot (8) + 20 numeric (freshness, cross-source, rolling z-score, NLI contradiction, consistency…) | `features/` |
| FR-7 Models | M1 Step Blame Transformer (3.3M params, BCE + Gaussian-smoothed CE), M2 LightGBM, M3 autoencoder (successful runs only), M4 Ochiai spectrum, validation-tuned ensemble + calibrated p_fail | `models/` |
| FR-8 Explanation | TreeSHAP reasons as sentences, data-flow chain, nearest success + first divergence, model votes, counterfactual result | `explain/evidence.py` |
| FR-9 Replay | Fork from checkpoint k; patch output / prompt / model / temperature; cache reuse; K variants; Wilson CI; token + time savings | `engine.py` |
| FR-10 Fix suggester | Retry, backup source, rolling-median historical value, matching run, hand edit in UI | `engine.suggest_fix` |
| FR-11 Trace diff | Needleman–Wunsch alignment, first divergence, DeepDiff state diff | `explain/trace_diff.py` |
| FR-12 Evaluation | Top-1/3, MRR, ±1, AUROC, F1, test / unseen-fault / natural / cross-provider splits, baselines incl. LLM judges, ablations, replay savings, latency | `evaluation.py`, `judge.py` |
| FR-13 API | All PRD endpoints + `/llm/status`, `/models/status`, `/runs/{id}/explain`, `/runs/{id}/spans`, `/dataset` | `api/app.py` |
| FR-14 Dashboard | 7 screens (Runs, Live, Diagnosis, Replay, Compare, Break it, Evaluation) | `dashboard/` |
| FR-15 Who&When | Adapter + zero-shot and in-domain 5-fold results | `benchmarks/whowhen.py` |
| FR-16 LLM explainer | Template / GPT narrative; QLoRA Qwen2.5-1.5B script for Kaggle + SFT export | `explain/narrator.py`, `scripts/kaggle_qlora_explainer.py` |

## Measured results (`data/metrics.json`)

Data: a sandbox dataset (2,452 runs, rule-based LLM stand-in with a 7% mistake rate) and a **GPT-6 Luna dataset** (2,918 runs / 29,110 steps, $0.39). GPT-6 Luna made 0 natural failures in 793 runs, so natural-failure numbers come from the sandbox agent. Models are trained on both datasets.

| Metric (GPT-6 Luna test split unless noted) | PRD target | Measured |
| --- | --- | --- |
| Top-1 localization | ≥ 60% | **100%** (322 runs; random 10%, rules 61%) |
| Top-3 | ≥ 85% | **100%** |
| Top-1 on unseen fault types | ≥ 40% | **92.2%** (Transformer alone 33%, autoencoder alone 91%) |
| Failure detection AUROC | ≥ 0.90 | **0.9997** |
| Cross-provider (trained on sandbox only → GPT runs) | — | **96.1%** of 2,118 failures |
| Natural failures (sandbox, counterfactual labels) | — | **100%** (31 runs) |
| Replay token savings vs full rerun | ≥ 30% | **63%** from checkpoints alone (real tokens), 98% with the cache |
| Diagnosis latency | < 1 s | **27 ms** / run (RTX 3050) |
| Beats LLM judge by 2× | 2× | **No: 1.16×** (GPT-6 Luna judge 86% all-at-once, 72% step-by-step) |

Public benchmark, Who&When (181 real multi-agent failures, ~52 steps): GPT-6 Luna judge **37.6%** step / 61% agent; Black Box trained in-domain (5-fold CV) 29.8% / 49%; sandbox-trained without retraining 3.3% (below random 11.6%).

Honest reading: on short, structured traces Black Box is near-perfect, fast (27 ms vs seconds per LLM call) and verifies its answer by replay, but a strong modern LLM judge is already good there, so the gain is 1.16×, not 2×. On long, free-text traces the GPT-6 Luna judge is currently better than our in-domain model. The 2× target from the PRD is not met.

## Run

```powershell
python -m venv .venv            # Python 3.10-3.12
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env     # LLM_PROVIDER=sandbox runs fully offline
.\.venv\Scripts\python.exe -m uvicorn blackbox.api.app:app --port 8010 --env-file .env
```

Dashboard: http://localhost:8010 (after `cd dashboard; npm ci; npm run build`) or `npm run dev` on :5174.

## Pipeline

```powershell
python -m blackbox.datagen --provider sandbox --noise 0.07 --tasks-per-family 200   # ~1 min
python -m blackbox.models.train --dataset data/dataset_sandbox.db                   # ~1-2 min on GPU
python -m blackbox.evaluation --dataset data/dataset_sandbox.db --ablations
python -m blackbox.benchmarks.whowhen
python -m blackbox.cost                                                             # GPT-6 Luna cost projection
```

### With GPT-6 Luna

Put `OPENAI_API_KEY` in `.env` (keep `LLM_BUDGET_USD` as a hard cap), then:

```powershell
python -m blackbox.datagen --provider openai --tasks-per-family 200 --workers 8 --label-limit 50
python -m blackbox.models.train --dataset data/dataset_sandbox.db --dataset data/dataset_openai.db
python -m blackbox.evaluation --dataset data/dataset_openai.db --cross-dataset data/dataset_sandbox.db --judge openai --judge-runs 100 --ablations
```

Projected spend for the whole pipeline: **about $0.8-$2.4** (`python -m blackbox.cost`).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```
