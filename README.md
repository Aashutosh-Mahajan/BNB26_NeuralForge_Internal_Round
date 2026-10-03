# Black Box

A local flight recorder and crash-investigation workbench for sandbox AI-agent workflows. Record a run, inject a realistic fault, inspect the evidence, replay from a checkpoint with a replacement output, and compare the traces.

This repository implements the first working vertical slice of the PRD. The agent is a **deterministic, offline Python graph**, and the active diagnosis method is an **observable-evidence heuristic**. It makes no LLM API calls and reports zero LLM tokens. It is not yet the PRD's trained Transformer/LightGBM/autoencoder ensemble or LangGraph integration.

## Run locally

Requirements: Python 3.10+ and Node.js 20+ (Node 22 recommended). No API key, GPU, Ollama server, or external database is required.

From PowerShell in the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn blackbox.api.app:app --reload --port 8010 --env-file .env
```

If Python is installed through the Windows Python install manager and is not on `PATH`, use its absolute `python.exe` path for the first command.

In a second terminal:

```powershell
cd dashboard
npm ci
npm run dev
```

Open [the dashboard](http://localhost:5174). The Vite development server proxies `/api` HTTP and WebSocket requests to port 8010. [Interactive API documentation](http://localhost:8010/api/docs) is available on the backend.

On macOS/Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

For one server without Vite:

```powershell
cd dashboard
npm ci
npm run build
cd ..
.\.venv\Scripts\python.exe -m uvicorn blackbox.api.app:app --host 127.0.0.1 --port 8010
```

The backend serves the built dashboard at [localhost:8010](http://localhost:8010). Restart the backend after the first dashboard build so static routes are registered.

## Docker

```powershell
docker compose up --build
```

Open [localhost:8010](http://localhost:8010). The multi-stage image builds the dashboard and serves it from FastAPI. A named volume preserves SQLite traces and evaluation artifacts. This is a single-user local workbench with mocked tools, not an authenticated production deployment.

## Try the investigation loop

1. The first startup seeds successful finance, SQL, document QA, and math examples plus a finance run with stale exchange-rate data. Seeding happens only when the database has no runs.
2. Open the failed finance run and inspect step 3, `currency_rate`. The diagnosis explains its freshness and source-disagreement evidence.
3. Open Replay, choose a suggested replacement output, and run the replay. The engine freezes the original run's time and preserves the original trace.
4. Compare the failed and repaired runs to see the first divergence, changed state, and outcome.
5. Use Break It on a successful run to create another failing trace. Fault availability depends on the selected node type.
6. Start a new run from Live to watch persisted step events arrive over WebSocket.

The supported prompts are small sandbox templates, not unrestricted natural-language agents. For example:

- Finance: `Convert INR 50,000 to USD and compute EMI for 12 months at 9%.`
- SQL: `Find the total sales in the north region.`
- Document QA: `How many days do I have to request a refund?`
- Math: `Buy 8 items at $12 each with a 10% discount. What is the total?`

The family and parsed parameters determine execution; unsupported wording may use family defaults. The REST API additionally accepts an optional `params` object for controlled experiments.

## Implemented and pending

| Area | Current implementation | Remaining PRD work |
| --- | --- | --- |
| Agent and tools | Four deterministic task families, local mocked tools, independent answer verification | LangGraph graph and OpenAI/Ollama provider adapters |
| Recorder | Per-step inputs/outputs/state, parent dependencies, SQLite checkpoints, content-addressed clean response cache | OpenTelemetry/OpenInference export and external-agent adapters |
| Fault injection | Twelve catalogued fault types, compatible-node validation, independent failed forks | Large varied natural-failure corpus and LLM counterfactual labeler |
| Diagnosis | Observable numeric evidence, freshness, source disagreement, historical divergence, ancestry, transparent factors | Frozen MiniLM/NLI features, Transformer, LightGBM, autoencoder, SHAP, calibrated ensemble |
| Replay | Output patches, checkpoint prefixes, cache reuse, changed-input execution, K variants, Wilson interval, trace immutability | Prompt/model/temperature interventions and stochastic provider execution |
| Comparison | Needleman-Wunsch node alignment and recursive JSON/state changes | Benchmark-scale trace ingestion |
| Evaluation | Measured synthetic sandbox evaluation with separate known/unseen-fault splits and simple baselines | Natural-failure, cross-provider, public-benchmark, LLM-judge and trained-ensemble evaluation |
| Interface | REST, durable WebSocket backlog/resume, investigation dashboard | Authentication, tenancy, production hardening |

Heuristic blame and failure scores are **uncalibrated scores**. Evidence factors are explicitly not SHAP values. A matching successful run is an available reference, not an oracle injected into the diagnostic model. Fault labels and gold answers are reserved for verification and evaluation.

The offline graph produces identical K replay variants. Its Wilson interval describes that batch's observed pass count; it is not evidence of independent stochastic trials or real-world reliability. No LLM tokens are consumed, so token-savings percentages remain `null`; step reuse is measured separately.

## Evaluation and tests

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m blackbox.evaluation --output data/metrics.json --tasks-per-family 8 --seed 42
```

The evaluator creates an isolated temporary SQLite dataset and does not modify your interactive demo traces. The evaluation page reads only the saved `data/metrics.json` artifact. Before an artifact exists, metrics are unavailable, not PRD target values. Known and unseen fault types are reported separately; natural failures and LLM-judge results remain unmeasured.

An optional lightweight trained baseline can be generated with `--train-baseline`. This experiment does not replace the default heuristic diagnosis and is not the PRD ensemble. The optional `ml` dependency group reserves libraries for later model development; installing it does not produce trained models.

The tests cover checkpoint/cache behavior, fault/replay correctness, evidence attribution, trace comparison, evaluation isolation, API validation, and durable WebSocket replay of missed events.

## Python SDK

The initial SDK wraps the built-in sandbox agent:

```python
import blackbox

agent = blackbox.wrap(blackbox.SandboxAgent(task_family="finance"))
run = agent.invoke("Convert INR 50,000 to USD and compute EMI for 12 months at 9%.")
print(run["run_id"], run["status"])
```

Passing an external LangGraph agent currently raises an explicit unsupported-adapter error. The API and SDK share the same recorder, SQLite store, and execution engine.

## API

All routes work both at the paths below and with an `/api` prefix.

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/health` | Readiness and execution mode |
| GET | `/faults` | Fault catalog with compatible node types |
| POST | `/runs` | Queue a background run; returns HTTP 202 and `run_id` |
| GET | `/runs` | Run summaries; accepts `limit` and `status` |
| GET | `/runs/{id}` | Full trace and diagnosis |
| WS | `/runs/{id}/stream?after=0` | Durable `step`, `complete`, or `error` events; `after` resumes from an event ID |
| POST | `/runs/{id}/inject` | Fork with `{ "step_id": 3, "fault_type": "stale_data" }` |
| GET | `/runs/{id}/suggest-fix?step=3` | Suggested output interventions |
| POST | `/runs/{id}/replay` | Fork with `{ "from_step": 3, "patch": { "output": {} }, "k": 5 }`; substitute the selected step's valid output schema |
| GET | `/compare?a={id}&b={id}` | Aligned trace changes and first divergence |
| GET | `/stats` | Recorded totals and failure components |
| GET | `/eval` | Saved evaluation measurements and provenance |

Run `total_time` is elapsed wall time in **milliseconds** (`total_time_unit: "ms"`), including graph execution, cache access, checkpoints, and synchronous recording callbacks; the final outcome-save operation is excluded. Step `latency_ms` measures actual uncached node execution and is zero for checkpoint, reused, patched, or cache-hit steps. These are different measurements: run duration is not the sum of step latencies. Evaluation reports diagnosis latency separately and leaves replay time-savings percentages unmeasured.

WebSocket events are written to SQLite before clients receive them. Late subscribers receive the whole backlog; a client may reconnect using its last `event_id`. Queued/running jobs interrupted by a server restart are marked `ERROR`, and an error event terminates subscribers. Use one API process for this local implementation.

## Configuration and layout

Copy `.env.example` to `.env` and pass `--env-file .env` to Uvicorn to load it. The four supported settings are `BLACKBOX_DB`, `BLACKBOX_METRICS`, `BLACKBOX_SEED_DEMO`, and `BLACKBOX_CORS_ORIGINS`. Defaults store traces in `data/traces.db`, read measurements from `data/metrics.json`, and seed the demo on an empty database. Secrets and local data are excluded from Git.

```text
blackbox/
  agent/          Sandbox tasks, deterministic graph, tools, verifier
  api/            FastAPI routes and persistent stream events
  features/       Observable diagnostic features
  explain/        Trace alignment and JSON differences
  injector/       Fault catalog
  models/         Optional trained baseline utilities
  recorder/       Recorder wrapper
  replay/         Replay statistics
  engine.py       Execution, fault injection, checkpoint replay
  storage.py      SQLite runs, checkpoints, response cache
  diagnosis.py    Transparent diagnostic baseline
  evaluation.py   Reproducible measured sandbox evaluation
dashboard/        React + Vite investigation UI
tests/            Backend and API regression checks
data/             Local traces and measured artifacts (generated)
```
