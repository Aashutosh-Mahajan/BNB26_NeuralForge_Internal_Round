# Black Box — Fixes & Additions Plan

_Last updated: 4 Oct 2026_

## Zero-credit rule (read first)

The existing results in `data/metrics.json` were produced with **GPT-6 Luna** and are final. Nothing in this plan spends OpenAI credits on regenerating data, re-judging or retraining.

**Cost tags used below**

- 🟢 **Free** — code/docs only, or runs on the sandbox LLM
- 🔵 **Local GPU** — runs on the RTX 3050 / Ollama, no API calls
- 🔴 **API credits** — avoid; listed only as optional, never required

**Do NOT run these commands (they bill OpenAI or overwrite measured results):**

| Command | Why not |
|---|---|
| `python -m blackbox.datagen --provider openai ...` | Regenerates the GPT dataset — billed |
| `python -m blackbox.evaluation ... --judge openai` | Re-runs the LLM judge — billed |
| `python -m blackbox.benchmarks.whowhen --judge openai` | Re-runs the Who&When judge — billed |
| `python -m blackbox.evaluation` **without** `--output data/metrics_new.json` | Overwrites `data/metrics.json` and drops the GPT-6 Luna judge numbers |
| `python -m blackbox.models.train` (into `data/models`) | Replaces the trained models behind every reported number |

**Safety settings in `.env`:**

```ini
LLM_PROVIDER=sandbox      # or ollama — never openai for experiments
LLM_BUDGET_USD=0.5        # hard cap, in case something calls OpenAI by mistake
```

Back up first: copy `data/metrics.json` → `data/metrics_gpt6luna_final.json` and `data/models/` → `data/models_final/`.

---

## A. Fixes (do first, ~2 hours total)

### Bugs

| # | Fix | Where | Cost | Effort |
|---|---|---|---|---|
| 1 | **Diagnosis crashes on runs longer than 24 steps.** `MAX_STEPS = 24`; `transformer.forward` raises on longer traces and `diagnosis.py` has no fallback. Fix at inference only: use M2+M3 for long runs (as `whowhen.py` does), or score overlapping 24-step windows with the existing Transformer. **No retraining.** | `models/ensemble.py`, `diagnosis.py` | 🟢 | 20 min |
| 2 | **Old runs replay on the currently configured model.** `engine._llm_for` uses `.env`'s `OPENAI_MODEL`, so a GPT-6 Luna run replays on gpt-5.4-nano (different model, cache misses, extra billing). Use the run's stored `model`; if that model isn't available, replay on the sandbox LLM. | `engine.py` | 🟢 | 20 min |
| 3 | Stale `.git/index.lock` blocks Git. Delete it. | `.git/` | 🟢 | 1 min |
| 4 | `LLM_PRICE_*` values are still GPT-6 Luna prices. Set them to gpt-5.4-nano's pricing so the cost tracker and budget cap stay correct. | `.env` | 🟢 | 5 min |
| 5 | `traces.db` mixes GPT-6 Luna and gpt-5.4-nano demo runs. Clean it, or show the model on each run in the dashboard. | `data/traces.db`, dashboard | 🟢 | 15 min |

### Credibility and consistency

| # | Fix | Where | Cost | Effort |
|---|---|---|---|---|
| 6 | Remove the docstring "No trained neural weights are shipped. The live diagnosis does not use this architecture." The Transformer is trained and carries 60% of the ensemble weight. | `models/transformer.py` | 🟢 | 2 min |
| 7 | Fix README numbers to match `metrics.json`: Who&When traces are **22.3** steps (not ~52), zero-shot ensemble is **4.4%** (not 3.3%), total spend is **~$0.65** including the Who&When judge (not $0.39). | `README.md` | 🟢 | 10 min |
| 8 | State clearly: "Benchmarks were run on GPT-6 Luna; the live demo uses gpt-5.4-nano." | README, dashboard header, pitch | 🟢 | 10 min |
| 9 | Don't claim temperature 0 for OpenAI runs (GPT-6 Luna rejects it). Reproducibility comes from K-replay with a Wilson CI. | README, pitch | 🟢 | 5 min |
| 10 | `data/` is git-ignored, so nobody can reproduce the numbers. Commit `metrics.json` plus the small model files (~16 MB), or attach them to a GitHub release. | `.gitignore` | 🟢 | 15 min |
| 11 | Run `pytest` once and make sure everything passes. | `tests/` | 🟢 | 10 min |

---

## B. Strengthen the core (biggest score impact, no OpenAI credits)

| # | Addition | Why it matters | Cost | Effort |
|---|---|---|---|---|
| 12 | **Natural failures from a real LLM, run locally.** `datagen --provider ollama` with `qwen2.5:3b`; a 3B model fails on its own. Label with the counterfactual labeler (careful-rerun fixer, free). **Evaluate with the existing trained models — no retraining** — and write to `--output data/metrics_natural.json`. | Answers the biggest criticism: "every failure was planted" | 🔵 | 2–3 h |
| 13 | **Hybrid judge using a local LLM.** Black Box shortlists the top 3 steps, then Ollama `qwen2.5:3b` picks one. Compare against Black Box alone and the local LLM alone. Present it as "LLM judges become cheaper and more accurate." | Addresses Who&When and the missed 2× target | 🔵 | 2 h |
| 14 | **Cross-model test on Ollama runs.** The models never saw qwen data. Report Top-1 on qwen failures as a third generalization result, next to unseen faults (92%) and cross-provider (96%). Comes free with #12. | More evidence of generalization | 🔵 | 30 min |
| 15 | **Long-trace inference via sliding window** (extends fix #1). Brings the Transformer back for long traces without retraining. Re-score Who&When locally (no `--judge`) into a separate output file. | Better Who&When number, safer `wrap()` | 🟢 | 1–2 h |
| 16 | Harder task variants (ambiguous prompts, conflicting documents) run on sandbox/Ollama only. | More realistic natural failures | 🔵 | 2 h |

> **Optional, uses API credits (🔴):** small gpt-5.4-nano natural-failure run (e.g. 50 tasks, `LLM_BUDGET_USD=0.2`). Do it only if #12 can't run locally.
>
> **Optional, local GPU only (🔵):** retraining with the new Ollama data to `--out data/models_v2`. Never overwrite `data/models`. Skip if time is short; #12–#15 don't need it.

---

## C. Demo & product additions ("wow" factor)

| # | Addition | Cost | Effort |
|---|---|---|---|
| 17 | **Wrap a stranger's agent live.** A stock LangGraph ReAct agent + 2 lines of `blackbox.wrap()`, running on the sandbox or Ollama LLM. Needs fix #1 first. | 🔵 | 1–2 h |
| 18 | **Audience "Break it".** A judge picks the fault and step → diagnosis in 27 ms → replay turns the run green. Run in sandbox mode. | 🟢 | already built; rehearse |
| 19 | **Blast-radius view.** Root step in red, the downstream steps it corrupted in orange, on the DAG. | 🟢 | 1–2 h |
| 20 | **Failure → regression test.** One click exports a confirmed failure as a pytest case (checkpoint + patch + assertion). | 🟢 | 1–2 h |
| 21 | **MCP server.** Claude/Cursor can ask "why did run R-123 fail?" and get the diagnosis from the existing API. | 🟢 | 1–2 h |
| 22 | **Cost-at-scale counter.** "At 100k runs/day: LLM judge $X/day vs Black Box ≈ $0," computed from the measured judge cost already in `metrics.json`. | 🟢 | 30 min |
| 23 | Alerts: Slack/webhook when p_fail crosses a threshold. | 🟢 | 1 h |
| 24 | OpenTelemetry export shown live in Phoenix or Jaeger. | 🟢 | 30 min |
| 25 | Narrative explainer in the demo: keep `EXPLAINER_NARRATIVE=template` (free). LLM narrative only via Ollama. QLoRA explainer is a stretch goal (Kaggle GPU). | 🟢/🔵 | 30 min |

---

## D. Presentation

| # | Item |
|---|---|
| 26 | Lead with **"verified root cause in 27 ms, zero API cost, proven by replay"**, not with "100%". |
| 27 | Evidence slide: unseen faults 92%, cross-provider 96%, rules 61%, random 10%, last step 0%, plus the new natural-failure number from #12. |
| 28 | Honesty slide: Who&When (29.8% vs 37.6%) and the missed 2× target, then the hybrid (#13) as the answer. |
| 29 | Backup slides: ablations, per-fault heatmap, cost breakdown, architecture. |
| 30 | Demo runs **offline** (`LLM_PROVIDER=sandbox`) so no Wi-Fi and no credits are needed; keep a recorded backup video. One person demos, one handles Q&A. |
| 31 | Rehearse the hard questions: 100% = leakage?, injected vs natural failures, why not just an LLM judge, Who&When loss, works on any agent?, LLM non-determinism, gold answer in production, what you built vs used off the shelf. |

---

## Suggested order

1. **Now (~2 h, 🟢):** Back up results → fixes 1–11.
2. **Next (~4 h, 🔵):** #12 + #14 (natural failures on Ollama) → #13 (hybrid judge).
3. **Demo polish (~3 h):** #17, #18, #19, #22.
4. **If time remains:** #15, #20, #21, #23, #24.
5. **Final:** slides (#26–#29) and rehearsal (#30–#31).

Total OpenAI spend for this plan: **$0**.
