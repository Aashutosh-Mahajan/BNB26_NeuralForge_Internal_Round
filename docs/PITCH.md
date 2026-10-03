# Black Box — pitch, demo script and Q&A

## One line

> The final answer is wrong, but the final step isn't where the mistake began. Black Box ranks the earlier suspect, shows the evidence, and tests alternative fixes from a saved checkpoint without throwing away the work that was fine.

Lead with **"suspect found in 27 ms, zero API cost, tested by replay"**, not with "100%".

## Demo (7 minutes, offline)

Run in the offline sandbox (`LLM_PROVIDER=sandbox`) on a fresh demo database (`BLACKBOX_DB=data/demo.db`); no Wi-Fi or credits needed. Keep a recorded backup video. One person drives, one answers questions.

| Time | Screen | Do | Say |
|---|---|---|---|
| 0:00 | — | Show 40 lines of raw trace JSON | "Which step broke this? No error was raised." |
| 0:30 | Runs | Point at the failure tiles and "where failures start" | "Every failure is ranked automatically." |
| 1:00 | Live run | Run the finance task; plant "year-old FX quote at step 3" | "The agent finishes confidently. The answer is wrong." |
| 1:45 | Run page | Acceptance checks: answer, fresh quote, source agreement fail | "Independent checks reject it, not our model." |
| 2:15 | Run page | Leading suspect step 3; evidence; Dependencies view | "Steps 5, 6, 8, 9, 10 depend on it; 4 and 7 can be reused." |
| 3:00 | Alternatives | Tick retry, backup provider, recent cached quote; Test | "Each fix is a hypothesis tested from the same checkpoint." |
| 3:45 | Branch table | Retry and backup pass; the 14-day-old cached quote is **rejected** | "We show failed repairs too. Total cost of the experiment is here." |
| 4:30 | Run page | Report → download; Export test | "The incident becomes a report and a regression test." |
| 5:00 | Live run | Same fault with live recovery on | "Caught at step 3 and repaired before anything used it." |
| 5:45 | Break it | A judge picks a run, step and fault | "You chose the fault. It was never shown to the models." |
| 6:30 | Evaluation | Scorecard, robustness, Who&When, limitations | "Here is where it is strong, and where it is not." |

## Evidence slide

- GPT-6 Luna runs: top-1 100% on unseen prompts, 92% on fault types withheld from training, 96% when trained only on another agent model.
- Natural mistakes by a different model (qwen2.5:3b): 98.9% top-1 on 88 counterfactually labelled failures (mostly one failure mode).
- Baselines on the same test runs: hand-written rules 61%, random 10%, last step 0%.
- Live recovery: 96.6% of 88 mid-run faults repaired, 0 false interruptions on 96 clean runs.
- 27 ms per diagnosis, $0 per diagnosis; partial replay saves 63% of tokens.

## Honesty slide

- GPT-6 Luna as a judge got 86% on our traces: Black Box wins by 1.16×, not the 2× we targeted.
- Who&When (long multi-agent conversations): GPT-6 Luna judge 37.6%, Black Box trained on that data 29.8%; without training on it, 4.4%.
- A workflow never seen in training: 23% top-1 (rules 40%). Learned diagnosis needs labelled runs from a new workflow.
- A small local judge choosing from Black Box's top 3 made it worse on our traces (73% vs 100%).
- Live-recovery checks were written knowing the fault catalogue.

## Backup slides

Ablations, per-fault × family heatmap, calibration and the "not sure" rule, cost at scale, architecture, integration capability badges.

## Hard questions

| Question | Answer |
|---|---|
| 100% — is that leakage? | Labels, fault names and expected answers never reach the models (allowlisted fields, tested). Splits are by prompt template; held-out fault types and another agent model are reported separately. The weak spots (new workflows, Who&When) are shown too. |
| Every failure was planted? | No: 107 of 160 qwen2.5:3b runs failed on their own and were labelled by counterfactual replay with a fixer that never sees the expected answer. GPT-6 Luna made no natural mistakes in 793 runs. |
| Why not just ask an LLM? | We measured it: 86% vs 100% on our traces, at a cost per run and seconds per call, versus 27 ms and $0. On long free-text traces the LLM is ahead, so we show both. |
| Does a passing repair prove the root cause? | It supports the suspect. We say "supported by replay", show rejected candidates and keep the top 3 suspects visible. |
| Does it work on any agent? | Any compiled LangGraph graph: record, diagnose, checkpoint, fork, resume. Claude Code: record and diagnose only. Diagnosis quality on a brand-new workflow is weak until it has labelled runs. |
| LLM non-determinism? | Unchanged steps come from checkpoints and the cache; repeated replays give a Wilson interval and we say when repeats are deterministic. |
| No gold answer in production? | Acceptance checks are pluggable: freshness, source agreement, argument consistency, or your own `check=` function; outcome can be "unknown". |
| Can it read the model's thoughts? | No. It analyses what the agent exposes: prompts, tool calls, results, state. |
| Does it retrain after every run? | No. Reviewed runs enter a queue; a new model version is trained separately and promoted only if it is not worse on held-out data. |
| What did you build vs use? | Built: recorder, checkpoint replay, features, the three trained models and ensemble, labeller, verifier, alternatives, live recovery, LangGraph adapter, UI. Used: LangGraph, MiniLM, an NLI cross-encoder, LightGBM, PyTorch. |
