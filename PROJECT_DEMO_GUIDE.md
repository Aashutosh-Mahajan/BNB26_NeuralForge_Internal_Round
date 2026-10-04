# Black Box: understand the project and demonstrate it to judges

Based on the code and saved reports in this checkout, reviewed 4 October 2026.
This guide explains existing behavior; it does not change the application.

## 1. The project in one sentence

**Black Box helps developers discover where an AI agent went wrong, understand the evidence, and test a better approach without repeating all the work.**

An AI agent is a program that uses an AI model and tools to complete a task through several steps. It might read a request, choose a tool, fetch information, calculate, and write an answer. A language model is only one worker inside that program.

Think of a restaurant order. The waiter hears the order, the kitchen receives it, ingredients arrive, the chef cooks, and the waiter serves it. A bad meal might begin with the wrong ingredient delivery, even though the visible complaint happens when the meal is served. Blaming the last person is not enough.

Your project keeps a record of each handoff, identifies the likely bad handoff, and lets the team try another delivery without taking the customer's order again.

## 2. What the hackathon actually asks for

The problem statement asks for a debugging system for AI-agent executions. The currency converter is a demonstration workload; the debugger is the product.

| Requirement | Meaning in everyday language | What to show |
| --- | --- | --- |
| Execution data | Keep a useful history of both good and bad attempts. | Live recording and stored runs. |
| Learned failure diagnosis | Teach a model to rank suspicious steps using past executions. | Trained model status, suspect ranking, evaluation report. |
| Failure explanation | Give reasons backed by the recorded history. | The timestamp, arguments, outputs, and evidence panel. |
| Checkpointed replay | Save progress so an investigation can begin in the middle. | Replay from the suspected step. |
| Alternative execution | Try a different approach and see whether it helps. | Several tested alternative branches. |
| Model evaluation | Test whether the model finds known faults and handles held-out cases. | Evaluation splits, baselines, sample counts, limitations. |
| Trace comparison | Show what changed between the original and another attempt. | Before/after trace and branch comparison. |

An attractive dashboard communicates these abilities, but the judges need to see the recording, diagnosis, intervention, and verification actually connect.

## 3. Who would use it, and where?

The primary user is a developer or operator of an AI-agent application. Examples include a customer-support agent that reads policies, a finance assistant that calls APIs, a reporting agent that queries databases, and an expense-processing workflow.

Black Box runs beside or around that application. It is not a customer-facing replacement for the application. A support customer asks for a refund; the support agent handles the request; the engineering team uses Black Box to investigate an incorrect response.

The dashboard is the inspection desk. The backend, recorder, storage, checkpoints, diagnosis models, and verifier do the work underneath it.

## 4. The pieces and their jobs

| Piece | Analogy | Actual role |
| --- | --- | --- |
| Agent | Worker following an order | Executes the task through model calls and tools. |
| LLM | Worker interpreting or writing text | Plans, routes, reasons, and produces a final answer at designated nodes. |
| Tools | Equipment and reference sources | Fetch rates, retrieve documents, query data, calculate. |
| Recorder | Flight recorder | Saves observable inputs, outputs, dependencies, state, timings, errors, and usage. |
| Checkpoint | Saved game | Holds execution state at an intermediate point. |
| Diagnosis ensemble | Investigation team | Ranks steps that look responsible for a failure. |
| Evidence explainer | Investigator's report | Turns recorded facts into an understandable account. |
| Strategy catalogue | Repair playbook | Offers applicable alternative actions with conditions and provenance. |
| Replay engine | Experiment room | Forks an execution and tests the intervention. |
| Acceptance verifier | Independent examiner | Checks whether an outcome meets the task's requirements. |
| Incident memory | Notebook of previous repairs | Summarizes which tested strategies helped in similar incidents. |
| Human review queue | Reviewed teaching examples | Collects confirmed or corrected labels for a later training round. |

The useful sequence is:

**Record → check outcome → rank suspects → show evidence → propose alternatives → replay → independently verify → compare.**

The explanation is not the examiner. A persuasive explanation cannot make a failed repair pass.

## 5. How the API key and Black Box relate

An API key is an access credential, like a library card. It is not a model and does not know which execution step failed.

At an LLM node, the agent prepares a prompt and sends it to a model provider using the configured credential. The provider returns a response. Black Box records the prompt and observable response metadata available through the integration. At a tool node, it records the tool call and result instead.

Example:

1. You submit a finance task in Live execution.
2. The backend starts a run and gives it an ID.
3. The planner interprets the task using the selected agent provider.
4. The router selects a tool and its arguments.
5. Tools fetch the quote and perform calculations.
6. Each completed step gets recorded and checkpointed.
7. The dashboard receives step events over a live connection.
8. The outcome is checked, and the saved execution can be investigated.

**Black Box cannot see the model's private internal thinking.** It can inspect the messages sent, exposed outputs, tool use, state changes, latency, and usage. A count of reasoning tokens is a count, not a transcript of private reasoning.

It can identify a mistake from observable evidence: the input requested 12 months, but the router passed 24; the quote says it is a year old; the reported answer disagrees with the calculator. It does not need to see hidden thoughts to notice these contradictions.

The free sandbox simulates the supported agent nodes without an external LLM call. Ollama can provide a real local model. A configured remote provider can provide real billed model calls. State which mode you demonstrate. The demo FX sources are mocked, including the backup provider; they are not live trading feeds.

## 6. A complete running example

Use this task:

> Convert ₹50,000 to USD and compute EMI for 12 months at 9%.

The built-in finance workflow has these ten steps:

| Step | Name | Meaning |
| --- | --- | --- |
| 1 | Planner | Extract the amount, currencies, loan term, and interest rate. |
| 2 | Router | Decide which tool to use and pass the extracted values. |
| 3 | Currency rate | Fetch the exchange-rate result. |
| 4 | Formula search | Obtain the calculation method/reference. |
| 5 | Convert currency | Multiply the amount by the exchange rate. |
| 6 | Memory | Store the converted amount for later use. |
| 7 | Date utility | Prepare the term information. |
| 8 | Calculator | Compute the monthly payment. |
| 9 | Reasoner | Interpret the calculator's result. |
| 10 | Final answer | Return the answer. |

Suppose step 3 returns a plausible rate with an old timestamp. Nothing needs to crash. The rest of the agent can produce a fluent, confidently wrong answer.

That is a **silent failure**: the program runs, but an important requirement is violated.

The outcome checks can detect a wrong answer, an old quote, or disagreement between quote sources. The diagnosis model examines the observed execution and ranks step 3. You inspect step 3 and see the timestamp and rate. You then test a retry or backup quote.

For illustration only, if ₹50,000 is multiplied by a rate of 0.0119, the converted principal is $595. A rate of 0.0108 gives $540. The EMI is a later calculation and should not be confused with that converted principal. Use the numbers displayed by your actual run during the presentation.

## 7. Why replay does not mean restarting everything

A checkpoint saves state, not the LLM's private mind. Saved outputs and dependency information let the execution engine determine which work can be preserved.

In this workflow, changing step 3 affects steps **5, 6, 8, 9, and 10**. Steps 4 and 7 appear later in the timeline but do not depend on the rate. They can be preserved in a recorded investigation when their inputs remain unchanged.

Thus, replay can:

- Restore steps 1 and 2 from checkpointed history.
- Replace or re-execute step 3.
- Reuse independent steps 4 and 7 when applicable.
- Recompute the dependent results at steps 5, 6, 8, 9, and 10.
- Save a linked branch while retaining the original execution.

This distinction is valuable: **later in time does not automatically mean dependent on the faulty value.**

The system checks whether inputs changed; it does not assume that every output after a repair is reusable. Execution and cache metadata show what happened. A scheduled node may be served from cache, so distinguish scheduled/recomputed steps from actual new model calls.

Recorded mode preserves the historical context and reuses unchanged responses, helping isolate the effect of one change. Fresh mode re-executes steps from the fork point without the cache under current conditions. Fresh mode can do more work than recorded mode.

For a tool with external effects, such as sending an email or charging a payment, replay needs application-specific safeguards. The supported demo tasks are primarily read and calculate operations. Do not claim arbitrary side effects can safely be repeated.

## 8. What an alternative path really means

An alternative path is another way to satisfy the same requirement at a problematic point. It is not a different user task, a prettier explanation, or a random answer.

| Failure | Alternative | Why it might help | Why it might fail |
| --- | --- | --- | --- |
| FX API timed out | Retry the call within a bounded retry budget | Temporary outage might clear. | Outage persists. |
| Quote is stale | Fetch from the backup FX source | Another source may provide an acceptable quote. | It may be unavailable or disagree. |
| No provider is available | Use a recorded quote if the task permits it | Can provide an explicitly permitted approximation. | It is too old for a current-rate requirement. |
| Router passed the wrong amount | Rebuild tool arguments from the plan | Repairs a bad handoff deterministically. | The plan itself may be wrong. |
| Archived refund policy was retrieved | Retrieve current documents only | Uses the correct policy version. | No current document exists. |
| LLM invents a number after calculation | Re-run the LLM step with a stricter instruction | Makes the transformation more constrained. | The model still changes the value. |
| No compliant repair is available | Ask the user or return an informative failure | Avoids inventing missing information. | Requires human input rather than automatic completion. |

The current alternatives catalogue is authored, not an unrestricted LLM inventing arbitrary plans. It matches supported steps to strategies, checks applicability, and supplies rationale and provenance. Incident memory adds evidence about previously tested strategies.

Important card states:

- **Suggested:** applicable candidate, not yet shown to work.
- **Check condition:** a policy concern needs to be tested or reviewed.
- **Unavailable:** a required condition is missing.
- **Manual:** requires the user or escalation.
- **Passed:** tested branch met the relevant acceptance checks.
- **Rejected:** tested branch failed those checks.
- **Inconclusive:** the experiment did not establish an acceptable outcome.

A good demo includes a rejected alternative. A cached quote might produce a plausible number but fail freshness. Show that the old timestamp remains old; Black Box does not relabel historical data as current.

Reusing the known successful output of the same task is explicitly a **diagnostic substitution**, not proof of a practical repair. It asks whether changing that step could remove the failure. A real repair instead obtains acceptable information or fixes the operation.

## 9. How a problem at step 2 is caught while the agent runs

There are two different modes.

### Investigation after completion

The agent completes or fails, the history is recorded, checks assess the outcome, and the trained diagnosis ensemble ranks suspects. A developer investigates and tests a branch. This does not automatically rewind an unrelated running program.

### Live recovery in the supported built-in workflow

Enable the Live recovery checkbox:

1. Step 1 extracts the task constraints.
2. Step 2 finishes and returns tool arguments.
3. Prefix checks compare those arguments with the plan, using information already available.
4. If an argument is wrong, the repair playbook can rebuild the call from the plan.
5. The repaired result is checked again.
6. On successful recovery, the corrected result is committed before the dependent tool executes.
7. The history records the violation, attempted strategy, and recovery outcome.

The default repair attempt limit is two. If repair cannot succeed, an escalation is recorded. The current engine does not provide a universal guarantee that escalation halts every later node; do not present it as a general production kill switch.

**Live checks are authored validators, not a trained streaming diagnosis model.** They know supported checks such as argument consistency, quote freshness, source disagreement, archived policies, and value preservation. Unexpected errors outside those checks can still need later diagnosis or human investigation.

## 10. Every screen: purpose, significance, and demonstration

### Homepage

Purpose: introduce the product and its debugging loop. The right-hand trace is an illustration, not a live incident or a benchmark result.

Say: "This is a flight recorder for AI agents. We connect recorded execution history to diagnosis and tested alternatives."

Click Open workspace. Spend about 10 seconds here; avoid narrating every decorative element.

### Overview / Execution overview

Purpose: the operator's inventory of executions.

- Runs recorded: activity volume, not model-training dataset size.
- Failed runs: runs marked failed in the execution inventory.
- Time to diagnose: diagnosis timing; not the time to run the task or test every fix.
- Model/baseline localization: saved evaluation accuracy, not correctness established from the visible six demo rows.
- Leading failure suspects: where recorded failures concentrate; a starting point for investigation.
- Activity: passed and failed runs grouped over time.
- Recorded runs: task, result, suspect, usage/cost, and time; filters narrow the list.

Say: "This is our operations view. I can find failed attempts and open the recorded evidence. The accuracy card comes from a separate evaluation report."

A numeric final answer matching the expected answer and every policy check passing are not always equivalent. Use the investigation's acceptance-check panel to assess the full outcome; do not infer universal correctness from a green inventory badge alone.

### Live execution / Live run

Purpose: submit a supported task and watch its recorded steps arrive.

Fields: agent provider, task family, task text, optional planted demo fault, and live recovery. Provider mode affects cost and whether there is a real remote/local LLM call. Supported built-in task families are Finance, SQL, Document QA, and Math; this is not a general agent for arbitrary requests.

Say: "Each completed step produces a recorded event and a checkpoint. The browser shows those events; it is not watching hidden model thoughts."

For your first demo run, choose the free sandbox explicitly and leave live recovery off. Later repeat the planted fault with recovery on.

### Run investigation / incident workspace

Purpose: answer "What failed, where might it have started, what evidence supports that, and what can we test?"

- Capabilities: what this integration can actually record, diagnose, checkpoint, fork, or resume.
- Acceptance checks: the failed requirements, separate from suspect scores.
- Leading suspect: the diagnosis model's highest-ranked candidate, not guaranteed truth.
- Ranking score: prioritization score, not automatically a calibrated probability of guilt.
- Next suspect and Not sure: uncertainty matters; the system may abstain.
- Flight tape: chronological history; select a step to inspect it.
- Dependencies: relationships between steps; explains propagation and reusable work.
- Inspector: Output, Input, LLM prompt when recorded, and State change.
- Evidence: timestamp, consistency, contradiction, and other observed features.
- Model votes: how Transformer, LightGBM, and anomaly detector contributed.
- Alternatives: applicable strategies, preconditions, test controls, and results.
- Human review: confirm the suspected origin, correct it, or mark uncertain.
- Report and Export test: create a portable incident account and a regression artifact.

Say: "The bad answer appeared at the end, but the suspect is the quote step. Here is the original timestamp. Here are the dependent calculations. Let's test whether changing this step actually helps."

For an external crash, distinguish ERROR from a silent failed outcome. Its last good checkpoint can be kept for a supported SDK resume. An external run without an independent outcome checker must remain unknown rather than being declared correct.

### Alternatives panel inside the investigation

Purpose: your mentor's request made concrete: users can see the failed operation and competing alternatives together.

Select candidates; select Recorded or Fresh; choose branch count; click Test alternatives. The table shows original versus candidate branches, checks, outputs, recomputed/reused work, usage, and experiment cost.

Say: "We are not just recommending a fix. Each suggestion earns its result through a replay and separate checks. This candidate is rejected; this one passed."

A branch count of three means three executions of each candidate, not three different strategies. Repeating a deterministic sandbox execution does not create three independent reliability observations. Use one branch for a quick deterministic demonstration.

### Replay & alternatives workbench

Purpose: manually construct an intervention when the catalogue is insufficient, or inspect replay behavior closely.

Choose the run, select the step, choose a supported fix type, supply the output/instruction/model setting, and replay. The output patch is JSON: structured data passed to later nodes, not an arbitrary final answer to hide the bug.

The available fixes are output replacement, an added prompt instruction, and a model change. Prompt and model changes apply only to LLM steps; model selection also depends on provider support. The temperature control has been removed from the dashboard to keep the demonstration focused.

Say: "This is the custom experiment desk. We can change one step and observe the affected execution."

Show the restored, reused, patched, rerun, and cached distinctions. A reported saved percentage must retain its denominator: steps avoided, tokens avoided, and billed cost are different things. Sandbox estimated tokens are not actual remote API savings.

### Trace comparison / Compare two runs

Purpose: inspect before and after, not merely celebrate a green result.

Choose the original run and a linked replay branch. Inspect output differences, input/state differences, final answer, and outcome.

Say: "The quote changed here, the converted amount changed here, and the final answer changed downstream. Independent work stayed reusable."

The earliest displayed difference can be input or state metadata; it is not automatically the root cause. Read the row detail and the independent acceptance checks.

### Preparing a failed demonstration case

The separate fault-injection dashboard has been removed. For the demonstration, select a previously recorded failed run from Overview and inspect its evidence before presenting it.

If the case contains a deliberately planted fault, disclose that it is a controlled evaluation example. The diagnosis should identify the origin from the execution evidence, without receiving the injection label as an input. Backend fault-injection tooling remains available for evaluation preparation.

### Model evaluation

Purpose: show measured evidence and its boundaries.

- Top-1: correct labelled origin ranked first.
- Top-3: labelled origin included in the three candidates.
- AUROC: ability to rank failed runs above successful ones; not a percentage of repaired runs.
- Held-out prompts: task wording/templates not used in training.
- Held-out fault types: fault labels withheld, although authored symptom features can still anticipate them.
- Cross-provider: transfer to another agent model in the studied workflow.
- Natural failures: mistakes without planted faults; inspect diversity and label quality.
- Ablation: remove a feature group/model and measure what is lost.
- Public/new-workflow tests: evaluate transfer outside the familiar workflow.
- Calibration/abstention: assess scores and when to say Not sure.
- Live recovery report: supported rule-check recovery results, not evidence of a learned live detector.

Say: "These are saved measurements with sample counts. We perform well on the workflow family studied, and our transfer tests show where that advantage does not generalize."

## 11. The ML models in ordinary language

There is no need to claim you trained a replacement for a frontier language model. You trained a specialized diagnosis system.

1. **Semantic features:** turn visible text into numerical descriptions and capture useful signals such as age, missing values, disagreement, retries, and inconsistent arguments.
2. **Transformer:** looks at the sequence and context of steps, like reading a story to find where it went off track.
3. **LightGBM:** learns combinations of observed signals, like an experienced inspector noticing several warning signs together.
4. **Autoencoder:** learns familiar execution patterns and flags unusual ones, like noticing a machine behaving differently from its normal behavior.
5. **Ensemble:** combines the three rankings. The saved report uses weights 0.6, 0.3, and 0.1.

Model files are present in data/models in this checkout. Presence is not proof that a particular running server loaded them: check the UI's model status and any semantic-fallback warning.

The feature pipeline explicitly excludes fault labels, expected answers, and outcome labels from diagnosis inputs. Labels are teaching targets; they must not become clues given to the model at inference.

## 12. How learning from future runs works

Recording, remembering a repair, and training model weights are three different processes.

**Recording:** every run adds history to storage.

**Incident memory:** tested strategy outcomes can be summarized for similar incidents. This can inform suggestions without changing model weights.

**Training:** a human confirms/corrects origin labels in the UI. Reviewed examples are exported, a new model version is trained separately, and a promotion check compares it with the current version before activation. This is an explicit workflow; a new run does not silently retrain the model.

An injected fault gives a known training label. Natural failures need review or careful counterfactual labelling. A plausible model explanation is not sufficient ground truth.

Training asks: "Given these observable steps, which labelled step started the failure?" It does not ask the diagnosis model to memorize the final expected answer.

The explanation layer currently uses a template by default, with optional local-model narration. A fine-tuning script exists, but the README explicitly says no fine-tuned explainer has been trained/deployed. Do not claim otherwise.

## 13. Integration: how it becomes a real product

| Integration | Current scope | Honest demonstration |
| --- | --- | --- |
| Built-in tasks | Recording, diagnosis, supported alternative experiments, checkpoints, live checks. | Use the dashboard. |
| External LangGraph graph | Wrapper records executions; native checkpoint support permits SDK fork/resume where available. | Use examples/expense_agent.py; show linked runs and preserved work. |
| Graph without checkpoint support | Recording/diagnosis can be available; replay/resume is restricted. | Show capability badges. |
| Claude Code hooks | Ingest exposed session/tool events. | Show recorded events; do not promise full checkpoint restoration from hooks. |
| MCP clients | Ask Black Box for failed runs, diagnosis, explanations, supported fix tests, and failure statistics. | Explain it as tools exposed to a compatible assistant. |

MCP provides access to Black Box; it does not automatically give Black Box control over every internal execution of the calling assistant. A provider API key alone does not capture editor actions or create checkpoints in an unrelated application.

External fork/resume operations require the actual graph and native checkpoint state. The wrapper may use an in-memory native checkpointer, so do not assume SDK resume survives a process restart just because a trace is stored in SQLite. Persistence requires the application's suitable native checkpoint setup.

## 14. The strongest selling points

Lead with these, in order:

1. **From diagnosis to a tested intervention.** A suspect becomes an experiment, with visible acceptance or rejection.
2. **Dependency-aware replay.** Preserve unaffected work instead of blindly repeating the entire task.
3. **Silent-failure detection.** A plausible answer can be wrong because of a stale quote or archived policy.
4. **Visible competing alternatives.** Explain what changes, why, applicability, outcome, and cost.
5. **Independent verification.** A fluent explanation cannot approve its own repair.
6. **Bounded live recovery for supported checks.** Catch known classes of bad handoffs before their dependants run.
7. **Local specialized diagnosis.** Ranking does not require a new remote LLM call when the local ensemble is loaded.
8. **An incident-to-improvement loop.** Review labels, preserve tested repair history, export reports/regression tests, and evaluate a new model before promotion.

These are differentiators for the hackathon presentation, not a researched claim that no other product has them.

## 15. The results you can say, with their boundaries

The following are reported by the saved project artifacts; they were not re-run while preparing this guide.

| Reported result | Meaning and boundary |
| --- | --- |
| 100% top-1/top-3 on 322 held-out failed test runs | Strong on the studied workflow family and largely injected faults; not universal accuracy. |
| 92.2% top-1 on 128 held-out-fault runs | Transfer to withheld fault labels; some authored symptoms were anticipated. |
| 96.1% cross-provider top-1 on 2,118 failures | Studied model-provider transfer, not transfer to every new workflow. |
| About 27ms diagnosis on the recorded RTX 3050 setup | Local diagnosis latency, not complete repair time. |
| 63% checkpoint token savings in the saved report | Aggregate experimental result; a particular demo may save a different amount. |
| 96.6% live recovery on 88 planted failures; 0 false interruptions on 96 clean runs | Supported validators and planted fault catalogue; sample-limited results. |
| 23% top-1 on a held-out expense workflow | Weak transfer to a new workflow; the rules baseline achieved 40%. |
| 4.4% zero-shot top-1 on 181 Who&When cases | Weak transfer to long, unfamiliar conversation traces. |

The saved whole-trace LLM-judge result is 86% on a 100-run sample. The project reports a 1.16x ratio against its aggregate top-1, not a demonstrated 2x improvement. The sample populations differ, so do not present that ratio as a matched all-cases comparison.

Natural Ollama failures add realism, but the README notes that 87 of 88 labelled origins were the router dropping arguments. This is evidence for that failure type, not broad natural-failure diversity.

Use the weak results as an engineering conclusion: "Our next step is adapting and evaluating the diagnosis system with labelled traces from each new workflow."

## 16. Recommended seven-minute judge demonstration

### Before the presentation

- Start the backend and dashboard according to README.md.
- Select Offline sandbox explicitly to avoid network/cost dependencies.
- Check model-loaded status; avoid presenting fallback rules as the trained ensemble.
- Check that the semantic encoders loaded; fallback warnings affect interpretation.
- Rehearse a finance stale-quote example and verify the supported candidates.
- Have a saved failed incident and tested branch ready if live execution stalls.
- To show the rejected cached-quote case, use a demo database seeded with an older same-currency quote. Confirm the candidate exists before your slot.
- Keep the actual expected output from your run available; do not memorize illustrative numbers as if they were fixed.

No re-training or paid data generation is needed during the presentation.

### 0:00–0:30 — State the problem

Homepage. Say:

> An AI agent can complete ten steps and still produce the wrong answer because one intermediate result was bad. Logs tell us what happened. Our project connects those records to a suspect, the evidence, and tested alternatives.

### 0:30–1:15 — Produce a visible failure

Open Live execution. Select Finance and Offline sandbox. Use the example task. Choose Step 3 returns a year-old FX quote. Keep live recovery off. Run the agent and open its investigation after completion.

Say:

> This is a controlled fault. We know where we planted it for demonstration, but that label is excluded from diagnosis inputs. Notice that the agent can continue and still answer incorrectly.

If a run is too fast to narrate live, use its recorded tape; do not invent time delays or claim private thinking is visible.

### 1:15–2:15 — Show evidence and propagation

Point to the failed acceptance checks, leading suspect, original quote timestamp, and dependency view. Show which calculations depend on it.

Say:

> The final answer is the symptom. This quote is the suspected origin. These checks explain what requirement was violated; these arrows show where the value propagated.

### 2:15–3:45 — Test competing alternatives

In the incident's Alternatives panel, select the supported retry and/or backup-provider strategy. Include the cached-quote candidate if present and marked Check condition. Choose Recorded mode and one branch. Click Test alternatives.

Say:

> These are hypotheses. We test each from the same saved point. A plausible historical quote is rejected if it is too old. An acceptable fresh quote passes. We preserve the original evidence.

Show the branch table: result, failed checks, answer, reused work, new tokens, and cost. Say explicitly that the FX providers are mocked in this demo.

### 3:45–4:30 — Show replay/comparison

Use the Replay workbench for a catalogue suggestion or a valid custom output patch. Select one replay for the deterministic sandbox, then Compare the two runs. Alternatively choose an existing linked branch from the run selectors.

Say:

> We restored earlier progress, changed the suspected step, and propagated that change. The table shows what changed rather than only showing a success message.

Do not manually replace the final answer with the expected answer and call it a repair.

### 4:30–5:30 — Repeat with live recovery

Return to Live execution. Submit the same supported fault with Live recovery enabled. Inspect its recovered step and summary.

Say:

> Earlier we investigated after completion. Now a supported step check notices the stale result as soon as it returns, tries a bounded repair, and passes the repaired value to dependent steps. This is rule-based live recovery; the learned ranking is our separate diagnosis component.

For the specific step-2 question, plant wrong arguments at step 2 instead and show the repaired call being passed to step 3.

### 5:30–6:30 — Prove you evaluated the system

Open Model evaluation. Show a held-out split, a baseline, sample counts, and the weaker new-workflow result.

Say:

> We evaluated localization separately from repair. We perform strongly on our supported workflow family. New workflows are harder, so we show that limitation rather than claim universal accuracy.

### 6:30–7:00 — Close with the developer workflow

Return to the incident. Show human review, Report, Export test, or capability badges as time allows.

Say:

> The result is not just a log or a chatbot answer. It is a reproducible incident: evidence, a suspect, accepted and rejected alternatives, and a repair experiment that keeps unaffected work.

## 17. A shorter three-minute version

1. 20 seconds: explain the silent intermediate-error problem.
2. 40 seconds: open a prepared stale-quote incident and inspect its evidence.
3. 70 seconds: test one acceptable alternative and one rejected cached quote, if available.
4. 30 seconds: show reused work and before/after outcome.
5. 20 seconds: show one measured result with sample count and one honest limitation.

Do not spend the short slot touring all six screens. The guide explains all of them so you can answer follow-up questions.

## 18. Questions the judges may ask

**Is this just logging?**

Logging is the input. The added value is learned suspect ranking, evidence, checkpointed intervention tests, independent acceptance, and comparison.

**Does it read the model's thoughts?**

No. It works from observable messages, results, tool calls, state, and metadata.

**Is the diagnosis guaranteed?**

No. It ranks hypotheses, exposes uncertainty, and tests interventions. A passing branch supports a repair hypothesis; it does not prove a unique cause.

**Why not ask a large LLM to read the log?**

A large model can help and can outperform this system on unfamiliar long traces. Here the local diagnosis model provides fast ranking on studied workflows, while checkpoints, verification, and replay provide executable evidence beyond a narrative.

**Can it fix any agent?**

No. Recording, diagnosis, fork, and resume depend on the adapter and application state/checkpoints. Built-in tasks have deeper repair support than hook-only integrations.

**Do you need an expected answer?**

The demo has an independent task verifier. Real applications need their own acceptance criteria: current policy, correct schema, completed operation, valid arguments, etc. Without an independent check, record unknown rather than assume success.

**Is the converter the product?**

No. It is a small, explainable workload that makes error propagation and alternative testing visible. The recorder/debugger is the product.

**Does every run retrain your model?**

No. It is recorded immediately; tested repairs can enrich incident memory. Model weights change only through reviewed examples, explicit retraining, and gated promotion.

**Why three diagnosis models?**

They offer sequence context, learned feature combinations, and unusual-pattern detection. The ablation report should justify the combination; more models alone is not the selling point.

**What happens when all alternatives fail?**

Show rejection or inconclusive outcomes and ask for missing information or human help. Do not fabricate success. Live recovery records escalation when bounded attempts cannot repair a detected problem.

**Can we repeat it?**

Use the preserved original run, linked branches, recorded-mode cache, saved checkpoints, report, and exported regression artifact. Separate historical reproduction from fresh validation.

## 19. Avoid these statements

- "We see the LLM's internal thoughts."
- "It repairs every AI agent automatically."
- "Our 100% result means it works on every task."
- "A passed replay proves this is the only root cause."
- "These three deterministic repeats prove statistical reliability."
- "The mocked backup quote is a live market integration."
- "All recorded runs automatically train the model."
- "A known successful output substitution is a production repair."
- "The saved evaluation figure is necessarily the model currently loaded by this server."
- "A hook or API key gives full external-agent checkpoint control."

## 20. The story to remember

**The agent makes a mistake. Black Box preserves what happened. It identifies a suspect, shows the evidence, offers alternatives, tests them from a saved point, and reports which requirements pass or fail.**

The strongest presentation makes that entire loop visible in one coherent incident.
