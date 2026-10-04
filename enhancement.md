# Black Box — Enhancement and Hackathon Execution Plan

**Prepared:** 3 October 2026  
**Scope:** Review findings, product design, architecture, ML/data improvements, and an implementation roadmap.  
**Authorization:** This document is a proposal. Application code, model training, paid API calls, and external integrations require separate authorization before implementation.  
**Objective:** Build a credible, memorable demonstration that satisfies the problem statement and withstands technical questioning. No feature list or plan can guarantee a hackathon win.

## 1. The central recommendation

Keep the existing project. Strengthen its evidence, trained diagnosis, and replay capabilities before adding more models or integration logos.

The strongest version of Black Box should demonstrate this complete loop:

> Record an agent execution → detect failure → rank likely error origins → explain the evidence → propose alternative actions → test them from a checkpoint → independently verify the outcome → compare the executions.

The most urgent gap is that the configured trained-model artifacts are absent from this checkout. The second is that patched checkpoint replay supports the built-in agent, while the external LangGraph wrapper supports recording rather than equivalent replay. The third is that users need a unified view of the failure and its alternatives.

Recommended positioning:

> **Black Box is an evidence-driven debugger for AI agents. It learns where executions go wrong and tests alternative actions from saved checkpoints while preserving unaffected work.**

Avoid presenting it as a system that reads an LLM's mind, repairs every agent automatically, or always proves a unique root cause.

## 2. What the problem statement expects

The attached statement asks for a debugging system that learns from successful and failed agent traces. It should identify suspicious or failure-causing steps, support replay and alternative execution, and measure the quality of its learned diagnosis.

In everyday language: an agent can make a mistake early, continue doing plausible work, and produce a wrong result much later. Black Box helps a developer find where the problem started and investigate whether a change there fixes the result.

| Required capability | What judges should be able to see |
| --- | --- |
| Execution data | Inputs, outputs, tool activity, retrieved context, state changes, and outcomes from both successful and failed runs |
| Learned failure diagnosis | An actual trained artifact ranking suspicious steps from observable execution data |
| Evidence-based explanation | Recorded facts supporting each suspicion, with links to the relevant events |
| Checkpointed replay | An execution restored to a meaningful intermediate state |
| Alternative execution | A change at a suspected step, followed by observation of its downstream effect |
| Model evaluation | Held-out known-failure and previously unseen-failure results, with transparent protocols |
| Trace comparison | A clear view of what changed, what was reused, and why the outcome differed |

Automatic real-time repair, fine-tuning an LLM, a dashboard, and integrations with Codex/Claude Code/Devin are useful product choices, but the screenshot does not explicitly require all of them.

The README's numerical PRD targets are additional project targets. Do not describe them as thresholds imposed by the uploaded problem statement.

## 3. Verified status of this checkout

These findings come from the repository review and existing offline tests conducted in this conversation. Dashboard source was inspected; the dashboard was not visually tested. Paid LLM calls and external-agent integrations were not exercised.

### 3.1 What already exists

- A built-in ten-node agent for finance, SQL, document QA, and arithmetic tasks.
- A recorder saving per-step inputs, outputs, state snapshots, dependencies, and checkpoints.
- A content-addressed response cache and selective reuse during built-in replay.
- Fault injection, counterfactual labeling, feature extraction, and training code.
- Transformer, LightGBM, autoencoder, and ensemble implementations.
- Heuristic diagnosis with observable evidence and trained-ensemble loading when artifacts are available.
- Retry, backup FX, historical-value, and matching-success fix suggestions.
- Output, prompt, model, and temperature interventions.
- Trace comparison, evaluation code, and a seven-screen dashboard implementation.
- External compiled-LangGraph recording through `blackbox.wrap()`.
- Optional template, hosted-LLM, and QLoRA-adapter explanation paths.

**Validation:** All 41 existing offline tests passed. They cover important built-in recording/replay and API behavior. They do not establish trained-model accuracy, production generalization, dashboard usability, or arbitrary external-agent restoration.

### 3.2 The evidence mismatch that must be resolved

| Item | Observed local evidence | Required response |
| --- | --- | --- |
| Active provider | `.env` selects the sandbox | Label offline and real-model executions clearly |
| Trained ensemble | Configured `data/models` artifacts are absent | Recover existing artifacts or train and evaluate a reproducible version |
| Trace database | Five demo runs: four passed, one injected failure | Do not confuse demonstration traces with a training corpus |
| Saved evaluation | `observable_evidence_heuristic`; `heuristic_not_trained` | Present it as a rules baseline |
| Saved test localization | Top-1 75.8%, top-3 89.4%, on 66 failed synthetic test runs | Include sample count, method, and synthetic scope |
| Saved detection AUROC | Approximately 0.939 | Explain that AUROC measures ranking across thresholds, not 93.9% accuracy |
| Larger datasets and trained results in README | Not substantiated by the files available here | Recover provenance or revise the claims |
| QLoRA explainer | Training/loading code exists; deployment is unverified | Do not advertise a fine-tuned model based on a script alone |

The saved report itself states that natural failures, cross-model testing, a public benchmark, LLM judges, and a trained neural ensemble were not measured. The larger datasets described in the README are not present in this checkout.

Experiments may exist elsewhere. Their absence here does not prove they never happened; it prevents this checkout from reproducing or supporting the claims.

### 3.3 Requirement coverage

| Requirement | Coverage | Main gap |
| --- | --- | --- |
| Capture execution data | Implemented for built-in agent; basic external wrapper | External exceptions, real dependencies, reducers, and richer event capture |
| Train diagnosis model | Code exists | Packaged, evaluated, loadable trained artifacts |
| Explain with evidence | Implemented | Semantic evidence, event references, and uncertainty |
| Replay intermediate states | Implemented for built-in agent | Complete external checkpoint/control adapter |
| Test alternatives | Basic suggestions and interventions | Executable strategy branches and side-by-side outcomes |
| Evaluate learned diagnosis | Pipeline exists | Reproducible trained-model, real-failure, and workflow-held-out results |
| Compare traces | Implemented | A compelling integrated repair comparison view |

## 4. How the agent, API key, and Black Box relate

An API key is a credential. It lets the application call a model provider; it does not think, execute a task, or connect the recorder automatically.

There are separate responsibilities:

| Component | Responsibility |
| --- | --- |
| Task LLM | Interpret the request and return decisions, tool requests, or answers |
| Agent runtime | Execute tools, update state, and control the workflow |
| Recorder | Capture observable activity and state at useful boundaries |
| Diagnosis model | Rank likely failure origins from recorded evidence |
| Repair proposer | Generate feasible alternative actions |
| Replay controller | Restore state and execute a candidate branch |
| Outcome verifier | Independently check whether the task meets its requirements |
| Dashboard | Explain the incident and let users inspect/test alternatives |

The same provider could serve the task agent and repair proposer, but they are logically separate jobs. Recording, state restoration, dependency tracking, and arithmetic validation do not inherently require an LLM.

```text
User request
    ↓
Agent runtime ↔ Task LLM API
    ↕
Tools, documents, workspace, external services
    ↓ observable events and checkpoints
Black Box recorder → Trace store → Diagnosis
                                  ↓
                           Evidence + alternatives
                                  ↓
                         Replay + independent verifier
                                  ↓
                         Dashboard + incident report
```

### 4.1 What Black Box can see

Black Box can record what the integration exposes: prompts sent by the application, returned responses, tool requests and arguments, tool results, retrieved documents, application state, errors, timing, and reported usage.

It generally cannot inspect the model's private internal reasoning. A returned reasoning summary is an observable summary, not a complete internal reasoning trace. A model-written explanation is not proof of its internal decision process.

Example: the task asks for INR → USD, but the model returns a tool call with USD → INR. Black Box can detect the mismatch by inspecting the arguments; it does not need access to private thoughts.

If an opaque agent exposes only a final answer, Black Box cannot honestly identify hidden internal steps. The product must display that observability limitation.

### 4.2 Observable steps are not a predetermined plan

An agent may repeat a tool, branch, stop early, or create new actions dynamically. Use unique event IDs and node-attempt IDs, rather than assuming every agent always follows ten numbered steps.

Capture model responses and actions at useful boundaries without requesting private chain-of-thought. Store concise explicit decisions when the agent naturally returns them.

## 5. Flagship demonstration: a silent currency-rate failure

Use this example to teach the system. The following rates are invented demonstration values, not current market quotes.

Task: **Convert INR 10,000 to USD using a sufficiently fresh quote.**

| Step | Operation | Failure behavior |
| --- | --- | --- |
| 1 | Extract amount and currencies | Correct: INR 10,000 → USD |
| 2 | Fetch exchange rate | Returns stale rate: 0.015 USD per INR |
| 3 | Calculate | Correctly multiplies 10,000 × 0.015 = 150 |
| 4 | Format final result | Returns USD 150 |
| Verification | Check quote metadata and task conditions | Rejects stale quote |

The arithmetic step is correct given its input. The likely failure origin is the rate-fetching step. This demonstrates delayed consequences rather than merely catching an exception.

Black Box should show the timestamp, requested pair, actual tool arguments, source metadata, and dependency chain. If the available evidence does not establish that the rate is stale or wrong, the system must not pretend it knows.

For a controlled offline demonstration, use explicit fixtures and disclose them. For a real-provider demonstration, use an independent validation policy and avoid treating one provider as an infallible oracle.

## 6. Failure and alternatives: the mentor's requested experience

Build a unified incident workspace around the failure. Users should not need to navigate several screens to understand what happened and what they can try.

### 6.1 Failure card

The incident should show:

- Task and acceptance criteria.
- Run outcome: passed, failed verification, crashed, or outcome unknown.
- Leading suspect and top alternatives in the diagnosis ranking.
- Suspect status: observed error, learned suspicion, or replay-supported hypothesis.
- Recorded evidence with links to exact events.
- Affected downstream work and independent work that can be preserved.
- Available checkpoints and what each checkpoint can actually restore.
- Proposed repair strategies and their test status.

Example:

```text
Run failed its freshness requirement

Leading suspect: Fetch exchange rate
Evidence: Quote timestamp exceeds the allowed age
Affected work: Conversion calculation → Final answer

Alternative A: Retry for a fresh quote       [Test]
Alternative B: Use backup rate provider     [Test]
Alternative C: Use recent cached quote      [Review condition]

Original execution is preserved.
```

Do not display a normalized step-ranking score as an empirically calibrated probability of root cause. If calibrated confidence is unavailable, say “ranking score.”

### 6.2 Alternative candidate card

Each candidate needs:

| Field | Purpose |
| --- | --- |
| Strategy | What operation changes |
| Rationale | Why it addresses the observed failure |
| Preconditions | Currency pair, freshness, permissions, available tools, and user constraints |
| Intervention | Actual executable operation or explicit output substitution |
| Checkpoint | State from which the experiment starts |
| Expected recomputation | Which downstream work must run again |
| Status | Suggested, running, passed, rejected, inconclusive, or unavailable |
| Validation | Which acceptance checks passed/failed |
| Measurements | Calls, tokens, cost, time, and preserved work |

Distinguish **suggested** from **tested**. Distinguish a **tool-execution branch** from a **recorded-output intervention**. Both are legitimate experiments, but they demonstrate different levels of integration.

### 6.3 Alternative paths for the currency example

| Failure | Candidate strategy | Conditions |
| --- | --- | --- |
| Temporary timeout | Bounded retry, then backup provider | Retry budget and service availability |
| Stale quote | Fetch fresh quote from another provider | Timestamp meets task policy |
| Wrong direction | Correct arguments; or invert a verified reverse quote | Correct currency units and nonzero quote |
| Missing direct pair | Convert through an intermediate currency | Compatible timestamps and conversion conventions |
| No live providers | Recent cached quote | Approximation explicitly permitted |
| No compliant quote | Request clarification or return an informative failure | Never invent an acceptable quote |

An alternative must preserve the user's goal. Silently changing “today's quote” to an old estimate is not a successful repair.

The existing mocked backup source is useful for repeatable experiments, but is not evidence of an independent real-market feed. Historical medians must not mix different currency pairs, timestamps, or task contexts. Do not relabel an old value with a new timestamp.

### 6.4 How suggestions are produced

Use a layered approach:

1. Retrieve applicable strategies from a structured catalogue: retry, change source, repair arguments, change retrieval, or ask for missing information.
2. Retrieve similar verified incidents, matching task conditions rather than only node names.
3. Optionally ask an LLM to propose a structured candidate using recorded evidence and the allowlisted tools.
4. Validate the proposal's schema, feasibility, and task constraints.
5. Test candidates in bounded branches and independently validate outcomes.

A suggested candidate is a hypothesis. It should not be marked successful before execution.

### 6.5 Compare branches

Provide a table with original run, retry branch, backup branch, and any additional branch. Show pass/fail/unknown, final answer, failed checks, source/timestamp, changed nodes, reused nodes, actual resource use, and total experiment cost.

Include a failed repair in the demo. It shows that Black Box tests alternatives rather than decorating every suggestion with a green checkmark.

## 7. Replay and real-time recovery

### 7.1 Two modes with different promises

| Mode | Behavior | Priority |
| --- | --- | --- |
| Post-run investigation | Diagnose completed execution, fork repairs, compare outcomes | Core hackathon delivery |
| Live recovery | Validate steps during execution, pause affected work, repair, resume | Enhancement after core delivery |

The current live stream displays events. It is not an automatic recovery controller. Completed-run diagnosis does not automatically make a trained model suitable for online prefix diagnosis.

### 7.2 If step 2 fails in a ten-step task

1. Save state after step 1.
2. Execute step 2 and record the outcome.
3. Validate explicit requirements and score observable anomalies.
4. If necessary, pause work depending on step 2.
5. Select or test a repair candidate.
6. Restore state before step 2, or apply an explicit output intervention with recorded provenance.
7. Recompute all already-executed descendants whose inputs changed.
8. Preserve unrelated work only when dependencies and state isolation support it.
9. Independently verify the final result.
10. Preserve the original history and export the incident report.

If all later steps depend on step 2, steps 2–10 must run again. Saving unaffected work is a dependency property, not a promise that most steps are always reusable.

### 7.3 Checkpoint fidelity

For the built-in mocked workflow, saved application state can be enough. External agents may require messages, reducer state, runtime execution position, filesystem snapshots, Git state, tool configuration, environment versions, and external-resource references.

A trace log is not an executable checkpoint. A JSON snapshot is not automatically a restoration of a coding workspace.

### 7.4 External graph correctness

The existing external wrapper accumulates state with dictionary updates and approximates parents using previously observed nodes. Improve it to respect the runtime's state reducers, real causal dependencies, parallel branches, attempts, and exceptions.

Record stream failures as terminal incidents with their last valid checkpoint. Do not leave a crashed external run indefinitely marked as running.

### 7.5 Replay modes and cache behavior

Offer explicit modes:

- **Recorded investigation:** Reuse original tool observations to isolate a decision change.
- **Fresh validation:** Re-execute appropriate calls to check the candidate under current conditions.

Store code/tool versions, model configuration, context identity, relevant timestamps, and randomness metadata with cache identity. Choose TTLs for changing external data. Never treat a cached failure or an inappropriate stale response as a fresh retry.

Distinguish logical recomputation, local cache hits, and actual external calls. A node marked “rerun” may consume no new model tokens if served from a response cache.

Do not present repeated cached or deterministic executions as independent evidence of reliability. Report effective independent trials and the conditions under which confidence intervals are meaningful.

### 7.6 External effects and recovery limits

Track tool effect types: read-only, sandboxed write, idempotent write, and irreversible action. Use isolated workspaces for repair experiments. For writes, track operation identity and whether an effect already occurred.

Replay should not send a second email or duplicate a payment merely because execution returns to an earlier checkpoint. Human review may be a valid alternative when the runtime cannot safely restore an operation.

## 8. Integration roadmap

Separate observation, diagnosis, and execution control. Advertise capabilities individually.

| Adapter capability | Meaning |
| --- | --- |
| Record | Can ingest the agent's observable events |
| Diagnose | Can analyze those events with supported features |
| Checkpoint | Can save or reference resumable runtime state |
| Fork | Can create an isolated alternative execution |
| Resume | Can continue the supported execution |

### 8.1 LangGraph: first complete integration

This is the best next integration because the project already has a wrapper. Add native checkpoint references, correct state semantics, error capture, and branch/resume support.

Prove it on an external graph that is not one of the four built-in task families. Include a dependency branch, an independently defined success check, and a repair that changes a real tool operation.

LangGraph supports replay and forks from checkpoints. Native replay re-executes subsequent nodes; selective reuse requires additional correctness work.

### 8.2 Codex

One documented route is orchestrating Codex CLI through MCP with the OpenAI Agents SDK. Capture the observable orchestration and expose proposed Black Box tools for diagnosis and repair experiments.

Potential tool names, not existing implementations:

- `blackbox.record_event`
- `blackbox.diagnose_run`
- `blackbox.suggest_alternatives`
- `blackbox.test_alternative`
- `blackbox.compare_branches`

MCP provides a tool interface; it does not automatically expose every internal event or supply workspace rollback. Verify the selected runtime's event and control capabilities before promising them.

### 8.3 Claude Code

Assuming “Cloud Code” refers to Claude Code: use supported lifecycle hooks to capture tool inputs, successful results, and failures. Return concise debugging context where supported.

Logging hooks do not by themselves create full session/workspace restoration. Demonstrate capture first; add controlled replay only with an explicit restoration mechanism.

### 8.4 Devin

Assuming “derivity” refers to Devin: verify the chosen account/API's session events, exported information, and execution controls. Session APIs do not establish arbitrary internal checkpoint access.

If only session-level observation is available, label the integration accordingly. Do not infer replay support from the presence of an API key.

### 8.5 MVP integration decision

Deliver one complete external LangGraph integration. Treat Codex/Claude Code capture as a stretch goal. Defer broad Devin support until its exposed capabilities are established.

## 9. Model strategy

### 9.1 Train diagnosis, not a new foundation model

| Job | Initial implementation |
| --- | --- |
| Task execution | Existing hosted or local LLM |
| Failure localization | Trained LightGBM baseline; compare with sequence model |
| Novel anomaly scoring | Autoencoder or another successful-run anomaly baseline, if it adds value |
| Explanation | Evidence templates, optionally rephrased by an existing LLM |
| Repair proposal | Strategy catalogue plus optional constrained LLM proposal |

Train the simplest useful diagnosis model first. Package its artifacts, preprocessing, feature schema, data manifest, and held-out results. Add the Transformer/autoencoder ensemble only if it improves measured behavior enough to justify complexity.

The repository's QLoRA script targets an explanation model. Fine-tuning that explainer does not substitute for learned failure localization. Defer it unless explanation quality, offline operation, latency, or cost is a measured bottleneck.

### 9.2 Long and dynamic traces

The standard training path uses `MAX_STEPS = 24`, while the problem discusses long execution histories. Audit actual inference limits and supported lengths.

Compare length buckets, include 30–100+ step cases when feasible, and handle unsupported lengths explicitly. Increasing a position limit does not itself improve long-trace understanding. Explore hierarchical windows or dependency-aware features only after obtaining a baseline.

### 9.3 Distinguish prediction outputs

- Run-level failure probability requires run-level outcome labels and calibration.
- Step ranking requires localization labels.
- Anomaly scores identify unusual behavior, which is not always a failure.
- Replay success supports an intervention, but is not always proof of a unique historical cause.

Use top suspects and an abstention state. Show confidence calibration only when it has actually been measured. Evaluate explanations separately from diagnosis accuracy.

### 9.4 Online diagnosis needs prefix training

Completed traces contain future events, descendant information, and later contradictions. An online detector cannot use those facts at step 2.

For live detection, create prefix examples using only evidence available at the decision time. Measure detection delay, false interruptions, and recovery outcomes. Do not reuse complete-trace accuracy as a live-detection claim.

### 9.5 Model lifecycle

API calls do not automatically train Black Box or fine-tune the provider's model for this project.

The explicit lifecycle is: collect → verify outcomes → label → build splits → train → evaluate → package → activate. Newly observed runs should enter a reviewed training queue; model updates should be versioned and gated by evaluation.

## 10. Dataset and labeling improvements

### 10.1 Corpus composition

Include successful and failed runs across several workflows:

- Currency conversion: wrong direction, stale source, timeout, malformed metadata.
- SQL: schema drift, wrong filter, valid query returning the wrong scope.
- Document QA: archived policy, wrong retrieved document, unsupported conclusion.
- Coding workflow: failing test, incorrect file change, misunderstood requirement.
- Arithmetic: incorrect units, rounding policy, missing constraint.

Combine injected faults with genuine model/tool failures. Report their proportions separately. Include successful retries, unusual valid outputs, and harmless warnings as hard negatives.

An initial curated target of several hundred to roughly a thousand traces is a planning estimate, not a statistical guarantee. Prioritize diversity and reviewed labels over repeated variants of the same scenario. Keep an untouched external-workflow test set and expand according to observed error coverage.

### 10.2 Data fields

Store workflow ID, scenario ID, original-run group, provider/model, code/tool versions, event/attempt IDs, graph edges, request context, tool arguments/results, state/checkpoint references, outcome checks, failure category, label provenance, and repair outcomes.

Keep diagnosis input fields separate from training targets. Exclude success labels, expected answers, injected step labels, fault names, and ground-truth metadata from predictions.

The current allowlist/exclusion approach is a useful foundation. Audit indirect shortcuts: position, special error strings, fault-specific output shapes, and domain checks designed around injected faults.

### 10.3 Splits

Keep the successful parent run and every injected/replayed variant in the same data split. Group related scenarios across providers to avoid near-duplicate leakage.

Evaluate separately:

1. New examples of known faults.
2. Fault types withheld from training.
3. New task templates.
4. A workflow withheld from training.
5. A model/provider withheld from training.
6. Naturally occurring failures.
7. Later executions when temporal drift is relevant.

Held-out fault types can still be easy for authored rules that anticipate their symptoms. Acknowledge that difference when interpreting “unseen” performance.

### 10.4 Labels and counterfactuals

Injected origins provide controlled labels. Natural failures need evidence-based human review and carefully scoped repair experiments.

The current natural-failure fixer may use the correct final answer while producing offline labels. Disclose that access, and keep it out of live prediction and repair evaluation.

Changing a later answer to the gold answer can make a run pass without locating its origin. A changed early decision may repair a result for a different reason. Record uncertain labels, multiple contributing steps, and labeler disagreement rather than forcing every failure into a single definitive cause.

## 11. Evaluation that judges can trust

### 11.1 Diagnosis metrics

| Metric | Meaning |
| --- | --- |
| Top-1 localization | Correct labeled origin is ranked first |
| Top-3 localization | Correct labeled origin is among the first three suspects |
| MRR | How high the labeled origin appears in the ranking |
| Detection precision/recall | Whether failed runs are identified without excessive false alarms |
| AUROC/PR-AUC | Detection ranking quality across thresholds |
| Calibration | Whether displayed probabilities match empirical frequencies |
| Abstention coverage/accuracy | How often the system answers, and how reliable those answers are |
| Latency | Time to diagnose, with cold/warm/cached conditions separated |

Report sample counts and uncertainty alongside every result. Separate validation from the untouched test set. Avoid selecting weights or thresholds on the final test set.

### 11.2 Baselines

Compare trained models with random ranking, last-step ranking, first explicit error, authored heuristic rules, and an LLM judge under a disclosed input/budget protocol.

Include a label-only/position baseline where appropriate to expose step-position shortcuts. Run feature ablations to determine whether embeddings, rules, and sequence context contribute.

### 11.3 Repair metrics

- Percentage of failed runs restored to independently verified success.
- Candidate coverage: how often at least one feasible alternative exists.
- Rejected and inconclusive repairs.
- Actual external calls, model tokens, money, and wall-clock time.
- Unaffected work preserved and affected work recomputed.
- Total cost of all candidate experiments, including unsuccessful candidates.
- Correctness after recovery and any duplicated external effects.

Measure both per-candidate savings and total investigation cost. Five cheap forks can still cost more than one full rerun. A green acceptance check is the success criterion; a changed answer alone is insufficient.

### 11.4 Replay reliability

Repeated deterministic or cached outputs are not independent trials. Separate isolated decision experiments from fresh stochastic trials. State the verifier and trial protocol next to any success rate or confidence interval.

### 11.5 Public benchmarks

Who&When is relevant for agent/step failure attribution. Use it as an additional test of transfer, not as proof of replay correctness. Check current dataset availability, license, label definitions, and evaluation protocols before importing it.

Do not cite the README's benchmark numbers as verified local results unless their evaluated artifacts and protocol are recovered.

### 11.6 Reproducibility package

Ship data manifests/hashes, group split definitions, feature schema, seeds, model artifacts, versions, training command, evaluation command, hardware/runtime conditions, and generated metrics. Keep targets visually distinct from measurements.

## 12. USP features worth building

### A. Repair experiments with evidence

Show a suspected origin, the specific intervention, and the independent outcome check. This is the centerpiece of the pitch.

### B. Alternative branch comparison

Let users compare two or three valid strategies from the same checkpoint. Show rejected candidates and the reason they failed.

### C. Dependency-aware impact view

Highlight the suspect, affected descendants, and preserved branches. Pair the graph with concrete input/output differences so it remains understandable.

### D. Honest integration capability badges

Display “recording supported,” “checkpoint supported,” and “replay supported” per adapter. This makes scope understandable and prevents overclaiming.

### E. Incident memory

Retrieve similar verified incidents and their repair outcomes. Match relevant task conditions and show provenance. Reuse a strategy rather than blindly copying old output.

### F. Bounded live recovery

After post-run investigation is sound, add configurable pause thresholds, attempt/cost limits, and escalation when evidence is insufficient. Measure false interruptions.

Do not try to build all six before the core is proven. A–C are the most coherent demonstration.

## 13. Dashboard implementation proposal

Reuse existing screens and components. Improve the story through an incident workspace rather than adding unrelated pages.

| Area | Content |
| --- | --- |
| Header | Task, outcome, model/adapter status, acceptance check |
| Timeline/graph | Steps, attempts, errors, suspects, impact, checkpoints |
| Evidence panel | Exact arguments/results, timestamps, constraints, explanation provenance |
| Alternatives panel | Feasible strategies, preconditions, test controls, statuses |
| Comparison panel | Branch outcomes, diffs, reused/executed work, total experiment cost |
| Report action | Export a self-contained incident summary |

Support invalid JSON, unavailable providers, missing models, unknown outcomes, disconnected streams, and unsupported replay as explicit UI states. Never show every streamed step as healthy merely because it completed.

Reveal details progressively. The default view should explain the incident in plain language; raw trace JSON and model votes belong in expandable technical details.

### Final report example

> The run failed its freshness requirement. The leading suspect was the exchange-rate step because its timestamp exceeded the allowed age. A retry returned another invalid quote; the backup-source branch passed the acceptance checks. The initial request interpretation was reused, dependent calculations were recomputed, and the original execution was preserved.

Include unresolved limitations, verifier identity, model version, evidence links, and actual measurements. Avoid “root cause proven” when the evidence only supports a hypothesis.

## 14. Engineering improvements supporting the demo

- Version the trace/event schema and adapter capability contract.
- Capture real retries and attempts instead of relying on a fixed zero retry field.
- Preserve error type, tool/model identity, and last valid state when execution stops.
- Separate expected task failure from recorder/infrastructure failure.
- Validate patches and strategy requests before scheduling work.
- Persist branch lineage and immutable originals.
- Make long experiments resumable and cancellable; expose their statuses.
- Show model-loading and encoder fallback status prominently.
- Prevent fallback embeddings from silently masquerading as the trained representation.
- Redact credentials and unnecessary sensitive payloads in recorded/exported traces.
- Include workflow-specific verifier callbacks with pass/fail/unknown outcomes.
- Keep recording overhead and diagnosis latency distinct.

Use targeted tests for meaningful correctness risks: external reducers/parallel graphs, state restoration, downstream invalidation, exception capture, candidate constraints, independent trials, and duplicate-effect prevention. Do not equate a large test count with model quality.

## 15. Implementation phases and exit criteria

Relative order is intentional. Calendar estimates depend on team size, remaining hackathon time, available datasets, and API budget.

| Phase | Deliverables | Exit criterion |
| --- | --- | --- |
| 0: Evidence cleanup | Recover artifacts; inventory experiments; correct claims | Every displayed metric points to a reproducible artifact or is labeled unavailable |
| 1: Learned baseline | Curated dataset, grouped splits, trained ranker, evaluation | Application loads the artifact and predicts on untouched traces |
| 2: Alternatives experience | Unified incident view, executable strategy contract, branch comparison | At least two candidates can be tested with independent validation |
| 3: External replay | LangGraph adapter with native checkpoints and correct state semantics | A non-built-in graph can fail, fork, and recover while preserving its original |
| 4: Realism and robustness | Natural failures, longer/dynamic traces, cost accounting | Known limitations and held-out performance are documented |
| 5: Demo polish | Scripted demonstration, fallback fixtures, incident export | A fresh environment can run the full demonstration reliably |
| Stretch | Online prefix detector, coding-agent capture, incident memory | Core phases remain reproducible and tested |

### Minimum credible submission

- Actual trained diagnosis model and packaged artifacts.
- Captured successful and failed executions.
- Evidence-backed localization on held-out examples.
- Checkpointed repair experiments with at least two alternatives.
- Independent verification and original-versus-repaired comparison.
- Honest scope, measured results, and a reliable demonstration.

### Stronger submission

Add a complete external adapter, silent semantic failures, accurate dependency reuse, and a second workflow.

### Defer when time is limited

Defer explanation-only fine-tuning, integrations without usable events/control, universal autonomous repair, unnecessary infrastructure migration, and decorative metrics. Protect the complete investigation loop first.

## 16. Judge-facing demonstration script

1. Explain the problem: a plausible early mistake can cause a late wrong answer.
2. Show a genuine instrumented execution with model/provider and fixture labels visible.
3. Introduce a controlled silent failure or select a reviewed natural failure.
4. Let the independently defined acceptance check reject the result.
5. Show the trained model's ranking without exposing ground-truth labels to it.
6. Inspect the leading suspect's recorded evidence and affected descendants.
7. Test a retry branch and a genuinely different recovery strategy.
8. Show a rejected candidate as well as a successful one.
9. Compare preserved/recomputed work and actual total resource use.
10. Export the report and finish with held-out evaluation and limitations.

Keep an explicitly labeled offline fixture available if a provider is unavailable. Do not silently replace a failed live demo with a mocked execution.

Suggested opening:

> “The final answer is wrong, but the final step isn't where the mistake began. Black Box identifies the earlier suspect, shows the evidence, and lets us test alternative actions without discarding all the valid work.”

## 17. Questions judges may ask

| Question | Defensible answer |
| --- | --- |
| What did you train? | Name the deployed diagnosis model, its inputs, labels, artifact version, and held-out protocol |
| Is this just a logging dashboard? | Demonstrate learned ranking, evidence, intervention, replay, and independent verification |
| Can it read the LLM's thoughts? | No; it analyzes exposed actions, results, context, and state |
| How do you know the repair worked? | Show acceptance checks and branch results, not only the explanation |
| Is a successful patch proof of the root cause? | It supports the intervention hypothesis; unique causal attribution may remain uncertain |
| Can you rewind any coding agent? | Only where the adapter has sufficient runtime and workspace restoration capabilities |
| What does unseen mean? | State which faults, workflows, providers, or scenarios were excluded from training |
| Why not ask a strong LLM to debug the trace? | Compare against that baseline; explain measured advantages, if any, in latency/cost and replay verification |
| Are your tools real? | Identify real calls, mocks, cached observations, and output interventions explicitly |
| Does the system retrain after every run? | No; reviewed data enters a versioned training/evaluation process |

Do not claim superiority over a named competitor without a fair measured comparison.

## 18. Risks and decisions to track

| Risk | Response |
| --- | --- |
| Strong README claims lack local evidence | Recover provenance or revise claims before presenting |
| Synthetic shortcuts inflate accuracy | Add hard negatives, semantic faults, grouped splits, and workflow holdouts |
| Output replacement is presented as a real tool retry | Label intervention type; implement executable strategies |
| External state accumulation is inaccurate | Use native checkpoints and reducer/graph semantics |
| Replay duplicates writes | Isolate experiments and track side-effect identity |
| Cached variants inflate reliability | Distinguish effective independent trials and fresh-validation mode |
| Future information inflates live detection | Train/evaluate prefix-only examples |
| Repair changes the user requirement | Validate candidate preconditions and acceptance criteria |
| Savings hide unsuccessful experiments | Report total branch cost as well as individual savings |
| Fallback path hides absent models | Make diagnosis method and artifact status visible |

Before implementation, choose the time budget, API budget, first external workflow, acceptance checks, and whether the initial demo uses real services or disclosed fixtures. These choices should narrow the plan, not expand it indefinitely.

## 19. Repository map for the proposed work

| Area | Existing files |
| --- | --- |
| Recording/replay/fix suggestions | `blackbox/engine.py`, `blackbox/storage.py` |
| External wrapper and telemetry | `blackbox/recorder/sdk.py`, `blackbox/recorder/otel.py` |
| Diagnosis and features | `blackbox/diagnosis.py`, `blackbox/features/extractor.py`, `blackbox/features/semantic.py` |
| Models/training | `blackbox/models/train.py`, `blackbox/models/lgbm.py`, `blackbox/models/transformer.py`, `blackbox/models/autoencoder.py`, `blackbox/models/ensemble.py` |
| Dataset/labels | `blackbox/datagen.py`, `blackbox/labeler/counterfactual.py` |
| Evaluation | `blackbox/evaluation.py`, `blackbox/judge.py`, `blackbox/benchmarks/whowhen.py` |
| Explanations/comparison | `blackbox/explain/evidence.py`, `blackbox/explain/trace_diff.py`, `blackbox/explain/narrator.py` |
| API/events | `blackbox/api/app.py`, `blackbox/api/events.py` |
| User experience | `dashboard/src/screens/RunDetail.jsx`, `Replay.jsx`, `Compare.jsx`, `Live.jsx`, `Evaluation.jsx` |
| Claims/artifacts | `README.md`, `data/metrics.json`, configured model directory and dataset manifests |

This map identifies likely implementation locations; it does not authorize edits.

## 20. Sources and verification boundaries

Primary project evidence: source files, configuration values relevant to provider/model paths, the saved metrics report, read-only trace-database inspection, and the existing 41-test offline suite.

The attached hackathon screenshot is the source for required features. The local research report and README are background material, not independent proof of implemented capabilities or measurements.

Official references consulted in the preceding review:

- [OpenAI: Codex CLI with MCP and the Agents SDK](https://developers.openai.com/cookbook/examples/codex/codex_mcp_agents_sdk/building_consistent_workflows_codex_cli_agents_sdk) — a documented orchestration and tracing route; not a guarantee of arbitrary internal replay.
- [LangGraph: Time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel) — checkpoint replay and modified-state forks, with subsequent nodes re-executed.
- [LangGraph: Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) — checkpointers and persisted state.
- [Claude Code: Hooks reference](https://code.claude.com/docs/en/hooks) — lifecycle observation/control events; integration scope still needs validation.
- [Devin: API overview](https://docs.devin.ai/api-reference/overview) — session API context; checkpoint access must be verified separately.
- [Who&When research](https://arxiv.org/abs/2505.00212) — failure-attribution benchmark context, distinct from replay validation.

Recheck integration documentation at implementation time. Model names, prices, provider availability, and compatibility claims in project files require separate verification before budget or deployment decisions.

## Recommended next implementation scope

**First approval package:** align claims with artifacts, produce a reproducible trained baseline, and build the unified failure/evidence/alternatives experience with independently verified branch outcomes.

**Second approval package:** complete external LangGraph replay and demonstrate a second workflow.

**Stretch package:** bounded live recovery and coding-agent observation, supported by measured prefix detection and explicit adapter capabilities.

The project becomes compelling when users can see exactly what failed, inspect why it is suspected, test valid alternatives, and trust the evidence that the repaired execution meets their original goal.
