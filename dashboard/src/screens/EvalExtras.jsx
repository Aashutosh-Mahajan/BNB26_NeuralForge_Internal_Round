import { useState } from "react";
import { money, nodeLabel, percent } from "../api";
import { HBars, Panel, Segmented } from "../ui";

const VOLUMES = [
  [1000, "1k"],
  [10000, "10k"],
  [100000, "100k"],
  [1000000, "1M"],
];

/* Daily cost of diagnosing every failed run with an LLM judge vs Black Box, from measured spend. */
export function CostAtScale({ m }) {
  const [perDay, setPerDay] = useState(100000);
  const groups = m.llm_cost?.by_purpose || [];
  const judged = m.baselines?.llm_all_at_once?.count || 100;
  const row = (purpose) => groups.find((g) => g.purpose === purpose);
  const once = row("judge_all_at_once"), step = row("judge_step_by_step");
  if (!once) return null;
  const perRun = [
    ["GPT judge, whole trace", once.cost_usd / judged, once.calls / judged],
    ["GPT judge, step by step", step ? step.cost_usd / judged : null, step ? step.calls / judged : null],
  ];
  return (
    <Panel title="Cost at scale" aside={`measured spend of the ${m.llm_judge_model || "LLM"} judge on ${judged} runs`}>
      <div className="alt-actions" style={{ marginTop: 0, marginBottom: 16 }}>
        <span className="label" style={{ margin: 0 }}>Failed runs to diagnose per day</span>
        <Segmented value={perDay} onChange={setPerDay} options={VOLUMES} />
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>API calls per run</th>
              <th>Cost per run</th>
              <th>Cost per day</th>
              <th>Cost per year</th>
            </tr>
          </thead>
          <tbody>
            {perRun
              .filter(([, cost]) => cost != null)
              .map(([label, cost, calls]) => (
                <tr key={label}>
                  <td>{label}</td>
                  <td className="num-cell">{calls.toFixed(1)}</td>
                  <td className="num-cell">{money(cost)}</td>
                  <td className="num-cell">{money(cost * perDay)}</td>
                  <td className="num-cell">{money(cost * perDay * 365)}</td>
                </tr>
              ))}
            <tr style={{ fontWeight: 700 }}>
              <td>Black Box (local GPU)</td>
              <td className="num-cell">0</td>
              <td className="num-cell">$0</td>
              <td className="num-cell">$0 API</td>
              <td className="num-cell">$0 API</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="small muted" style={{ marginTop: 12 }}>
        Judge prices are what GPT-6 Luna actually cost us ($0.10 / $0.50 per million tokens). gpt-5.4-nano costs about 2–2.5× more. Black Box
        needs a GPU or CPU but no API calls; it diagnoses a run in about {Math.round(m.latency_ms)} ms.
      </p>
    </Panel>
  );
}

export function NaturalLocal({ n }) {
  if (!n) return null;
  const r = n.report || {};
  const rows = [
    ...["ensemble", "m1", "m2", "m3"].map((k) => [k, r.models?.[k]?.top1]),
    ...Object.entries(r.baselines || {}).map(([k, v]) => [k, v?.top1]),
  ]
    .filter(([, v]) => v != null)
    .map(([k, v]) => ({
      label: { ensemble: "Black Box (ensemble)", m1: "Transformer alone", m2: "LightGBM alone", m3: "Anomaly detector alone", heuristic_rules: "Hand-written rules", random: "Random step", last_step: "Last step", first_tool_error: "First step with an error" }[k] || nodeLabel(k),
      value: v,
      kind: k === "ensemble" ? "ours" : "",
    }));
  return (
    <Panel title={`Natural failures from ${n.agent_models?.join(", ")} (run locally)`} aside="nothing planted; models never saw this agent model">
      <div className="metric-row" style={{ marginTop: 0 }}>
        <div>
          <strong>{n.natural_failures}</strong>
          <span>of {n.runs} runs failed on their own</span>
        </div>
        <div>
          <strong>{n.counterfactually_labelled}</strong>
          <span>labelled by counterfactual replay</span>
        </div>
        <div>
          <strong>{percent(n.top1)}</strong>
          <span>top-1 (true origin ranked first)</span>
        </div>
        <div>
          <strong>{n.auroc != null ? n.auroc.toFixed(3) : "—"}</strong>
          <span>AUROC, failed vs passed</span>
        </div>
      </div>
      <HBars rows={rows} />
      <p className="small muted" style={{ marginTop: 12 }}>
        Labels come from a careful re-run fixer that never sees the expected answer. Where the origin was labelled:{" "}
        {Object.entries(n.label_steps || {})
          .map(([k, v]) => `${nodeLabel(k)} ${v}`)
          .join(", ")}
        .
      </p>
    </Panel>
  );
}

export function HybridJudge({ h }) {
  if (!h) return null;
  const sets = [
    ["gpt6luna_test_split", "GPT-6 Luna test runs"],
    ["who_and_when", "Who&When"],
  ].filter(([k]) => h[k]);
  return (
    <Panel title="Hybrid judge: Black Box shortlist + local LLM" aside={`judge ${h[sets[0]?.[0]]?.judge_model || ""}, $0 API`}>
      <div className="grid-2">
        {sets.map(([key, label]) => {
          const x = h[key];
          return (
            <div key={key}>
              <span className="label">
                {label} · {x.runs} failed runs
              </span>
              <HBars
                rows={[
                  { label: "Black Box alone", value: x.blackbox_alone.top1, kind: "ours" },
                  { label: "Local LLM alone", value: x.local_llm_alone.top1, kind: "llm" },
                  { label: "Black Box top-3 → LLM picks", value: x.hybrid_blackbox_shortlist_plus_llm.top1, kind: "ours" },
                ]}
              />
            </div>
          );
        })}
      </div>
      <p className="small muted" style={{ marginTop: 12 }}>{h.protocol}</p>
    </Panel>
  );
}

const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
const ci = (c) => (c?.ci ? ` (95% CI ${pct(c.ci[0])}–${pct(c.ci[1])}, n=${c.n})` : "");

export function Reliability({ x }) {
  if (!x) return null;
  const a = x.abstention, c = x.calibration_p_fail, k = x.confidence_intervals_95;
  return (
    <Panel title="How much to trust a single diagnosis" aside={`GPT-6 Luna test split · threshold chosen on validation`}>
      <div className="metric-row" style={{ marginTop: 0 }}>
        <div>
          <strong>{pct(k?.top1?.value)}</strong>
          <span>top-1{ci(k?.top1)}</span>
        </div>
        <div>
          <strong>{c?.ece != null ? c.ece.toFixed(3) : "—"}</strong>
          <span>calibration error of "chance the run failed" (ECE; 0 is perfect)</span>
        </div>
        <div>
          <strong>{c?.pr_auc != null ? c.pr_auc.toFixed(3) : "—"}</strong>
          <span>PR-AUC for detecting failed runs</span>
        </div>
        <div>
          <strong>{pct(x.position_only_baseline?.top1)}</strong>
          <span>"guess by step position" baseline (shortcut check)</span>
        </div>
      </div>
      {a && (
        <p className="small" style={{ marginTop: 12 }}>
          <strong>Not-sure rule:</strong> below a ranking score of {pct(a.threshold_from_validation)} Black Box says "not sure" and shows its
          top 3. On test it answers {pct(a.test_at_threshold?.coverage)} of runs, and those answers are right{" "}
          {pct(a.test_at_threshold?.accuracy_when_answering)} of the time.
        </p>
      )}
      {x.whowhen_by_length && (
        <div className="table-wrap section-gap">
          <table>
            <thead>
              <tr>
                <th>Who&When trace length</th>
                <th>Cases</th>
                <th>Black Box</th>
                <th>Transformer (windows)</th>
                <th>Random</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(x.whowhen_by_length).map(([bucket, r]) => (
                <tr key={bucket}>
                  <td>{bucket} steps</td>
                  <td className="num-cell">{r.cases}</td>
                  <td className="num-cell">{pct(r.ensemble)}</td>
                  <td className="num-cell">{pct(r.transformer_windows)}</td>
                  <td className="num-cell">{pct(r.random)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

export function Robustness({ hard, heldout, live }) {
  if (!hard && !heldout && !live) return null;
  return (
    <Panel title="Robustness" aside="local experiments, $0 API">
      <div className="grid-2">
        {heldout && (
          <div>
            <span className="label">Workflow never seen in training ({heldout.failed_runs} failed runs)</span>
            <HBars
              rows={[
                { label: "Black Box", value: heldout.black_box?.top1, kind: "ours" },
                { label: "Hand-written rules", value: heldout.rules?.top1 },
                { label: "Random step", value: heldout.random?.top1 },
              ]}
            />
            <p className="small muted" style={{ marginTop: 8 }}>
              Expense agent built with stock LangGraph; AUROC {heldout.auroc?.toFixed(3)}. Per broken node:{" "}
              {Object.entries(heldout.per_fault_node || {})
                .map(([k, v]) => `${nodeLabel(k)} ${pct(v.top1)}`)
                .join(", ")}
              .
            </p>
          </div>
        )}
        {hard && (
          <div>
            <span className="label">Odd but correct runs (false alarms at p_fail ≥ {hard.alarm_threshold})</span>
            <div className="table-wrap">
              <table>
                <tbody>
                  {Object.entries(hard.categories || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td>{nodeLabel(k)}</td>
                      <td className="num-cell">{v.runs} runs</td>
                      <td className="num-cell">{pct(v.false_alarm_rate)} false alarms</td>
                    </tr>
                  ))}
                  <tr style={{ fontWeight: 700 }}>
                    <td>Planted failures detected</td>
                    <td className="num-cell">{hard.planted_failures?.runs}</td>
                    <td className="num-cell">{pct(hard.planted_failures?.detected)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
        {live && (
          <div>
            <span className="label">Live recovery during execution</span>
            <p>
              <strong>{pct(live.recovered_rate)}</strong> of {live.planted_failures} faults planted mid-run were repaired before dependent steps
              ran; <strong>{pct(live.false_interruption_rate)}</strong> false interruptions on {live.clean_runs} clean runs.
            </p>
            <p className="small muted">{live.limitation}</p>
          </div>
        )}
      </div>
    </Panel>
  );
}
