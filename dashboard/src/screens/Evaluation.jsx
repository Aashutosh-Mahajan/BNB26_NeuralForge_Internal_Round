import { useEffect, useState } from "react";
import { FlaskConical } from "lucide-react";
import { api, familyLabels, nodeLabel, percent } from "../api";
import { Empty, HBars, PageHead, Panel, Segmented, Stat } from "../ui";
import { CostAtScale, HybridJudge, NaturalLocal, Reliability, Robustness } from "./EvalExtras";

const SPLITS = [
  ["test", "Unseen prompts"],
  ["unseen_fault", "Unseen fault types"],
  ["natural_failures", "Natural failures"],
  ["cross_provider", "Other agent model"],
  ["validation", "Validation"],
];
const SPLIT_NOTE = {
  test: "Held-out task templates. Interpretation depends on the diagnosis method and report protocol.",
  unseen_fault: "Fault types withheld from the training split. Authored rules may still anticipate their symptoms.",
  natural_failures: "Failures that occurred without injected faults, with labels established through review or counterfactual experiments.",
  cross_provider: "Transfer to executions from another agent model. Refer to the report for training and test populations.",
  validation: "Development split used for model selection or calibration; separate from the final test set.",
};
const NAMES = {
  ensemble: "Black Box (ensemble)",
  m1: "Transformer alone",
  m2: "LightGBM alone",
  m3: "Anomaly detector alone",
  llm_all_at_once: "LLM judge · full trace",
  llm_step_by_step: "LLM judge · step by step",
  heuristic_rules: "Hand-written rules",
  first_tool_error: "First step with an error",
  random: "Random step",
  last_step: "Last step",
};

function Scorecard({ m }) {
  const rows = [
    ["Finds the guilty step (top-1)", "≥ 60%", percent(m.top1), m.top1 >= 0.6],
    ["Guilty step in the top 3", "≥ 85%", percent(m.top3), m.top3 >= 0.85],
    ["Top-1 on fault types never seen", "≥ 40%", percent(m.unseen_top1), m.unseen_top1 >= 0.4],
    ["Detects that a run failed (AUROC)", "≥ 0.90", m.auroc?.toFixed(3), m.auroc >= 0.9],
    ["Better than an LLM judge", "2×", m.llm_improvement_ratio ? m.llm_improvement_ratio.toFixed(2) + "×" : "not measured", m.llm_improvement_ratio >= 2],
    ["Tokens saved by partial replay", "≥ 30%", m.replay?.tokens_saved_pct_without_cache != null ? m.replay.tokens_saved_pct_without_cache.toFixed(0) + "%" : "—", m.replay?.tokens_saved_pct_without_cache >= 30],
    ["Time to diagnose one run", "< 1 s", m.latency_ms != null ? Math.round(m.latency_ms) + " ms" : "—", m.latency_ms < 1000],
  ];
  return (
    <Panel title="Project targets" aside="planning goals · not hackathon thresholds">
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Goal</th>
              <th>Target</th>
              <th>Measured</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([goal, target, value, ok]) => (
              <tr key={goal}>
                <td>{goal}</td>
                <td className="num-cell">{target}</td>
                <td className="num-cell" style={{ fontWeight: 700, fontSize: 15 }}>
                  {value ?? "—"}
                </td>
                <td>
                  <span className={"badge " + (value == null || value === "—" || value === "not measured" ? "error" : ok ? "passed" : "failed")}>{value == null || value === "—" || value === "not measured" ? "Unmeasured" : ok ? "Met" : "Not met"}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {m.llm_judge_top1 != null && (
        <div className="note warn section-gap">
          <span>
            Recorded LLM-judge top-1: {percent(m.llm_judge_top1, 0)}. Reported relative localization ratio: {m.llm_improvement_ratio != null ? m.llm_improvement_ratio.toFixed(2) + "×" : "unavailable"}. Replay is a separate intervention check.
          </span>
        </div>
      )}
    </Panel>
  );
}

function Heatmap({ cells }) {
  const faults = [...new Set(cells.map((c) => c.fault))];
  const families = Object.keys(familyLabels);
  if (!faults.length) return null;
  return (
    <Panel title="By fault and task type" aside="top-1 · number of runs">
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Fault</th>
              {families.map((f) => (
                <th key={f}>{familyLabels[f]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {faults.map((fault) => (
              <tr key={fault}>
                <td>{nodeLabel(fault)}</td>
                {families.map((family) => {
                  const c = cells.find((x) => x.fault === fault && x.family === family);
                  if (!c) return <td key={family} className="muted">—</td>;
                  const good = c.top1 >= 0.9, ok = c.top1 >= 0.6;
                  return (
                    <td key={family}>
                      <span
                        className="heat"
                        style={{
                          background: good ? "var(--pass-soft)" : ok ? "var(--affected-soft)" : "var(--root-soft)",
                          color: good ? "var(--pass)" : ok ? "var(--affected-ink)" : "var(--root)",
                        }}
                      >
                        {percent(c.top1, 0)}
                      </span>{" "}
                      <span className="small muted">n={c.count}</span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

export default function Evaluation() {
  const [m, setM] = useState(null);
  const [split, setSplit] = useState("test");
  const [error, setError] = useState("");
  useEffect(() => {
    api("/eval").then(setM).catch((e) => setError(e.message));
  }, []);
  if (error) return <div className="note bad">{error}</div>;
  if (!m) return <><PageHead title="Model evaluation">Loading the saved evaluation report.</PageHead><Panel><Empty title="Loading measured results…" /></Panel></>;
  if (m.status === "not_available" || !m.splits || !Object.keys(m.splits).length)
    return (
      <>
        <PageHead title="Evaluation" />
        <Panel>
          <Empty title="Not measured yet" icon={FlaskConical}>
            Run <code>python -m blackbox.evaluation</code> to measure accuracy. Targets are never shown as results.
          </Empty>
        </Panel>
      </>
    );

  const data = split === "cross_provider" ? m.cross_provider : m.splits[split];
  const rows = data
    ? [
        ...(!data.models?.ensemble && data.top1 != null ? [["current_method", data.top1]] : []),
        ...["ensemble", "m1", "m2", "m3"].map((k) => [k, data.models?.[k]?.top1]),
        ...Object.entries(data.baselines || {}).map(([k, v]) => [k, v?.top1]),
      ]
        .filter(([, v]) => v != null)
        .map(([k, v]) => ({
          label: k === "current_method" ? m.method === "ensemble" ? "Black Box" : "Black Box · rules baseline" : NAMES[k] || nodeLabel(k),
          value: v,
          kind: k === "ensemble" || k === "current_method" ? "ours" : k.startsWith("llm") ? "llm" : "",
        }))
    : [];
  const ww = m.public_benchmarks?.who_and_when;
  const manifests = Object.values(m.provenance?.manifests || {});

  return (
    <>
      <PageHead title="Model evaluation">
        Saved measurements, comparison baselines, and the limits of the evidence. Results come from the evaluation report.
        {manifests.length > 0 && <> Agent data: {manifests.map((x) => `${x.runs.toLocaleString()} runs from ${x.model}`).join(", ")}.</>}
      </PageHead>
      <div className="note section-gap" style={{ marginBottom: 22 }}><FlaskConical size={18} /><div><strong>{m.model_status === "heuristic_not_trained" ? "Rules baseline · no trained ensemble in this report" : "Saved evaluation artifact"}</strong><p>{m.provenance?.limitations?.[0] || "Review the dataset, split definitions, and diagnosis method before interpreting these measurements."}</p></div></div>
      <Scorecard m={m} />

      <h2 className="section-gap" style={{ fontSize: 26, margin: "28px 0 12px" }}>
        Accuracy by test set
      </h2>
      <div className="split-tabs">
        <Segmented
          value={split}
          onChange={setSplit}
          options={SPLITS.map(([id, label]) => [id, label, id === "cross_provider" ? !m.cross_provider : !m.splits[id]])}
        />
      </div>
      <p className="muted" style={{ marginBottom: 14 }}>{SPLIT_NOTE[split]}</p>
      {data ? (
        <>
          <div className="stats">
            <Stat label="Top-1" value={percent(data.top1)} tone="tone-pass" note="right step ranked first" />
            <Stat label="Top-3" value={percent(data.top3)} note="right step in the top 3" />
            <Stat label="AUROC" value={data.auroc != null ? data.auroc.toFixed(3) : "—"} note="telling failed from passed runs" />
            <Stat label="Runs" value={(data.count ?? 0).toLocaleString()} note="broken runs scored" />
          </div>
          <Panel title="Compared with other ways to find the step" aside="top-1 accuracy">
            <HBars rows={rows} />
            <p className="small muted" style={{ marginTop: 12 }}>
              {rows.some(r => r.kind === "llm") ? "LLM-judge measurements are shown only where recorded in this report." : "No LLM-judge measurement is recorded for this split."}
            </p>
          </Panel>
          {(split === "test" || split === "unseen_fault") && (
            <div className="section-gap">
              <Heatmap cells={data.heatmap || []} />
            </div>
          )}
        </>
      ) : (
        <Panel>
          <Empty title="Not measured for this set" />
        </Panel>
      )}

      <div className="grid-2 section-gap">
        {m.ablations && (
          <Panel title="What each part adds" aside="retrained without a feature group, or one model alone">
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Variant</th>
                    <th>Unseen prompts</th>
                    <th>Unseen faults</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(m.ablations).map(([k, v]) => (
                    <tr key={k} style={k === "full_ensemble" ? { fontWeight: 700 } : null}>
                      <td>{nodeLabel(k).replace("m1", "Transformer").replace("m2", "LightGBM").replace("m3", "anomaly detector")}</td>
                      <td className="num-cell">{percent(v.top1)}</td>
                      <td className="num-cell">{percent(v.unseen_top1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        )}
        {ww && (
          <Panel title="Public benchmark: Who&When" aside={`${ww.cases} real multi-agent failures, ~${Math.round(ww.mean_steps)} steps each`}>
            <HBars
              rows={Object.entries(ww.results).map(([k, v]) => ({
                label: k.startsWith("llm") ? "LLM as judge" : k === "in_domain_lightgbm_5fold_cv" ? "Black Box, trained on it" : k === "ensemble" ? "Black Box, no retraining" : nodeLabel(k),
                value: v.top1,
                kind: k.startsWith("llm") ? "llm" : k === "in_domain_lightgbm_5fold_cv" ? "ours" : "",
              }))}
            />
            <p className="small muted" style={{ marginTop: 12 }}>
              This benchmark measures failure attribution. It does not establish checkpoint restoration or repair correctness.
            </p>
          </Panel>
        )}
      </div>

      <div className="section-gap">
        <Reliability x={m.extras?.extra} />
      </div>
      <div className="section-gap">
        <NaturalLocal n={m.extras?.natural_local} />
      </div>
      <div className="section-gap">
        <Robustness hard={m.extras?.hard_negatives} heldout={m.extras?.heldout_workflow} live={m.extras?.live_recovery} />
      </div>
      <div className="section-gap">
        <HybridJudge h={m.extras?.hybrid_judge} />
      </div>
      <div className="section-gap">
        <CostAtScale m={m} />
      </div>

      {m.replay && (
        <Panel className="section-gap" title="Replay evaluation" aside={`${m.replay.attempted_runs ?? "—"} attempted runs`}>
          <div className="metric-row" style={{ margin: 0 }}>
            <div>
              <strong>{percent(m.replay.fix_success_rate ?? m.replay.success_rate, 0)}</strong>
              <span>automatic fixes that restored the answer</span>
            </div>
            <div>
              <strong>{m.replay.avoided_steps_pct != null ? Number(m.replay.avoided_steps_pct).toFixed(0) + "%" : "—"}</strong>
              <span>steps not re-run</span>
            </div>
            <div>
              <strong>{m.replay.tokens_saved_pct_without_cache != null ? Number(m.replay.tokens_saved_pct_without_cache).toFixed(0) + "%" : "—"}</strong>
              <span>tokens saved by checkpoints</span>
            </div>
            <div>
              <strong>{m.latency_ms != null ? Math.round(m.latency_ms) + " ms" : "—"}</strong>
              <span>to diagnose one run</span>
            </div>
          </div>
        </Panel>
      )}
    </>
  );
}
