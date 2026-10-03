import { useEffect, useState } from "react";
import { FlaskConical } from "lucide-react";
import { api, familyLabels, nodeLabel, percent } from "../api";
import { Empty, HBars, PageHead, Panel, Segmented, Stat } from "../ui";

const SPLITS = [
  ["test", "Unseen prompts"],
  ["unseen_fault", "Unseen fault types"],
  ["natural_failures", "Natural failures"],
  ["cross_provider", "Other agent model"],
  ["validation", "Validation"],
];
const SPLIT_NOTE = {
  test: "Broken runs whose task wording never appeared in training.",
  unseen_fault: "Fault types 10–12 (premature answer, memory overwrite, loops) were never shown to the models.",
  natural_failures: "Mistakes the agent made by itself, labelled by counterfactual replay. From the offline agent: GPT-6 Luna made none in 793 runs.",
  cross_provider: "Models trained only on the offline agent, tested on GPT-6 Luna runs they never saw.",
  validation: "Used to tune the ensemble weights; shown for completeness.",
};
const NAMES = {
  ensemble: "Black Box (ensemble)",
  m1: "Transformer alone",
  m2: "LightGBM alone",
  m3: "Anomaly detector alone",
  llm_all_at_once: "GPT-6 Luna as judge",
  llm_step_by_step: "GPT-6 Luna, step by step",
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
    <Panel title="Targets from the plan" aside="measured on held-out data">
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
                  {value}
                </td>
                <td>
                  <span className={"badge " + (ok ? "passed" : "failed")}>{ok ? "Met" : "Not met"}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {m.llm_judge_top1 != null && (
        <div className="note warn section-gap">
          <span>
            On our traces GPT-6 Luna is a strong judge ({percent(m.llm_judge_top1, 0)}), so Black Box wins by{" "}
            {m.llm_improvement_ratio?.toFixed(2)}×, not 2×. Black Box answers in {Math.round(m.latency_ms)} ms without an API call
            and proves its answer by replay.
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
  if (!m) return null;
  if (!m.splits)
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
        ...["ensemble", "m1", "m2", "m3"].map((k) => [k, data.models?.[k]?.top1]),
        ...Object.entries(data.baselines || {}).map(([k, v]) => [k, v?.top1]),
      ]
        .filter(([, v]) => v != null)
        .map(([k, v]) => ({
          label: NAMES[k] || nodeLabel(k),
          value: v,
          kind: k === "ensemble" ? "ours" : k.startsWith("llm") ? "llm" : "",
        }))
    : [];
  const ww = m.public_benchmarks?.who_and_when;
  const manifests = Object.values(m.provenance?.manifests || {});

  return (
    <>
      <PageHead title="How accurate is it?">
        Every number here is measured on runs the models never trained on and read from <code>metrics.json</code>.
        {manifests.length > 0 && <> Agent data: {manifests.map((x) => `${x.runs.toLocaleString()} runs from ${x.model}`).join(", ")}.</>}
      </PageHead>
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
              GPT-6 Luna judges were run on a sample of 100 broken runs to limit cost.
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
                label: k.startsWith("llm") ? "GPT-6 Luna as judge" : k === "in_domain_lightgbm_5fold_cv" ? "Black Box, trained on it" : k === "ensemble" ? "Black Box, no retraining" : nodeLabel(k),
                value: v.top1,
                kind: k.startsWith("llm") ? "llm" : k === "in_domain_lightgbm_5fold_cv" ? "ours" : "",
              }))}
            />
            <p className="small muted" style={{ marginTop: 12 }}>
              Long free-text conversations are much harder. A frontier LLM judge is ahead here; the paper reports 8–25% for older prompted
              LLMs.
            </p>
          </Panel>
        )}
      </div>

      {m.replay && (
        <Panel className="section-gap" title="Replay" aside={`${m.replay.attempted_runs} broken runs, top suspect fixed automatically`}>
          <div className="metric-row" style={{ margin: 0 }}>
            <div>
              <strong>{percent(m.replay.fix_success_rate, 0)}</strong>
              <span>automatic fixes that restored the answer</span>
            </div>
            <div>
              <strong>{Number(m.replay.avoided_steps_pct).toFixed(0)}%</strong>
              <span>steps not re-run</span>
            </div>
            <div>
              <strong>{Number(m.replay.tokens_saved_pct_without_cache).toFixed(0)}%</strong>
              <span>tokens saved by checkpoints</span>
            </div>
            <div>
              <strong>{Math.round(m.latency_ms)} ms</strong>
              <span>to diagnose one run</span>
            </div>
          </div>
        </Panel>
      )}
    </>
  );
}
