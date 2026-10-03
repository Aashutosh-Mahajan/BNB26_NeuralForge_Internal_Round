import { useEffect, useState } from "react";
import { ArrowLeft, CheckCircle2, GitBranch, Sparkles, Zap } from "lucide-react";
import { api, diagnosisOf, familyLabels, money, nodeLabel, percent, short } from "../api";
import { Json, Panel, StatusBadge, Tape } from "../ui";

const LEGEND = [
  ["#ff5a3c", "Root cause"],
  ["#f0a020", "Affected downstream"],
  ["#2f9e64", "Not affected"],
];
const MODEL_NAMES = { m1: "Transformer", m2: "LightGBM", m3: "Anomaly detector" };

function Explain({ runId }) {
  const [text, setText] = useState(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => setText(null), [runId]);
  if (text)
    return (
      <div className="note">
        <Sparkles size={17} />
        <p className="narrative">{text.text}</p>
      </div>
    );
  return (
    <button
      className="btn"
      disabled={busy}
      onClick={async () => {
        setBusy(true);
        try {
          setText(await api("/runs/" + runId + "/explain"));
        } catch (e) {
          setText({ text: e.message });
        } finally {
          setBusy(false);
        }
      }}
    >
      <Sparkles size={16} /> {busy ? "Writing explanation…" : "Explain in plain English"}
    </button>
  );
}

function Inspector({ step }) {
  const [tab, setTab] = useState("output");
  useEffect(() => setTab("output"), [step?.step_id]);
  if (!step) return null;
  const tabs = [
    ["output", "Output"],
    ["input", "Input"],
    ...(step.prompt ? [["prompt", "LLM prompt"]] : []),
    ["state", "State change"],
  ];
  return (
    <Panel title={`Step ${step.step_id} · ${nodeLabel(step.node_name)}`} aside={step.llm_call ? "LLM step" : step.node_type + " step"}>
      <div className="chips">
        {step.llm_call && <span className="tag">{step.model}</span>}
        {step.llm_call && (
          <span className="tag">
            {step.tokens_in} in / {step.tokens_out} out tokens{step.tokens_estimated ? " (est.)" : ""}
          </span>
        )}
        {step.llm_call && <span className="tag">{money(step.cost_usd)}</span>}
        {!!step.parent_step_ids?.length && <span className="tag">uses step {step.parent_step_ids.join(", ")}</span>}
        {step.tool_error && <span className="tag" style={{ color: "var(--root)" }}>error</span>}
      </div>
      <div className="tabs" role="tablist">
        {tabs.map(([id, label]) => (
          <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </div>
      {tab === "output" && <Json value={step.output} />}
      {tab === "input" && <Json value={step.input} />}
      {tab === "prompt" && <pre className="json">{step.prompt}</pre>}
      {tab === "state" && <Json value={step.state_diff} />}
    </Panel>
  );
}

export default function RunDetail({ run, onBack, onReplay, onBreak }) {
  const d = diagnosisOf(run);
  const failed = run.status === "FAILED";
  const root = failed ? d.root_cause : null;
  const flow = failed ? d.evidence?.data_flow || [] : [];
  const [selected, setSelected] = useState(root?.step || run.steps?.[0]?.step_id);
  useEffect(() => setSelected(root?.step || run.steps?.[0]?.step_id), [run.run_id]);
  const step = run.steps?.find((s) => s.step_id === selected);
  const factors = (d.evidence?.factors || []).filter((f) => !f.feature.startsWith("node_") && f.feature !== "position");
  const maxContribution = Math.max(0.0001, ...factors.map((f) => Math.abs(f.contribution || 0)));
  const votes = d.evidence?.model_votes || {};
  const cf = d.evidence?.counterfactual;

  return (
    <>
      <button className="back" onClick={onBack}>
        <ArrowLeft size={16} /> All runs
      </button>
      <div className="page-head" style={{ marginBottom: 16 }}>
        <div>
          <h1 style={{ fontSize: 32 }}>{run.prompt}</h1>
          <p className="small">
            {familyLabels[run.task_family]} · {run.model || "sandbox"} · {(run.total_tokens || 0).toLocaleString()} tokens ·{" "}
            {money(run.cost_usd)} · run <span className="mono">{short(run.run_id)}</span>
            {run.fault_type && <> · injected fault: <strong>{nodeLabel(run.fault_type)}</strong> at step {run.label_step}</>}
          </p>
        </div>
        <div className="actions">
          <StatusBadge status={run.status} />
          {failed ? (
            <button className="btn primary" onClick={onReplay}>
              <GitBranch size={16} /> Test a fix
            </button>
          ) : (
            <button className="btn primary" onClick={onBreak}>
              <Zap size={16} /> Break this run
            </button>
          )}
        </div>
      </div>

      <div className={"verdict " + (failed ? "failed" : "passed")}>
        <div>
          {failed && root ? (
            <>
              <h2>
                <em>Step {root.step} · {nodeLabel(root.node)}</em> caused this failure.
              </h2>
              <p>{(d.evidence?.summary || "").replace(/^[\w ]+: /, "")}</p>
            </>
          ) : (
            <>
              <h2>The agent's answer was correct.</h2>
              <p>
                Final answer <strong>{JSON.stringify(run.final_answer)}</strong> matches the expected answer. Break this
                run to see a failure diagnosed.
              </p>
            </>
          )}
        </div>
        {failed && root && (
          <div className="gauges">
            <div className="gauge">
              <div className="stat-value tone-root">{percent(root.confidence, 0)}</div>
              <div className="stat-note">blame on this step</div>
            </div>
            <div className="gauge">
              <div className="stat-value">{percent(d.p_fail, 0)}</div>
              <div className="stat-note">chance the run failed</div>
            </div>
          </div>
        )}
      </div>

      <Tape
        title="Flight tape"
        steps={run.steps || []}
        selected={selected}
        onSelect={(s) => setSelected(s.step_id)}
        role={(s) => (!failed ? "fine" : s.step_id === root?.step ? "root" : flow.includes(s.step_id) ? "affected" : "fine")}
        flag={(s) => (failed && s.step_id === root?.step ? "Root cause" : null)}
        legend={failed ? LEGEND : [["#2f9e64", "Step completed"]]}
        footer={
          failed && flow.length > 1 ? (
            <>
              The bad output of step <strong>{root.step}</strong> flowed into steps <strong>{flow.slice(1).join(", ")}</strong>. They look wrong too,
              but they are symptoms. Click any step to inspect it.
            </>
          ) : (
            "Click any step to inspect its input, output and prompt."
          )
        }
      />

      {!(failed && root) && (
        <div className="section-gap">
          <Inspector step={step} />
        </div>
      )}
      {failed && root && <div className="grid-main section-gap">
        <div className="stack">
          {failed && root && (
            <Panel title={`Why step ${root.step}`} aside={d.method === "ensemble" ? "reasons from SHAP values" : "rule-based reasons"}>
              <ul className="reasons">
                {factors.slice(0, 5).map((f) => (
                  <li className="reason" key={f.feature}>
                    <div>
                      <strong>{nodeLabel(f.feature)}</strong>
                      <p>{f.description}</p>
                    </div>
                    <div className="bar" title={"Contribution " + f.contribution}>
                      <span style={{ width: (Math.abs(f.contribution) / maxContribution) * 100 + "%" }} />
                    </div>
                  </li>
                ))}
              </ul>
              {Object.keys(votes).length > 0 && (
                <>
                  <hr className="divider" />
                  <span className="label">What each model says</span>
                  <div className="votes">
                    {Object.entries(votes).map(([k, v]) => (
                      <div key={k} className={"vote " + (v.step === root.step ? "agree" : "")}>
                        <div className="who">{MODEL_NAMES[k] || v.model}</div>
                        <div className="what">Step {v.step}</div>
                        <div className="how">
                          {percent(v.blame, 0)} confident · weight {v.weight}
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
              <hr className="divider" />
              <dl className="kv">
                <dt>Closest successful run</dt>
                <dd className="mono">{d.evidence?.nearest_success ? short(d.evidence.nearest_success) : "none found"}</dd>
                <dt>First step that differs from it</dt>
                <dd>{d.evidence?.diverges_at != null ? "Step " + d.evidence.diverges_at : "—"}</dd>
                <dt>Final answer given</dt>
                <dd className="mono">{JSON.stringify(run.final_answer)}</dd>
              </dl>
            </Panel>
          )}
          {failed && root && (
            <Panel title="Proof">
              {cf ? (
                <div className={"note " + (cf.verdict === "confirmed" ? "good" : "warn")}>
                  <CheckCircle2 size={18} />
                  <span>
                    Replaying from step {cf.from_step} with a fix: <strong>{cf.passed} of {cf.k}</strong> runs passed. Diagnosis{" "}
                    <strong>{cf.verdict}</strong>.
                  </span>
                </div>
              ) : (
                <div className="note">
                  <GitBranch size={18} />
                  <span>
                    Not tested yet. Use <strong>Test a fix</strong> to replay from step {root.step} with a corrected output. If the
                    answer becomes correct, the diagnosis is proven.
                  </span>
                </div>
              )}
              <div style={{ marginTop: 14 }}>
                <Explain runId={run.run_id} />
              </div>
            </Panel>
          )}
        </div>
        <Inspector step={step} />
      </div>}
    </>
  );
}
