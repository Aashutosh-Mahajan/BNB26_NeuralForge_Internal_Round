import { useEffect, useState } from "react";
import { AlertTriangle, ArrowLeft, Download, FileCode2, Sparkles, Zap } from "lucide-react";
import { api, diagnosisOf, familyLabels, money, nodeLabel, percent, post, short } from "../api";
import { BUILTIN_CAPS, Capabilities, DepGraph, Json, Panel, Segmented, StatusBadge, Tape } from "../ui";
import Alternatives, { CheckList } from "./Alternatives";

const LEGEND = [
  ["#ff5a3c", "Leading suspect"],
  ["#f0a020", "Depends on it"],
  ["#2f9e64", "Independent work"],
];
const MODEL_NAMES = { m1: "Transformer", m2: "LightGBM", m3: "Anomaly detector" };
const OUTCOME_TEXT = {
  passed: "Passed every acceptance check",
  failed: "Failed its acceptance checks",
  unknown: "Outcome unknown: no independent check",
};

function download(path, filename) {
  fetch("/api" + path)
    .then(async (r) => {
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "Download failed");
      return path.endsWith("regression-test") ? (await r.json()) : { filename, source: await r.text() };
    })
    .then(({ filename: name, source }) => {
      const url = URL.createObjectURL(new Blob([source], { type: "text/plain" }));
      const a = Object.assign(document.createElement("a"), { href: url, download: name || filename });
      a.click();
      URL.revokeObjectURL(url);
    })
    .catch((e) => alert(e.message));
}

function Explain({ runId }) {
  const [text, setText] = useState(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => setText(null), [runId]);
  if (text)
    return (
      <div className="note">
        <Sparkles size={17} />
        <p className="narrative">
          {text.text}
          <span className="muted small"> ({text.mode} explainer)</span>
        </p>
      </div>
    );
  return (
    <button
      className="btn"
      disabled={busy}
      onClick={async () => {
        setBusy(true);
        try {
          setText(await api("/runs/" + runId + "/explain?mode=template"));
        } catch (e) {
          setText({ text: e.message, mode: "error" });
        } finally {
          setBusy(false);
        }
      }}
    >
      <Sparkles size={16} /> {busy ? "Writing…" : "Explain in plain English"}
    </button>
  );
}

function Review({ run, root, onDone }) {
  const [wrong, setWrong] = useState(false);
  const [label, setLabel] = useState(root?.step);
  const [error, setError] = useState("");
  const send = async (verdict) => {
    try {
      await post(`/runs/${run.run_id}/review`, { verdict, label_step: verdict === "wrong_step" ? Number(label) : null });
      onDone();
    } catch (e) {
      setError(e.message);
    }
  };
  if (run.review)
    return (
      <p className="small muted">
        Reviewed: <strong>{run.review.verdict.replace("_", " ")}</strong>
        {run.review.label_step ? ` (origin step ${run.review.label_step})` : ""}.{" "}
        {run.review.queued_for_training ? "Queued for the next reviewed training round." : "Not used for training."}
      </p>
    );
  return (
    <div className="review">
      <span className="label" style={{ margin: 0 }}>Is the suspect right?</span>
      <button className="btn small" onClick={() => send("confirmed")}>Yes, step {root.step}</button>
      {wrong ? (
        <>
          <select value={label} onChange={(e) => setLabel(e.target.value)}>
            {run.steps.map((s) => (
              <option key={s.step_id} value={s.step_id}>
                {s.step_id} · {nodeLabel(s.node_name)}
              </option>
            ))}
          </select>
          <button className="btn small" onClick={() => send("wrong_step")}>Save correct step</button>
        </>
      ) : (
        <button className="btn small" onClick={() => setWrong(true)}>No, another step</button>
      )}
      <button className="btn small" onClick={() => send("uncertain")}>Not sure</button>
      {error && <span className="small" style={{ color: "var(--root)" }}>{error}</span>}
    </div>
  );
}

function Inspector({ step }) {
  const [tab, setTab] = useState("output");
  useEffect(() => setTab("output"), [step?.step_id]);
  if (!step) return null;
  const tabs = [["output", "Output"], ["input", "Input"], ...(step.prompt ? [["prompt", "LLM prompt"]] : []), ["state", "State change"]];
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
        {step.effect && <span className="tag">{step.effect.replace("_", " ")}</span>}
        {step.retries > 0 && <span className="tag">{step.retries} retr{step.retries === 1 ? "y" : "ies"}</span>}
        {step.recovery && (
          <span className="tag" style={{ color: "var(--pass)" }}>
            live check: {step.recovery.violations.join("; ")} → {step.recovery.recovered ? "repaired by " + step.recovery.strategy : "escalated"}
          </span>
        )}
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

export default function RunDetail({ run: initial, onBack, onBreak }) {
  const [run, setRun] = useState(initial);
  useEffect(() => setRun(initial), [initial]);
  const refresh = () => api("/runs/" + run.run_id).then(setRun).catch(() => {});
  const d = diagnosisOf(run);
  const failed = run.status === "FAILED";
  const root = failed ? d.root_cause : null;
  const flow = failed ? d.evidence?.data_flow || [] : [];
  const [selected, setSelected] = useState(root?.step || run.steps?.[0]?.step_id);
  const [view, setView] = useState("tape");
  const [altStep, setAltStep] = useState(root?.step);
  useEffect(() => {
    setSelected(root?.step || run.steps?.[0]?.step_id);
    setAltStep(root?.step);
  }, [run.run_id]);
  const step = run.steps?.find((s) => s.step_id === selected);
  const factors = (d.evidence?.factors || []).filter((f) => !f.feature.startsWith("node_") && f.feature !== "position");
  const maxContribution = Math.max(0.0001, ...factors.map((f) => Math.abs(f.contribution || 0)));
  const votes = d.evidence?.model_votes || {};
  const crashed = run.status === "ERROR";
  const outcome = crashed ? "failed" : run.acceptance?.outcome || (failed ? "failed" : "passed");
  const external = run.adapter === "langgraph";
  const crashStep = run.steps?.find((s) => s.step_id === run.crashed_step);
  const role = (s) => (!failed ? "fine" : s.step_id === root?.step ? "root" : flow.includes(s.step_id) ? "affected" : "fine");

  return (
    <>
      <button className="back" onClick={onBack}>
        <ArrowLeft size={16} /> All runs
      </button>
      <div className="page-head" style={{ marginBottom: 16 }}>
        <div>
          <h1 style={{ fontSize: 32 }}>{run.prompt}</h1>
          <p className="small">
            {familyLabels[run.task_family] || run.task_family} · agent model <strong>{run.model || "unknown"}</strong>
            {run.llm_provider === "sandbox" && " (offline sandbox fixture)"} · {(run.total_tokens || 0).toLocaleString()} tokens ·{" "}
            {money(run.cost_usd)} · run <span className="mono">{short(run.run_id)}</span>
            {run.fault_type && (
              <>
                {" "}· planted fault for this demo: <strong>{nodeLabel(run.fault_type)}</strong> (hidden from the models)
              </>
            )}
          </p>
        </div>
        <div className="actions">
          <StatusBadge status={run.status} />
          {failed ? (
            <>
              <button className="btn" onClick={() => download(`/runs/${run.run_id}/report`, `incident-${run.run_id}.md`)}>
                <Download size={16} /> Report
              </button>
              <button className="btn" onClick={() => download(`/runs/${run.run_id}/regression-test`)}>
                <FileCode2 size={16} /> Export test
              </button>
            </>
          ) : crashed || external ? null : (
            <button className="btn primary" onClick={onBreak}>
              <Zap size={16} /> Break this run
            </button>
          )}
        </div>
      </div>

      <Capabilities
        caps={run.capabilities || BUILTIN_CAPS}
        adapter={external ? "External LangGraph agent" : "Built-in agent"}
      />
      {(run.fork || run.resumed_from) && (
        <div className="note section-gap" style={{ marginTop: 0, marginBottom: 16 }}>
          <span>
            {run.fork
              ? `Fork of run ${short(run.parent_run_id)}: step ${run.fork.from_step} (${nodeLabel(run.fork.node)}) was patched at its native LangGraph checkpoint; LangGraph re-ran the steps that depend on it.`
              : `Resumed from run ${short(run.parent_run_id)} at its last good checkpoint; only the failed and remaining steps ran.`}
          </span>
        </div>
      )}
      {d.abstain && (
        <div className="status-banner">
          <AlertTriangle size={17} /> Not sure. {d.abstain_reason} Candidates:{" "}
          {(d.top_suspects || []).map((s) => `step ${s.step} (${nodeLabel(s.node)})`).join(", ")}.
        </div>
      )}
      {run.live_recovery?.interruptions > 0 && (
        <div className="note good" style={{ marginBottom: 16 }}>
          <span>
            Live recovery caught {run.live_recovery.interruptions} problem{run.live_recovery.interruptions === 1 ? "" : "s"} while the run
            was executing and repaired {run.live_recovery.recovered} before dependent steps ran
            {run.live_recovery.escalated ? `; ${run.live_recovery.escalated} escalated` : ""}. Marked on the tape.
          </span>
        </div>
      )}
      {d.semantic_fallback && (
        <div className="status-banner">
          <AlertTriangle size={17} /> The text encoders could not load, so hashed embeddings stand in for MiniLM; this ranking is less reliable.
        </div>
      )}
      {d.method && d.method !== "ensemble" && (
        <div className="status-banner">
          <AlertTriangle size={17} /> Trained models are not loaded, so this ranking comes from the rule-based fallback.
        </div>
      )}

      <div className={"verdict " + (outcome === "passed" ? "passed" : "failed")}>
        <div>
          <span className="label">{OUTCOME_TEXT[outcome]}</span>
          {crashed ? (
            <>
              <h2>
                Crashed at <em>{crashStep ? `step ${crashStep.step_id} · ${nodeLabel(crashStep.node_name)}` : "an unknown step"}</em>
              </h2>
              <p>
                {run.error}. The last good checkpoint was kept
                {run.last_checkpoint?.checkpoint_id ? ` (${run.last_checkpoint.checkpoint_id.slice(0, 13)}…)` : ""}
                {external ? ": once the cause is fixed, continue it from your code with agent.resume(run_id)." : "."}
              </p>
            </>
          ) : failed && root ? (
            <>
              <h2>
                Leading suspect: <em>step {root.step} · {nodeLabel(root.node)}</em>
              </h2>
              <p>{(d.evidence?.summary || "").replace(/^[\w ]+: /, "")}</p>
            </>
          ) : outcome === "unknown" ? (
            <h2>Recorded. No acceptance check was supplied, so the outcome is unknown.</h2>
          ) : (
            <h2>The agent's answer passed.</h2>
          )}
          <CheckList acceptance={run.acceptance} compact />
        </div>
        {failed && root && (
          <div className="gauges">
            <div className="gauge">
              <div className="stat-value tone-root">{percent(root.confidence, 0)}</div>
              <div className="stat-note">ranking score (not a probability)</div>
            </div>
            <div className="gauge">
              <div className="stat-value">{d.top_suspects?.length > 1 ? "Step " + d.top_suspects[1].step : "—"}</div>
              <div className="stat-note">next suspect</div>
            </div>
          </div>
        )}
      </div>

      <div className="view-toggle">
        <Segmented value={view} onChange={setView} options={[["tape", "Flight tape"], ["graph", "Dependencies"]]} />
      </div>
      {view === "tape" ? (
        <Tape
          title="Flight tape"
          steps={run.steps || []}
          selected={selected}
          onSelect={(s) => setSelected(s.step_id)}
          role={(s) => (s.recovery?.recovered ? "patched" : crashed && s.tool_error ? "root" : run.fork || run.resumed_from ? ({ checkpoint: "checkpoint", reused: "reused", patched: "patched", rerun: "rerun" }[s.action] || role(s)) : role(s))}
          flag={(s) => (s.recovery?.recovered ? "Recovered" : s.recovery?.escalated ? "Escalated" : crashed && s.tool_error ? "Error" : failed && s.step_id === root?.step ? "Suspect" : run.fork && s.action === "patched" ? "Patched" : null)}
          legend={failed ? LEGEND : [["#2f9e64", "Step completed"]]}
          footer={
            failed && flow.length > 1 ? (
              <>
                Steps <strong>{flow.slice(1).join(", ")}</strong> depend on step <strong>{root.step}</strong> and must re-run in any fix; the
                green steps are independent and can be reused.
              </>
            ) : (
              "Click a step to inspect its input, output and prompt."
            )
          }
        />
      ) : (
        <Panel title="Dependencies" aside="red: suspect · amber: depends on it · green: independent, reusable">
          <DepGraph steps={run.steps || []} role={role} selected={selected} onSelect={(s) => setSelected(s.step_id)} />
        </Panel>
      )}

      {failed && root ? (
        <>
          <div className="grid-main section-gap">
            <Panel title={`Why step ${root.step} is suspected`} aside={d.method === "ensemble" ? "recorded evidence, SHAP-ranked" : "rule-based evidence"}>
              <ul className="reasons">
                {factors.slice(0, 4).map((f) => (
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
              <div className="section-gap">
                <Explain runId={run.run_id} />
              </div>
              <div className="section-gap">
                <Review run={run} root={root} onDone={refresh} />
              </div>
              <details className="disclosure section-gap">
                <summary>Technical details: model votes and other suspects</summary>
                <div className="votes section-gap" style={{ marginTop: 12 }}>
                  {Object.entries(votes).map(([k, v]) => (
                    <div key={k} className={"vote " + (v.step === root.step ? "agree" : "")}>
                      <div className="who">{MODEL_NAMES[k] || v.model}</div>
                      <div className="what">Step {v.step}</div>
                      <div className="how">score {percent(v.blame, 0)} · weight {v.weight}</div>
                    </div>
                  ))}
                </div>
                <dl className="kv section-gap">
                  {(d.top_suspects || []).map((s, i) => (
                    <div key={s.step} style={{ display: "contents" }}>
                      <dt>Suspect {i + 1}</dt>
                      <dd>
                        Step {s.step} · {nodeLabel(s.node)} ({percent(s.blame, 0)})
                      </dd>
                    </div>
                  ))}
                  <dt>Closest successful run</dt>
                  <dd className="mono">{d.evidence?.nearest_success ? short(d.evidence.nearest_success) : "none"}</dd>
                  <dt>Model version</dt>
                  <dd className="mono">{d.model_version || d.method}</dd>
                </dl>
              </details>
            </Panel>
            <Inspector step={step} />
          </div>
          <div className="section-gap">
            {d.top_suspects?.length > 1 && (
              <div className="view-toggle" style={{ justifyContent: "flex-start" }}>
                <Segmented
                  value={altStep}
                  onChange={setAltStep}
                  options={d.top_suspects.slice(0, 3).map((s) => [s.step, `Fix step ${s.step} · ${nodeLabel(s.node)}`])}
                />
              </div>
            )}
            {altStep && <Alternatives run={run} step={altStep} onDone={refresh} />}
          </div>
        </>
      ) : (
        <div className="section-gap">
          <Inspector step={step} />
        </div>
      )}
    </>
  );
}
