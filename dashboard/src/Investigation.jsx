import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Box,
  Check,
  CheckCircle2,
  CircleDot,
  Code2,
  Database,
  FlaskConical,
  GitBranch,
  GitCompareArrows,
  Layers3,
  LoaderCircle,
  Play,
  Radio,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
  Zap,
} from "lucide-react";
import { ReactFlow, Background, Controls, MarkerType } from "@xyflow/react";
import {
  BarChart,
  Bar,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  api,
  post,
  pretty,
  short,
  percent,
  diagnosisOf,
  familyLabels,
} from "./api";
import {
  Badge,
  Empty,
  Json,
  Metric,
  PageHeader,
  RunSelect,
} from "./components";
const examples = {
  finance: "Convert ₹50,000 to USD and compute EMI for 12 months at 9%.",
  sql: "What is the total revenue for north?",
  doc_qa: "How many days do I have to return a purchase?",
  math: "Buy 8 items at $12 each with a 10% discount.",
};

function TraceGraph({ run, onSelect }) {
  const d = diagnosisOf(run),
    root = run.status === "FAILED" ? d.root_cause?.step : null,
    flow = run.status === "FAILED" ? d.evidence?.data_flow || [] : [],
    steps = run.steps || [];
  const nodes = steps.map((s, i) => ({
    id: String(s.step_id),
    position: { x: i * 192, y: i % 2 ? 66 : 20 },
    data: {
      label: (
        <div className="graph-node">
          <span>
            STEP {String(s.step_id).padStart(2, "0")}
            <span>{s.action === "reused" ? "CACHED" : s.node_type}</span>
          </span>
          <strong>{s.node_name}</strong>
          <small>
            {s.step_id === root
              ? "Top suspect · " + percent(d.root_cause.confidence)
              : Number(s.latency_ms || 0).toFixed(1) + " ms"}
          </small>
        </div>
      ),
    },
    style: {
      background:
        s.step_id === root
          ? "#fff2f0"
          : flow.includes(s.step_id)
            ? "#fff7eb"
            : "#fff",
      border:
        "1px solid " +
        (s.step_id === root
          ? "#ec8b7d"
          : flow.includes(s.step_id)
            ? "#efc68f"
            : "#d9e4df"),
      borderRadius: 8,
      width: 170,
      padding: 0,
      color: "#28302d",
    },
  }));
  const edges = steps.flatMap((s) =>
    (s.parent_step_ids || []).map((p) => ({
      id: p + "-" + s.step_id,
      source: String(p),
      target: String(s.step_id),
      type: "smoothstep",
      markerEnd: { type: MarkerType.ArrowClosed, color: "#acb5bd" },
      style: { stroke: "#bac3ca" },
      animated: s.action === "rerun",
    })),
  );
  return (
    <div className="trace-graph">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        minZoom={0.2}
        maxZoom={1.4}
        nodesDraggable={false}
        nodesConnectable={false}
        onNodeClick={(_, n) =>
          onSelect?.(steps.find((s) => String(s.step_id) === n.id))
        }
      >
        <Background gap={18} color="#dce1e5" />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
export function Diagnosis({ run, onBack, onReplay, onBreak }) {
  const d = diagnosisOf(run),
    [step, setStep] = useState(
      run.steps?.find((s) => s.step_id === d.root_cause?.step) ||
        run.steps?.[0],
    );
  useEffect(
    () =>
      setStep(
        run.steps?.find(
          (s) => s.step_id === diagnosisOf(run).root_cause?.step,
        ) || run.steps?.[0],
      ),
    [run],
  );
  return (
    <>
      <button className="back-link" onClick={onBack}>
        <ArrowLeft size={14} />
        All runs
      </button>
      <PageHeader
        eyebrow={"INVESTIGATION / " + short(run.run_id)}
        title={
          run.status === "FAILED"
            ? "Find the signal in the failure."
            : "Every step, accounted for."
        }
        description={run.prompt}
      >
        <Badge status={run.status} />
        <button
          className="button primary"
          onClick={run.status === "FAILED" ? onReplay : onBreak}
        >
          {run.status === "FAILED" ? (
            <GitBranch size={15} />
          ) : (
            <Zap size={15} />
          )}{" "}
          {run.status === "FAILED" ? "Replay with a fix" : "Break this run"}
        </button>
      </PageHeader>
      <div
        className={
          "diagnosis-banner " + (run.status === "FAILED" ? "bad" : "good")
        }
      >
        <span className="diagnosis-symbol">
          {run.status === "FAILED" ? <TriangleAlert /> : <CheckCircle2 />}
        </span>
        <div>
          <strong>
            {run.status === "PASSED"
              ? "The verifier passed this execution."
              : d.evidence?.summary ||
                "Inspect the trace to investigate this result."}
          </strong>
          <p>
            {d.method || "Evidence-based baseline"} · Confidence is a ranking
            signal; replay tests the hypothesis.
          </p>
        </div>
        {run.status === "FAILED" && d.root_cause && (
          <div className="confidence">
            <strong>{percent(d.root_cause.confidence)}</strong>
            <span>top-step score</span>
          </div>
        )}
      </div>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Execution trace</h2>
            <p>
              {run.steps?.length} steps · select a node to inspect inputs and
              output
            </p>
          </div>
          <div className="legend">
            <span>
              <i className="red-bg" />
              Suspect
            </span>
            <span>
              <i className="orange-bg" />
              Downstream
            </span>
            <span>
              <i className="green-bg" />
              Other
            </span>
          </div>
        </div>
        <TraceGraph run={run} onSelect={setStep} />
      </section>
      <div className="detail-grid">
        <section className="panel">
          <div className="panel-heading">
            <h2>Evidence & attribution</h2>
            <span className="subtle-chip">BASELINE</span>
          </div>
          <div className="panel-body">
            <div className="evidence-chain">
              {(d.evidence?.data_flow || []).map((n, i) => (
                <span key={n}>
                  {i > 0 && <ArrowRight size={13} />}
                  <code>Step {n}</code>
                </span>
              ))}
            </div>
            {(d.evidence?.factors || []).map((f, i) => (
              <div className="factor" key={i}>
                <span className="factor-icon">
                  <Activity size={15} />
                </span>
                <div>
                  <strong>{f.feature?.replaceAll("_", " ")}</strong>
                  <p>{f.description || pretty(f.value)}</p>
                </div>
                <span className="mono">
                  {typeof f.contribution === "number"
                    ? f.contribution.toFixed(2)
                    : ""}
                </span>
              </div>
            ))}
            <div className="muted small">
              Nearest successful run:{" "}
              <code>{short(d.evidence?.nearest_success)}</code>
              {d.evidence?.diverges_at != null &&
                " · first divergence at step " + d.evidence.diverges_at}
            </div>
            <div className="outcome-block">
              <span>Final answer</span>
              <Json value={run.final_answer} />
            </div>
          </div>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <h2>
              {step
                ? "Step " + step.step_id + " · " + step.node_name
                : "Step inspector"}
            </h2>
            <Code2 size={16} />
          </div>
          {step && (
            <div className="panel-body">
              <div className="io-label">INPUT</div>
              <Json value={step.input} />
              <div className="io-label">OUTPUT</div>
              <Json value={step.output} />
              <details>
                <summary>Checkpoint & state changes</summary>
                <p className="mono small">{step.checkpoint_id}</p>
                <Json value={step.state_diff} />
              </details>
            </div>
          )}
        </section>
      </div>
    </>
  );
}
export function LiveRun({ onDone, openRun }) {
  const [family, setFamily] = useState("finance"),
    [prompt, setPrompt] = useState(examples.finance),
    [steps, setSteps] = useState([]),
    [busy, setBusy] = useState(false),
    [run, setRun] = useState(null),
    [error, setError] = useState(""),
    wsRef = useRef(null);
  useEffect(() => () => wsRef.current?.close(), []);
  const start = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    setRun(null);
    setSteps([]);
    try {
      const r = await post("/runs", { prompt, task_family: family });
      const socket = new WebSocket(
        (location.protocol === "https:" ? "wss://" : "ws://") +
          location.host +
          "/api/runs/" +
          r.run_id +
          "/stream",
      );
      wsRef.current = socket;
      socket.onmessage = (event) => {
        const ev = JSON.parse(event.data);
        if (ev.type === "step")
          setSteps((current) =>
            current.some((s) => s.step_id === ev.step_id)
              ? current
              : [...current, ev.step || ev],
          );
        if (ev.type === "complete") {
          setRun(ev.run || { run_id: r.run_id, status: ev.status });
          setBusy(false);
          onDone();
          socket.close();
        }
        if (ev.type === "error") {
          setError(ev.message || ev.error || "Execution failed");
          setBusy(false);
          socket.close();
        }
      };
      socket.onerror = () => {
        setError(
          "Live stream disconnected. Recording continues in the background; open Runs to inspect it.",
        );
        setBusy(false);
      };
      socket.onclose = () => setBusy(false);
    } catch (e) {
      setError(e.message);
      setBusy(false);
    }
  };
  return (
    <>
      <PageHeader
        eyebrow="RECORDER / NEW EXECUTION"
        title="Watch an agent work in steps."
        description="Run a sandbox task and capture its tools, state, and checkpoints."
      />
      <div className="live-layout">
        <form className="panel" onSubmit={start}>
          <div className="panel-heading">
            <h2>New run</h2>
            <span className="subtle-chip">OFFLINE SANDBOX</span>
          </div>
          <div className="panel-body">
            <label className="field-label">
              Task family
              <select
                disabled={busy}
                value={family}
                onChange={(e) => {
                  setFamily(e.target.value);
                  setPrompt(examples[e.target.value]);
                }}
              >
                {Object.entries(familyLabels).map(([k, v]) => (
                  <option value={k} key={k}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
            <label className="field-label">
              Task prompt
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={5}
                required
                disabled={busy}
              />
            </label>
            <div className="info-note">
              <ShieldCheck size={16} />
              <span>
                Deterministic tools, isolated data, and a checkpoint at every
                step. No API key required.
              </span>
            </div>
            {error && (
              <div className="inline-error" role="alert">
                {error}
              </div>
            )}
            <button
              className="button primary full-width"
              disabled={busy || !prompt.trim()}
            >
              {busy ? (
                <LoaderCircle size={16} className="spin" />
              ) : (
                <Play size={16} />
              )}{" "}
              {busy ? "Recording execution…" : "Start recording"}
            </button>
          </div>
        </form>
        <section className="panel">
          <div className="panel-heading">
            <div className="inline-title">
              <h2>Live step stream</h2>
              <span className="count">{steps.length}</span>
            </div>
            {busy && (
              <span className="recording">
                <i />
                RECORDING
              </span>
            )}
            {run && <Badge status={run.status} />}
          </div>
          <div className="stream">
            {steps.length ? (
              steps.map((s) => (
                <div className="stream-step" key={s.step_id}>
                  <span className="step-number">
                    {String(s.step_id).padStart(2, "0")}
                  </span>
                  <div>
                    <div className="stream-title">
                      <strong>{s.node_name || s.node}</strong>
                      <span>{Number(s.latency_ms || 0).toFixed(1)} ms</span>
                    </div>
                    <Json value={s.output} />
                  </div>
                  <CheckCircle2 size={16} className="green-text" />
                </div>
              ))
            ) : (
              <Empty icon={Radio} title="Listening for the first step">
                Start a run to see the execution unfold.
              </Empty>
            )}
          </div>
          {run && (
            <div className="panel-body">
              <button
                className="button primary"
                onClick={() => openRun(run.run_id)}
              >
                Open recorded trace
                <ArrowRight size={15} />
              </button>
            </div>
          )}
        </section>
      </div>
    </>
  );
}
export function Replay({ runs, selected, setSelected, onCompare, onUpdate }) {
  const [id, setId] = useState(selected?.run_id || ""),
    [step, setStep] = useState(diagnosisOf(selected).root_cause?.step || 3),
    [patch, setPatch] = useState("{}"),
    [options, setOptions] = useState([]),
    [k, setK] = useState(3),
    [result, setResult] = useState(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  useEffect(() => {
    if (!id) return;
    let active = true;
    api("/runs/" + id)
      .then((r) => {
        if (active) {
          setSelected(r);
          setStep(diagnosisOf(r).root_cause?.step || r.steps[0]?.step_id || 1);
          setResult(null);
        }
      })
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, [id, setSelected]);
  useEffect(() => {
    if (!id) return;
    let active = true;
    setOptions([]);
    api("/runs/" + id + "/suggest-fix?step=" + step)
      .then((r) => {
        if (active) {
          setOptions(r.options || []);
          setPatch(pretty(r.options?.[0]?.patch?.output ?? {}));
          setError("");
        }
      })
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, [id, step]);
  const execute = async () => {
    setBusy(true);
    setError("");
    try {
      const output = JSON.parse(patch);
      setResult(
        await post("/runs/" + id + "/replay", {
          from_step: Number(step),
          patch: { output },
          k: Number(k),
        }),
      );
      onUpdate();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  const current = selected?.run_id === id ? selected : null,
    original = current?.steps?.find((s) => s.step_id === Number(step));
  return (
    <>
      <PageHeader
        eyebrow="COUNTERFACTUAL / REPLAY"
        title="A fix is a hypothesis. Test it."
        description="Fork a checkpoint, change one output, and replay only what depends on it."
      />
      <div className="detail-grid">
        <section className="panel">
          <div className="panel-heading">
            <h2>Configure a replay</h2>
            <GitBranch size={17} />
          </div>
          <div className="panel-body">
            <RunSelect runs={runs} value={id} onChange={setId} />
            <div className="form-row">
              <label className="field-label">
                Fork at step
                <select
                  value={step}
                  onChange={(e) => setStep(Number(e.target.value))}
                >
                  {current?.steps?.map((s) => (
                    <option key={s.step_id} value={s.step_id}>
                      {s.step_id} · {s.node_name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field-label">
                Replay variants
                <select
                  value={k}
                  onChange={(e) => setK(Number(e.target.value))}
                >
                  {[1, 3, 5, 10].map((n) => (
                    <option key={n}>{n}</option>
                  ))}
                </select>
              </label>
            </div>
            <div className="io-label">SUGGESTED FIXES</div>
            <div className="fix-options">
              {options.map((o, i) => (
                <button
                  key={o.id || i}
                  onClick={() => setPatch(pretty(o.patch.output))}
                >
                  <Sparkles size={13} />
                  {o.label || o.id}
                </button>
              ))}
            </div>
            <label className="field-label">
              Patched output · JSON
              <textarea
                className="code-input"
                rows={9}
                value={patch}
                onChange={(e) => setPatch(e.target.value)}
                spellCheck={false}
              />
            </label>
            {error && (
              <div role="alert" className="inline-error">
                {error}
              </div>
            )}
            <button
              disabled={!id || busy}
              className="button primary full-width"
              onClick={execute}
            >
              {busy ? (
                <LoaderCircle className="spin" size={15} />
              ) : (
                <Play size={15} />
              )}{" "}
              {busy ? "Replaying…" : "Run partial replay"}
            </button>
          </div>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <h2>{result ? "Replay results" : "Original output"}</h2>
            {result && (
              <Badge
                status={result.passed === result.k ? "PASSED" : "FAILED"}
              />
            )}
          </div>
          <div className="panel-body">
            {result ? (
              <>
                <div className="replay-score">
                  <strong>
                    {result.passed}
                    <span> / {result.k}</span>
                  </strong>
                  <p>variants passed verification</p>
                </div>
                <div className="replay-metrics">
                  <div>
                    <strong>
                      {Number(result.avoided_steps_pct || 0).toFixed(0)}%
                    </strong>
                    <span>steps avoided</span>
                  </div>
                  <div>
                    <strong>
                      {result.tokens_saved_pct == null
                        ? "N/A"
                        : result.tokens_saved_pct.toFixed(1) + "%"}
                    </strong>
                    <span>tokens saved</span>
                  </div>
                </div>
                <div className="info-note">
                  <CircleDot size={17} />
                  <span>
                    {result.verdict}
                    <br />
                    95% Wilson interval:{" "}
                    {percent(result.ci95?.[0] ?? result.ci95?.low)} –{" "}
                    {percent(result.ci95?.[1] ?? result.ci95?.high)}
                    {result.deterministic && (
                      <>
                        <br />
                        Deterministic repeats share one execution behavior; this
                        interval does not establish stochastic reliability.
                      </>
                    )}
                  </span>
                </div>
                <div className="step-actions">
                  {[
                    ["checkpoint_steps", "Checkpoint"],
                    ["reused_steps", "Reused"],
                    ["rerun_steps", "Re-run"],
                  ].map(([key, label]) => (
                    <div key={key}>
                      <span>{label}</span>
                      <code>
                        {Array.isArray(result[key])
                          ? result[key].join(", ") || "—"
                          : result[key]}
                      </code>
                    </div>
                  ))}
                </div>
                <button
                  className="button secondary full-width"
                  onClick={() =>
                    onCompare(
                      id,
                      result.new_run_id ||
                        result.run_id ||
                        result.replay_run_ids?.[0],
                    )
                  }
                >
                  <GitCompareArrows size={15} />
                  Compare original and replay
                </button>
              </>
            ) : original ? (
              <>
                <p className="muted small">
                  Step {step} · {original.node_name}
                </p>
                <Json value={original.output} />
                <div className="info-note">
                  <Database size={16} />
                  <span>
                    The original run stays intact. Replayed traces are saved as
                    independent forks.
                  </span>
                </div>
              </>
            ) : (
              <Empty icon={GitBranch} title="Pick your fork point">
                Select a recorded run to prepare a patch.
              </Empty>
            )}
          </div>
        </section>
      </div>
    </>
  );
}
export function Compare({ runs, initialIds }) {
  const [a, setA] = useState(initialIds?.[0] || ""),
    [b, setB] = useState(initialIds?.[1] || ""),
    [data, setData] = useState(null),
    [error, setError] = useState("");
  useEffect(() => {
    if (!a || !b) {
      setData(null);
      return;
    }
    let active = true;
    setData(null);
    Promise.all([
      api(
        "/compare?a=" + encodeURIComponent(a) + "&b=" + encodeURIComponent(b),
      ),
      api("/runs/" + a),
      api("/runs/" + b),
    ])
      .then(([diff, left, right]) => {
        if (active) {
          setData({ ...diff, left, right });
          setError("");
        }
      })
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, [a, b]);
  return (
    <>
      <PageHeader
        eyebrow="TRACE ANALYSIS / COMPARE"
        title="One change. A different outcome."
        description="Align two executions and inspect the first point where they diverge."
      />
      <section className="panel">
        <div className="panel-body compare-selectors">
          <RunSelect
            runs={runs}
            value={a}
            onChange={setA}
            label="Original run"
          />
          <GitCompareArrows size={22} />
          <RunSelect
            runs={runs}
            value={b}
            onChange={setB}
            label="Replayed run"
          />
        </div>
      </section>
      {error && <div className="inline-error">{error}</div>}
      {data ? (
        <>
          <div
            className={
              "diagnosis-banner " + (data.outcome.after ? "good" : "bad")
            }
          >
            <GitCompareArrows />
            <div>
              <strong>
                {data.first_divergence != null
                  ? "First divergence: step " + data.first_divergence
                  : "No output divergence found"}
              </strong>
              <p>
                {data.outcome.before ? "Passed" : "Failed"} →{" "}
                {data.outcome.after ? "Passed" : "Failed"} ·{" "}
                {data.changes.length} steps with state or output changes
              </p>
            </div>
          </div>
          <section className="panel">
            <div className="panel-heading">
              <h2>Aligned trace</h2>
              <span className="subtle-chip">
                {data.aligned_steps.length} STEPS
              </span>
            </div>
            <div className="diff-table">
              {data.aligned_steps.map((row, i) => {
                const left = data.left.steps.find(
                    (s) => s.step_id === row.a_step,
                  ),
                  right = data.right.steps.find(
                    (s) => s.step_id === row.b_step,
                  );
                return (
                  <div
                    className={
                      "diff-row " +
                      (row.status !== "unchanged" ? "changed" : "")
                    }
                    key={i}
                  >
                    <div>
                      <span className="io-label">
                        ORIGINAL · {row.a_step ?? "—"} / {row.node}
                      </span>
                      <Json value={left ? left.output : "Step absent"} />
                    </div>
                    <ArrowRight size={16} />
                    <div>
                      <span className="io-label">
                        REPLAY · {row.b_step ?? "—"} / {row.status}
                      </span>
                      <Json value={right ? right.output : "Step absent"} />
                    </div>
                  </div>
                );
              })}
            </div>
            <details className="panel-body">
              <summary>State and output differences</summary>
              <Json value={data.changes} />
            </details>
          </section>
        </>
      ) : (
        <Empty icon={GitCompareArrows} title="Put two traces side by side">
          Choose an original run and a replay to inspect the change.
        </Empty>
      )}
    </>
  );
}
export function BreakRun({ runs, selected, onDone }) {
  const [id, setId] = useState(
      selected?.status === "PASSED" ? selected.run_id : "",
    ),
    [run, setRun] = useState(null),
    [step, setStep] = useState(3),
    [catalog, setCatalog] = useState([]),
    [fault, setFault] = useState("stale_data"),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  useEffect(() => {
    api("/faults")
      .then((d) => setCatalog(d.faults || d))
      .catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    if (!id) return;
    let active = true;
    api("/runs/" + id)
      .then((r) => {
        if (active) {
          setRun(r);
          setStep(
            r.steps.find((s) => s.node_name === "currency_rate")?.step_id ||
              r.steps[0]?.step_id,
          );
        }
      })
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, [id]);
  const chosen = run?.steps.find((s) => s.step_id === Number(step)),
    faults = catalog.filter(
      (f) => !f.node_types || f.node_types.includes(chosen?.node_type),
    );
  useEffect(() => {
    if (faults.length && !faults.some((f) => f.id === fault))
      setFault(faults[0].id);
  }, [step, catalog, run]);
  const inject = async () => {
    setBusy(true);
    setError("");
    try {
      const r = await post("/runs/" + id + "/inject", {
        step_id: Number(step),
        fault_type: fault,
      });
      onDone(r.new_run_id || r.run_id);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <PageHeader
        eyebrow="SANDBOX / JUDGE CHALLENGE"
        title="Go ahead. Break it."
        description="Choose the fault. Let Black Box follow the evidence."
      />
      <div className="challenge-layout">
        <section className="panel">
          <div className="panel-heading">
            <h2>Inject a controlled failure</h2>
            <Zap size={17} />
          </div>
          <div className="panel-body">
            <RunSelect
              runs={runs}
              onlyPassed
              value={id}
              onChange={setId}
              label="Successful source run"
            />
            <label className="field-label">
              Target step
              <select
                value={step}
                onChange={(e) => setStep(Number(e.target.value))}
              >
                {run?.steps.map((s) => (
                  <option key={s.step_id} value={s.step_id}>
                    {s.step_id} · {s.node_name}
                  </option>
                ))}
              </select>
            </label>
            <label className="field-label">
              Fault type
              <select value={fault} onChange={(e) => setFault(e.target.value)}>
                {faults.map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.label}
                    {f.held_out ? " · held out" : ""}
                  </option>
                ))}
              </select>
            </label>
            {chosen && (
              <>
                <div className="io-label">CURRENT OUTPUT</div>
                <Json value={chosen.output} />
              </>
            )}
            {error && (
              <div role="alert" className="inline-error">
                {error}
              </div>
            )}
            <button
              className="button danger full-width"
              onClick={inject}
              disabled={!id || busy || !faults.length}
            >
              {busy ? (
                <LoaderCircle className="spin" size={16} />
              ) : (
                <Zap size={16} />
              )}{" "}
              {busy ? "Injecting fault…" : "Break run & investigate"}
            </button>
          </div>
        </section>
        <div className="challenge-explainer">
          <div className="challenge-art">
            <Box size={84} strokeWidth={1} />
            <span className="orbit one" />
            <span className="orbit two" />
            <span className="art-signal">
              <Zap size={20} />
            </span>
          </div>
          <h2>
            A controlled failure.
            <br />A real investigation.
          </h2>
          <p>
            The original stays safe. Black Box forks the run, changes the
            selected output, and checks whether the fault actually changes the
            outcome.
          </p>
          <div className="challenge-points">
            <span>
              <Check size={15} />
              Twelve fault types
            </span>
            <span>
              <Check size={15} />
              Frozen execution time
            </span>
            <span>
              <Check size={15} />
              Fully isolated tools
            </span>
          </div>
        </div>
      </div>
    </>
  );
}
export function Evaluation() {
  const [data, setData] = useState(null),
    [split, setSplit] = useState("test"),
    [error, setError] = useState("");
  useEffect(() => {
    api("/eval")
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);
  const metrics = data?.splits?.[split],
    learned = data?.trained_baseline?.splits?.[split];
  const present = Boolean(data?.splits),
    baselines = metrics?.baselines || {};
  const barData = Object.entries(baselines)
    .filter(([, v]) => v?.top1 != null)
    .map(([name, v]) => ({
      name: name.replaceAll("_", " "),
      top1: v.top1 * 100,
    }));
  if (learned?.top1 != null)
    barData.unshift({ name: "Trained logistic", top1: learned.top1 * 100 });
  if (metrics?.top1 != null)
    barData.unshift({ name: "Live heuristic", top1: metrics.top1 * 100 });
  const faults = [...new Set(metrics?.heatmap?.map((r) => r.fault) || [])];
  return (
    <>
      <PageHeader
        eyebrow="VALIDATION / EVALUATION"
        title="Evidence over promises."
        description="Measured localization performance on held-out prompt templates in the synthetic sandbox."
      >
        <span className="subtle-chip">
          <Database size={13} />
          metrics.json
        </span>
      </PageHeader>
      {error && <div className="inline-error">{error}</div>}
      {!present ? (
        <section className="panel">
          <Empty
            icon={FlaskConical}
            title="The benchmark starts with a measurement."
          >
            No evaluation artifact is available yet. Run the local evaluation
            suite to see measured results here.
          </Empty>
          <div className="eval-command">
            <code>python -m blackbox.evaluation</code>
            <p>The PRD's targets are never presented as achieved results.</p>
          </div>
        </section>
      ) : (
        <>
          <div className="eval-tabs">
            {[
              ["test", "Held-out templates"],
              ["unseen_fault", "Held-out fault types"],
              ["natural_failure", "Natural failures"],
            ].map(([id, l]) => (
              <button
                key={id}
                className={"button " + (id === split ? "primary" : "secondary")}
                onClick={() => setSplit(id)}
              >
                {l}
              </button>
            ))}
          </div>
          {!metrics ? (
            <section className="panel">
              <Empty
                icon={FlaskConical}
                title="Natural failures · not measured"
              >
                This split needs real agent failures and counterfactual labels.
                No synthetic result is substituted for it.
              </Empty>
            </section>
          ) : (
            <>
              <div className="metrics-grid">
                <Metric
                  label="Top-1 localization"
                  value={percent(metrics.top1)}
                  detail="Live heuristic · highest-ranked step"
                  icon={CircleDot}
                />
                <Metric
                  label="Top-3 localization"
                  value={percent(metrics.top3)}
                  detail="Live heuristic · true cause in top 3"
                  icon={Layers3}
                />
                <Metric
                  label="Failure detection AUROC"
                  value={
                    metrics.auroc != null
                      ? Number(metrics.auroc).toFixed(3)
                      : "—"
                  }
                  detail={metrics.total_runs + " passed and failed runs"}
                  icon={Activity}
                />
                <Metric
                  label="Mean reciprocal rank"
                  value={
                    metrics.mrr != null ? Number(metrics.mrr).toFixed(3) : "—"
                  }
                  detail={metrics.count + " labeled failed runs"}
                  icon={FlaskConical}
                />
              </div>
              <div className="detail-grid">
                <section className="panel">
                  <div className="panel-heading">
                    <h2>Localization vs. baselines</h2>
                    <span className="muted small">Top-1 · %</span>
                  </div>
                  <div className="eval-chart">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={barData}
                        layout="vertical"
                        margin={{ left: 15, right: 30 }}
                      >
                        <CartesianGrid
                          horizontal={false}
                          strokeDasharray="3 5"
                        />
                        <XAxis
                          type="number"
                          domain={[0, 100]}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis
                          type="category"
                          dataKey="name"
                          width={125}
                          axisLine={false}
                          tickLine={false}
                          tick={{ fontSize: 11 }}
                        />
                        <Tooltip
                          formatter={(v) => Number(v).toFixed(1) + "%"}
                        />
                        <Bar
                          dataKey="top1"
                          fill="#759786"
                          radius={[0, 4, 4, 0]}
                          barSize={20}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>
                <section className="panel">
                  <div className="panel-heading">
                    <h2>Measurement context</h2>
                    <ShieldCheck size={16} />
                  </div>
                  <div className="panel-body">
                    <div className="info-note">
                      <FlaskConical size={17} />
                      <span>
                        {split === "unseen_fault"
                          ? "These fault types were excluded from logistic training. The heuristic already contains rules for them; its scores do not measure learned generalization."
                          : "Templates use different prompts over the same fixed family workflows. These measurements do not establish performance on new workflows or real-world agents."}
                      </span>
                    </div>
                    <p className="muted small">
                      Live diagnoses use the heuristic. The logistic ranker is a
                      separately trained and evaluated baseline. LLM judges,
                      natural failures, and the PRD ensemble are not measured.
                    </p>
                    <details>
                      <summary>Artifact & methodology</summary>
                      <Json value={data} />
                    </details>
                  </div>
                </section>
              </div>
              <section className="panel heatmap-panel">
                <div className="panel-heading">
                  <div>
                    <h2>Localization by fault & task family</h2>
                    <p>Live heuristic · Top-1 · cells include sample count</p>
                  </div>
                  <span className="subtle-chip">MEASURED</span>
                </div>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>FAULT TYPE</th>
                        {Object.values(familyLabels).map((f) => (
                          <th key={f}>{f.toUpperCase()}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {faults.map((f) => (
                        <tr key={f}>
                          <td>{f.replaceAll("_", " ")}</td>
                          {Object.keys(familyLabels).map((family) => {
                            const cell = metrics.heatmap.find(
                              (r) => r.fault === f && r.family === family,
                            );
                            return (
                              <td key={family}>
                                {cell ? (
                                  <span
                                    className="heat-cell"
                                    style={{
                                      background:
                                        "rgba(83,139,107," +
                                        (0.06 + cell.top1 * 0.18) +
                                        ")",
                                    }}
                                  >
                                    {percent(cell.top1)}
                                    <small>n={cell.count}</small>
                                  </span>
                                ) : (
                                  <span className="muted">—</span>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}
        </>
      )}
    </>
  );
}
