import { useEffect, useRef, useState } from "react";
import { ArrowRight, LoaderCircle, Play, Radio } from "lucide-react";
import { compactJson, familyLabels, money, nodeLabel, post } from "../api";
import { Empty, Field, PageHead, Panel, StatusBadge, Tape } from "../ui";

const PLANTS = [
  ["", "No planted fault"],
  ["3:stale_data", "Step 3 returns a year-old FX quote (finance)"],
  ["2:wrong_arguments", "Step 2 passes a wrong value"],
  ["1:dropped_constraint", "Step 1 drops a task constraint"],
  ["9:hallucinated_fact", "Step 9 states an invented number"],
];

const EXAMPLES = {
  finance: [
    "Convert ₹50,000 to USD and compute EMI for 12 months at 9%.",
    "If I take Rs. 1.5 lakh at 10.5% for 24 months, how much is each monthly payment in USD?",
  ],
  sql: ["What is the total revenue for the north region in Q3?", "How many units did the west region sell in Q1?"],
  doc_qa: ["How many days do I have to return a laptop?", "What is the deadline, in days, to file a damage claim for a sofa?"],
  math: ["Buy 8 items at $12 each with a 10% discount. What is the total?", "What do 25 mugs cost at $6 apiece after a 15% discount?"],
};

export default function Live({ llm, onDone, openRun }) {
  const [family, setFamily] = useState("finance");
  const [prompt, setPrompt] = useState(EXAMPLES.finance[0]);
  const [provider, setProvider] = useState("default");
  const [liveRecovery, setLiveRecovery] = useState(false);
  const [plant, setPlant] = useState("");
  const [steps, setSteps] = useState([]);
  const [run, setRun] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const socket = useRef(null);
  useEffect(() => () => socket.current?.close(), []);

  const start = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    setRun(null);
    setSteps([]);
    try {
      const [plantStep, plantFault] = plant.split(":");
      const r = await post("/runs", {
        prompt,
        task_family: family,
        provider,
        live_recovery: liveRecovery,
        plant_fault: plant ? { step: Number(plantStep), fault: plantFault } : null,
      });
      const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/api/runs/" + r.run_id + "/stream");
      socket.current = ws;
      ws.onmessage = (event) => {
        const ev = JSON.parse(event.data);
        if (ev.type === "step") setSteps((cur) => (cur.some((s) => s.step_id === ev.step_id) ? cur : [...cur, ev.step]));
        if (ev.type === "complete") {
          setRun(ev.run);
          setBusy(false);
          onDone();
          ws.close();
        }
        if (ev.type === "error") {
          setError(ev.message || "The run stopped with an error.");
          setBusy(false);
          ws.close();
        }
      };
      ws.onerror = () => {
        setError("Lost the live connection. The run keeps recording; find it on the Runs page.");
        setBusy(false);
      };
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  };

  const gpt = llm?.openai_key_configured;
  return (
    <>
      <PageHead title="Live run">
        Give the agent a task and watch it work step by step. Every step is checkpointed and cached as it happens.
      </PageHead>
      <div className="grid-main" style={{ gridTemplateColumns: "minmax(0, 0.8fr) minmax(0, 1.2fr)" }}>
        <form className="panel" onSubmit={start}>
          <div className="panel-head">
            <h2>New task</h2>
          </div>
          <div className="panel-body">
            <Field label="Agent model">
              <select value={provider} onChange={(e) => setProvider(e.target.value)} disabled={busy}>
                <option value="default">Default ({llm?.active_provider === "sandbox" ? "offline sandbox" : llm?.active_model})</option>
                <option value="openai" disabled={!gpt}>
                  OpenAI {llm?.openai_model || "gpt-6-luna"} {gpt ? "· billed, capped by LLM_BUDGET_USD" : "· add a key to .env"}
                </option>
                <option value="sandbox">Offline sandbox · free, no internet</option>
                <option value="ollama">Ollama {llm?.ollama_model} · local</option>
              </select>
            </Field>
            <Field label="Task type">
              <select
                value={family}
                disabled={busy}
                onChange={(e) => {
                  setFamily(e.target.value);
                  setPrompt(EXAMPLES[e.target.value][0]);
                }}
              >
                {Object.entries(familyLabels).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Task">
              <textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} disabled={busy} required />
              <div className="examples">
                {EXAMPLES[family].map((ex) => (
                  <button type="button" key={ex} onClick={() => setPrompt(ex)} disabled={busy}>
                    {ex.length > 52 ? ex.slice(0, 52) + "…" : ex}
                  </button>
                ))}
              </div>
            </Field>
            <Field label="Demo: plant a fault during the run" hint="The fault is hidden from the diagnosis models.">
              <select value={plant} onChange={(e) => setPlant(e.target.value)} disabled={busy}>
                {PLANTS.map(([v, l]) => (
                  <option key={v} value={v} disabled={v.startsWith("3:stale") && family !== "finance"}>
                    {l}
                  </option>
                ))}
              </select>
            </Field>
            <label className="switch">
              <input type="checkbox" checked={liveRecovery} onChange={(e) => setLiveRecovery(e.target.checked)} disabled={busy} />
              Live recovery: check each step as it finishes and repair it before dependent steps run
            </label>
            {error && (
              <div className="note bad" role="alert" style={{ marginBottom: 14 }}>
                {error}
              </div>
            )}
            <button className="btn primary block" disabled={busy || !prompt.trim()}>
              {busy ? <LoaderCircle size={17} className="spin" /> : <Play size={17} />}
              {busy ? "Recording…" : "Run the agent"}
            </button>
          </div>
        </form>

        <div className="stack">
          {steps.length || busy ? (
            <Tape
              title={busy ? "Recording" : "Recorded"}
              steps={steps}
              placeholders={busy ? Math.max(0, 10 - steps.length) : 0}
              role={(s) => (s.recovery?.recovered ? "patched" : "recorded")}
              flag={(s) => (s.recovery?.recovered ? "Recovered" : s.recovery?.escalated ? "Escalated" : null)}
              legend={[["#6b7a94", "Recorded (not yet checked)"], ...(liveRecovery ? [["#f0a020", "Repaired live"]] : [])]}
            />
          ) : null}
          <section className="panel">
            <div className="panel-head">
              <h2>Steps</h2>
              {busy ? <span className="recording">RECORDING</span> : run && <StatusBadge status={run.status} />}
            </div>
            <div className="panel-body" style={{ padding: "12px 0 0" }}>
              {steps.length ? (
                <div className="stream">
                  {steps.map((s) => (
                    <div className="stream-row" key={s.step_id}>
                      <span className="n">{String(s.step_id).padStart(2, "0")}</span>
                      <div>
                        <strong>{nodeLabel(s.node_name)}</strong>
                        <div className="out">{compactJson(s.output, 180)}</div>
                      </div>
                      <span className="t">
                        {s.llm_call ? (s.tokens_in || 0) + (s.tokens_out || 0) + " tok" : Number(s.latency_ms || 0).toFixed(1) + " ms"}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <Empty title="No run yet" icon={Radio}>
                  Steps appear here as the agent works.
                </Empty>
              )}
            </div>
            {run && (
              <div className="panel-body" style={{ borderTop: "1px solid var(--line)" }}>
                <div className={"note " + (run.status === "PASSED" ? "good" : "bad")} style={{ marginBottom: 12 }}>
                  <span>
                    {run.status === "PASSED" ? "Passed its acceptance checks." : "Failed its acceptance checks. Open the investigation to see the leading suspect."}
                  </span>
                </div>
                <p style={{ marginBottom: 12 }}>
                  Answer <strong className="mono">{JSON.stringify(run.final_answer)}</strong>
                  {run.status === "PASSED" ? " matches the expected value." : ` (expected ${JSON.stringify(run.gold_answer)}).`}{" "}
                  <span className="muted">
                    {(run.total_tokens || 0).toLocaleString()} tokens{run.tokens_estimated ? " (estimated)" : ""} · {money(run.cost_usd)}
                  </span>
                </p>
                <button className="btn primary" onClick={() => openRun(run.run_id)}>
                  Open the investigation <ArrowRight size={16} />
                </button>
              </div>
            )}
          </section>
        </div>
      </div>
    </>
  );
}
