import { useEffect, useMemo, useState } from "react";
import { LoaderCircle, Zap } from "lucide-react";
import { api, nodeLabel, post } from "../api";
import { Empty, Json, PageHead, Panel, RunPicker, Tape } from "../ui";

const WHAT = {
  wrong_tool_value: "The tool returns a plausible but wrong number.",
  empty_result: "The tool returns nothing.",
  tool_timeout: "The tool call times out.",
  stale_data: "The FX tool returns a year-old exchange rate.",
  retrieval_poisoning: "An outdated document is ranked first.",
  wrong_tool_choice: "The router picks the wrong tool.",
  wrong_arguments: "The router passes a wrong value (wrong units, region or item count).",
  dropped_constraint: "The planner forgets one of the task's constraints.",
  hallucinated_fact: "The reasoner states an invented number.",
  premature_final: "The router jumps straight to the final answer.",
  memory_overwrite: "Memory silently overwrites a stored value.",
  loop_repetition: "The router keeps repeating the same call.",
};

export default function BreakRun({ runs, selected, onDone }) {
  const passed = runs.filter((r) => r.status === "PASSED");
  const [id, setId] = useState(selected?.status === "PASSED" ? selected.run_id : passed[0]?.run_id || "");
  const [run, setRun] = useState(null);
  const [step, setStep] = useState(null);
  const [catalog, setCatalog] = useState([]);
  const [fault, setFault] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api("/faults").then((d) => setCatalog(d.faults || [])).catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    if (!id) return;
    let live = true;
    api("/runs/" + id)
      .then((r) => {
        if (!live) return;
        setRun(r);
        setStep(r.steps.find((s) => s.node_name === "currency_rate")?.step_id || r.steps[2]?.step_id || 1);
      })
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [id]);
  const chosen = run?.steps.find((s) => s.step_id === step);
  const faults = useMemo(() => catalog.filter((f) => chosen && f.node_types.includes(chosen.node_type)), [catalog, chosen]);
  useEffect(() => {
    if (faults.length && !faults.some((f) => f.id === fault)) setFault(faults[0].id);
  }, [faults]);

  const inject = async () => {
    setBusy(true);
    setError("");
    try {
      const r = await post("/runs/" + id + "/inject", { step_id: step, fault_type: fault });
      onDone(r.new_run_id || r.run_id);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHead title="Break a run">
        Pick a run that worked, choose a step and a fault. Black Box copies the run, breaks that step, replays the rest, and
        then has to find the step you broke. The original run is never changed.
      </PageHead>
      {!passed.length ? (
        <Panel>
          <Empty title="No successful runs to break">Start a run first.</Empty>
        </Panel>
      ) : (
        <>
          <div style={{ maxWidth: 720 }}>
            <RunPicker runs={runs} value={id} onChange={setId} label="1 · Run to break" filter={(r) => r.status === "PASSED"} />
          </div>
          {run && (
            <Tape
              title="2 · Click the step to break"
              steps={run.steps}
              selected={step}
              onSelect={(s) => setStep(s.step_id)}
              role={(s) => (s.step_id === step ? "patched" : "fine")}
              flag={(s) => (s.step_id === step ? "Target" : null)}
            />
          )}
          {chosen && (
            <div className="grid-main section-gap">
              <Panel title="3 · Choose the fault" aside={`for a ${chosen.node_type} step`}>
                <div className="option-list">
                  {faults.map((f) => (
                    <button key={f.id} className={"option" + (fault === f.id ? " active" : "")} onClick={() => setFault(f.id)}>
                      <strong>
                        {f.label} {f.held_out && <span className="tag">never seen in training</span>}
                      </strong>
                      <span>{WHAT[f.id]}</span>
                    </button>
                  ))}
                  {!faults.length && <p className="muted">No fault applies to this step. Pick another one.</p>}
                </div>
                {error && (
                  <div className="note bad section-gap" role="alert">
                    {error}
                  </div>
                )}
                <button className="btn danger block section-gap" onClick={inject} disabled={busy || !faults.length}>
                  {busy ? <LoaderCircle size={17} className="spin" /> : <Zap size={17} />}
                  {busy ? "Breaking and replaying…" : `Break step ${chosen.step_id} and investigate`}
                </button>
              </Panel>
              <Panel title={`Step ${chosen.step_id} · ${nodeLabel(chosen.node_name)} today`} aside="current output">
                <Json value={chosen.output} />
              </Panel>
            </div>
          )}
        </>
      )}
    </>
  );
}
