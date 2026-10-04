import { useEffect, useState } from "react";
import { GitCompareArrows, LoaderCircle, Play } from "lucide-react";
import { api, diagnosisOf, money, nodeLabel, percent, post, pretty } from "../api";
import { Field, Json, PageHead, Panel, RunPicker, Segmented, Tape } from "../ui";

const ACTION_LEGEND = [
  ["#6b7a94", "Loaded from checkpoint"],
  ["#4f8ef7", "Reused (inputs unchanged)"],
  ["#f0a020", "Patched / re-run"],
];

export default function Replay({ runs, selected, setSelected, initialCandidate, onCompare, onUpdate }) {
  const [id, setId] = useState(selected?.run_id || runs.find((r) => r.status === "FAILED")?.run_id || "");
  const [step, setStep] = useState(diagnosisOf(selected).root_cause?.step || 3);
  const [mode, setMode] = useState("output");
  const [patch, setPatch] = useState("{}");
  const [options, setOptions] = useState([]);
  const [picked, setPicked] = useState(0);
  const [instruction, setInstruction] = useState("");
  const [model, setModel] = useState("");
  const [k, setK] = useState(5);
  const [result, setResult] = useState(null);
  const [replayRun, setReplayRun] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!id) return;
    let live = true;
    api("/runs/" + id)
      .then((r) => {
        if (!live) return;
        setSelected(r);
        setStep(initialCandidate?.runId === id ? initialCandidate.step : diagnosisOf(r).root_cause?.step || r.steps[0]?.step_id || 1);
        setResult(null);
        setReplayRun(null);
      })
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [id, initialCandidate]);
  useEffect(() => {
    if (!id) return;
    let live = true;
    setOptions([]);
    api("/runs/" + id + "/suggest-fix?step=" + step)
      .then((r) => {
        if (!live) return;
        setOptions(r.options || []);
        const chosen = initialCandidate?.runId === id && initialCandidate.step === step ? initialCandidate.patch?.output : undefined;
        setPicked(chosen === undefined ? 0 : (r.options || []).findIndex(o => pretty(o.patch.output) === pretty(chosen)));
        setPatch(pretty(chosen === undefined ? r.options?.[0]?.patch?.output ?? {} : chosen));
        setError("");
      })
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [id, step, initialCandidate]);

  const current = selected?.run_id === id ? selected : null;
  const original = current?.steps?.find((s) => s.step_id === step);
  const root = current?.status === "FAILED" ? diagnosisOf(current).root_cause : null;
  const isLlm = Boolean(original?.llm_call);
  useEffect(() => {
    if (!isLlm) setMode("output");
  }, [isLlm]);

  const run = async () => {
    setBusy(true);
    setError("");
    try {
      const body =
        mode === "output"
          ? { output: JSON.parse(patch) }
          : mode === "prompt"
            ? { prompt: instruction }
            : { model };
      const r = await post("/runs/" + id + "/replay", { from_step: step, patch: body, k });
      setResult(r);
      setReplayRun(await api("/runs/" + r.run_id));
      onUpdate();
    } catch (e) {
      setError(e instanceof SyntaxError ? "The patched output is not valid JSON." : e.message);
    } finally {
      setBusy(false);
    }
  };

  const confirmed = result?.verdict === "confirmed";
  return (
    <>
      <PageHead title="Replay & alternatives">
        A diagnosis is a hypothesis. Fix the suspected step and replay the run from there: earlier steps load from their
        checkpoint, unaffected steps are reused, and only what depends on the fix runs again.
      </PageHead>
      <div style={{ maxWidth: 720 }}>
        <RunPicker runs={runs} value={id} onChange={setId} label="Run to repair" />
      </div>
      {current && (
        <Tape
          title={result ? "What the replay did" : "1 · Click the step to fix"}
          steps={replayRun?.steps || current.steps}
          selected={result ? null : step}
          onSelect={(s) => !result && setStep(s.step_id)}
          role={(s) =>
            result
              ? { checkpoint: "checkpoint", reused: "reused", patched: "patched", rerun: "rerun" }[s.action] || "fine"
              : s.step_id === root?.step
                ? "root"
                : "fine"
          }
          flag={(s) => (result ? (s.step_id === result.from_step ? "Fixed" : null) : s.step_id === root?.step ? "Suspect" : null)}
          legend={result ? ACTION_LEGEND : root ? [["#ff5a3c", "Leading suspect"]] : null}
          footer={
            result ? (
              <>
                Checkpoint: steps <strong>{result.checkpoint_steps.join(", ") || "none"}</strong> · reused:{" "}
                <strong>{result.reused_steps.join(", ") || "none"}</strong> · re-run: <strong>{result.rerun_steps.join(", ")}</strong>.{" "}
                <button className="btn small" style={{ marginLeft: 8 }} onClick={() => { setResult(null); setReplayRun(null); }}>
                  Try another fix
                </button>
              </>
            ) : null
          }
        />
      )}
      {current && !result && (
        <div className="grid-main section-gap">
          <Panel title={`2 · Fix step ${step} · ${nodeLabel(original?.node_name)}`}>
            <Field label="Type of fix">
              <Segmented
                value={mode}
                onChange={setMode}
                options={[
                  ["output", "Replace the output"],
                  ["prompt", "Add to the prompt", !isLlm, "LLM steps only"],
                  ["model", "Use another model", !isLlm, "LLM steps only"],
                ]}
              />
            </Field>
            {mode === "output" && (
              <>
                {options.length > 0 && (
                <Field label="Suggested alternatives · output interventions">
                    <div className="option-list">
                      {options.map((o, i) => (
                        <button
                          key={o.id + i}
                          type="button"
                          className={"option" + (picked === i ? " active" : "")}
                          onClick={() => {
                            setPicked(i);
                            setPatch(pretty(o.patch.output));
                          }}
                        >
                          <strong>{o.label}</strong>
                          <span>{o.source || o.source_run_id}</span>
                        </button>
                      ))}
                    </div>
                  </Field>
                )}
                <Field label="Corrected output (editable JSON)">
                  <textarea className="code" rows={8} value={patch} onChange={(e) => { setPatch(e.target.value); setPicked(-1); }} spellCheck={false} />
                </Field>
              </>
            )}
            {mode === "prompt" && (
              <Field label="Instruction added to this step's prompt">
                <textarea rows={3} value={instruction} onChange={(e) => setInstruction(e.target.value)} placeholder="Convert lakh amounts to plain numbers before planning." />
              </Field>
            )}
            {mode === "model" && (
              <Field label="Model for this step" hint="For example gpt-6.1-sol. Costs more per call.">
                <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="gpt-6.1-sol" />
              </Field>
            )}
          </Panel>
          <Panel title="3 · Replay">
            <Field label="How many replays" hint="Stochastic replays can reveal variability. Offline repeats are deterministic.">
              <Segmented value={k} onChange={setK} options={[[1, "1"], [3, "3"], [5, "5"], [10, "10"]]} />
            </Field>
            <span className="label">Step {step} output today</span>
            <Json value={original?.output} />
            {error && (
              <div className="note bad section-gap" role="alert">
                {error}
              </div>
            )}
            <button className="btn primary block section-gap" onClick={run} disabled={busy}>
              {busy ? <LoaderCircle size={17} className="spin" /> : <Play size={17} />}
              {busy ? "Replaying…" : `Replay from step ${step}`}
            </button>
          </Panel>
        </div>
      )}
      {result && (
        <div className="grid-main section-gap">
          <Panel title="Result">
            <div className="result-big">
              {result.passed}
              <small> / {result.k} replays passed</small>
            </div>
            <div className={"note section-gap " + (confirmed ? "good" : result.passed ? "warn" : "bad")}>
              <span>
                <strong>{confirmed ? "Repair supported by replay." : result.passed ? "Inconclusive." : "This alternative did not work."}</strong>{" "}
                {result.verdict_text}
              </span>
            </div>
            <div className="metric-row">
              <div>
                <strong>{Number(result.avoided_steps_pct).toFixed(0)}%</strong>
                <span>steps not re-run</span>
              </div>
              <div>
                <strong>{result.tokens_saved_pct_without_cache != null ? result.tokens_saved_pct_without_cache.toFixed(0) + "%" : "—"}</strong>
                <span>tokens saved by checkpoints{result.tokens_estimated ? " (est.)" : ""}</span>
              </div>
              <div>
                <strong>{result.tokens_saved_pct != null ? result.tokens_saved_pct.toFixed(0) + "%" : "—"}</strong>
                <span>with the response cache</span>
              </div>
              <div>
                <strong>{money(result.cost_usd)}</strong>
                <span>cost of these replays</span>
              </div>
            </div>
            <p className="small muted">
              {result.deterministic ? "Deterministic execution: repeated variants do not provide independent reliability evidence." : <>Reported 95% interval: {percent(result.ci95?.[0], 0)} – {percent(result.ci95?.[1], 0)}. Interpret it alongside the sampling and cache protocol.</>}
            </p>
          </Panel>
          <Panel title="Before and after">
            <dl className="kv">
              <dt>Original answer</dt>
              <dd className="mono" style={{ color: "var(--root)" }}>{JSON.stringify(current.final_answer)}</dd>
              <dt>Answer after the fix</dt>
              <dd className="mono" style={{ color: result.passed ? "var(--pass)" : "var(--root)" }}>{JSON.stringify(result.variants?.[0]?.final_answer)}</dd>
            </dl>
            <button className="btn block section-gap" onClick={() => onCompare(id, result.run_id)}>
              <GitCompareArrows size={16} /> Compare the two runs side by side
            </button>
          </Panel>
        </div>
      )}
    </>
  );
}
