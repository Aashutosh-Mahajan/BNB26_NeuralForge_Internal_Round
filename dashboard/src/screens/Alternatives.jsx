import { useEffect, useState } from "react";
import { Check, CircleHelp, FlaskConical, LoaderCircle, X } from "lucide-react";
import { api, money, nodeLabel, post } from "../api";
import { Panel, Segmented } from "../ui";

const KIND = {
  tool_execution: "Tool called again",
  llm_rerun: "LLM step re-run",
  argument_repair: "Arguments rebuilt",
  output_substitution: "Recorded output reused",
  ask_user: "Escalate",
};
const STATUS_CLASS = {
  original_failed: "failed",
  passed: "passed",
  rejected: "failed",
  inconclusive: "running",
  suggested: "queued",
  needs_review: "running",
  unavailable: "error",
  manual: "error",
};
const STATUS_TEXT = {
  original_failed: "Failed",
  passed: "Passed",
  rejected: "Rejected",
  inconclusive: "Inconclusive",
  suggested: "Suggested",
  needs_review: "Check condition",
  unavailable: "Unavailable",
  manual: "Manual",
};

function Pre({ p }) {
  return (
    <li className={"pre " + (p.ok ? "ok" : p.hard === false ? "soft" : "bad")}>
      {p.ok ? <Check size={14} /> : p.hard === false ? <CircleHelp size={14} /> : <X size={14} />}
      {p.ok ? p.name : p.detail}
    </li>
  );
}

export function CheckList({ acceptance, compact }) {
  if (!acceptance?.checks) return null;
  return (
    <ul className={"checks" + (compact ? " compact" : "")}>
      {acceptance.checks.map((c) => (
        <li key={c.id} className={"check " + c.status} title={c.detail}>
          {c.status === "pass" ? <Check size={14} /> : c.status === "fail" ? <X size={14} /> : <CircleHelp size={14} />}
          <span>{c.label}</span>
          {!compact && <small>{c.detail}</small>}
        </li>
      ))}
    </ul>
  );
}

export default function Alternatives({ run, step, onDone }) {
  const [cards, setCards] = useState([]);
  const [picked, setPicked] = useState([]);
  const [k, setK] = useState(3);
  const [mode, setMode] = useState("recorded");
  const [memory, setMemory] = useState(null);
  const [experiment, setExperiment] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let live = true;
    setCards([]);
    setError("");
    Promise.all([
      api(`/runs/${run.run_id}/alternatives?step=${step}`),
      api(`/runs/${run.run_id}/experiments`),
      api(`/runs/${run.run_id}/memory?step=${step}`).catch(() => null),
    ])
      .then(([alt, past, mem]) => {
        if (!live) return;
        setMemory(mem);
        setCards(alt.candidates);
        setPicked(alt.candidates.filter((c) => c.status === "suggested" && c.kind !== "output_substitution").map((c) => c.id));
        setExperiment(past.experiments.find((x) => x.step_id === step) || null);
      })
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [run.run_id, step]);

  const toggle = (id) => setPicked((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]));
  const test = async () => {
    setBusy(true);
    setError("");
    try {
      setExperiment(await post(`/runs/${run.run_id}/experiments`, { step, strategies: picked, k, mode }));
      onDone?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  const resultFor = (id) => experiment?.branches.find((b) => b.id === id);
  const remembered = (id) => memory?.strategies?.find((m) => m.strategy === id);

  return (
    <Panel title={`Alternatives at step ${step} · ${nodeLabel(run.steps[step - 1]?.node_name)}`} aside="each one is tested from the same checkpoint">
      <div className="alt-list">
        {cards.map((c) => {
          const r = resultFor(c.id);
          const status = r?.status || c.status;
          const selectable = c.status === "suggested" || c.status === "needs_review";
          return (
            <div key={c.id} className={"alt " + (picked.includes(c.id) ? "picked" : "")}>
              <label className="alt-head">
                <input type="checkbox" disabled={!selectable || busy} checked={picked.includes(c.id)} onChange={() => toggle(c.id)} />
                <span>
                  <strong>{c.label}</strong>
                  <span className="tag">{KIND[c.kind] || c.kind}</span>
                </span>
                <span className={"badge " + STATUS_CLASS[status]}>{STATUS_TEXT[status] || status}</span>
              </label>
              <p className="small">{c.rationale}</p>
              {remembered(c.id) && (
                <p className="small memory-hint">
                  Worked before: passed {remembered(c.id).passed} of {remembered(c.id).tested} tests in {remembered(c.id).incidents} similar
                  incident{remembered(c.id).incidents === 1 ? "" : "s"}.
                </p>
              )}
              {c.preconditions.length > 0 && (
                <ul className="pres">
                  {c.preconditions.map((p) => (
                    <Pre key={p.name} p={p} />
                  ))}
                </ul>
              )}
              {c.provenance?.quote_time && (
                <p className="small muted">
                  Recorded quote from {c.provenance.quote_time.slice(0, 10)} ({c.provenance.age_days} days old), timestamp kept as recorded.
                </p>
              )}
              {r?.tested && (
                <p className="small">
                  Answer <strong className="mono">{JSON.stringify(r.final_answer)}</strong> · {r.passed}/{r.k} branches passed
                  {r.failed_checks?.length ? <> · failed: {r.failed_checks.join(", ")}</> : null}
                </p>
              )}
            </div>
          );
        })}
      </div>
      {error && <div className="note bad section-gap">{error}</div>}
      <div className="alt-actions">
        <span className="label" style={{ margin: 0 }}>Replay mode</span>
        <Segmented
          value={mode}
          onChange={setMode}
          options={[
            ["recorded", "Recorded", false, "Reuse recorded responses for unchanged inputs: isolates the change"],
            ["fresh", "Fresh", false, "Re-execute every step after the fork without the cache: current conditions"],
          ]}
        />
        <span className="label" style={{ margin: 0 }}>Branches</span>
        <Segmented value={k} onChange={setK} options={[[1, "1"], [3, "3"], [5, "5"]]} />
        <button className="btn primary" onClick={test} disabled={busy || !picked.length}>
          {busy ? <LoaderCircle size={16} className="spin" /> : <FlaskConical size={16} />}
          {busy ? "Testing…" : `Test ${picked.length} alternative${picked.length === 1 ? "" : "s"}`}
        </button>
      </div>
      {experiment && <BranchTable run={run} experiment={experiment} />}
    </Panel>
  );
}

function BranchTable({ run, experiment }) {
  const t = experiment.totals;
  const rows = [
    {
      key: "original",
      label: "Original run",
      status: experiment.original.acceptance?.outcome === "passed" ? "passed" : "original_failed",
      answer: experiment.original.final_answer,
      failed: experiment.original.acceptance?.checks?.filter((c) => c.status === "fail").map((c) => c.label) || [],
      rerun: "—",
      tokens: "—",
      cost: "—",
    },
    ...experiment.branches
      .filter((b) => b.tested)
      .map((b) => ({
        key: b.id,
        label: b.label,
        status: b.status,
        answer: b.final_answer,
        failed: b.failed_checks || [],
        rerun:
          (b.rerun_steps || []).join(", ") +
          (b.cache_hit_steps?.length ? ` · ${b.cache_hit_steps.length} served from cache` : "") +
          (b.reused_steps?.length ? ` · reused ${b.reused_steps.join(", ")}` : ""),
        tokens: b.billed_tokens,
        cost: money(b.cost_usd),
      })),
  ];
  return (
    <div className="section-gap">
      <div className={"note " + (experiment.verdict === "supported" ? "good" : "warn")}>
        <span>
          <strong>{experiment.verdict === "supported" ? "Suspect supported by replay." : "Suspect not supported."}</strong> {experiment.verdict_text}
        </span>
      </div>
      <div className="table-wrap section-gap">
        <table>
          <thead>
            <tr>
              <th>Branch</th>
              <th>Result</th>
              <th>Answer</th>
              <th>Failed checks</th>
              <th>Steps re-run</th>
              <th>New tokens</th>
              <th>Cost</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key}>
                <td>
                  <strong>{r.label}</strong>
                </td>
                <td>
                  <span className={"badge " + STATUS_CLASS[r.status]}>{STATUS_TEXT[r.status] || r.status}</span>
                </td>
                <td className="num-cell">{JSON.stringify(r.answer)}</td>
                <td className="small">{r.failed.length ? r.failed.join(", ") : <span className="muted">none</span>}</td>
                <td className="num-cell">{r.rerun}</td>
                <td className="num-cell">{r.tokens}</td>
                <td className="num-cell">{r.cost}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted section-gap" style={{ marginTop: 12 }}>
        {experiment.mode === "fresh" ? "Fresh validation (no cache). " : "Recorded investigation (cache reused for unchanged inputs). "}
        Total for this experiment: {t.tested} branches tested ({t.passed} passed, {t.rejected} rejected), {t.billed_tokens} tokens,{" "}
        {money(t.cost_usd)}, {Math.round(t.wall_ms)} ms. The original run {run.run_id.slice(0, 14)} is unchanged.
      </p>
    </div>
  );
}
