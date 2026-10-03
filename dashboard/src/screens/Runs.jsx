import { useMemo, useState } from "react";
import { Play, RefreshCw, Zap } from "lucide-react";
import { familyLabels, money, nodeLabel, percent, short, time } from "../api";
import { Empty, MiniTape, PageHead, Panel, Stat, StatusBadge } from "../ui";

function Activity({ runs }) {
  const columns = useMemo(() => {
    const stamps = runs.filter((r) => r.created_at).map((r) => ({ t: new Date(r.created_at).getTime(), s: r.status }));
    if (!stamps.length) return [];
    const lo = Math.min(...stamps.map((x) => x.t));
    const hi = Math.max(...stamps.map((x) => x.t));
    const buckets = 24;
    const width = Math.max(1, (hi - lo) / buckets);
    const out = Array.from({ length: buckets }, (_, i) => ({ passed: 0, failed: 0, label: time(new Date(lo + i * width).toISOString()) }));
    for (const x of stamps) {
      const c = out[Math.min(buckets - 1, Math.floor((x.t - lo) / width))];
      if (x.s === "PASSED") c.passed++;
      if (x.s === "FAILED") c.failed++;
    }
    return out;
  }, [runs]);
  const max = Math.max(1, ...columns.map((c) => c.passed + c.failed));
  if (!columns.length) return <p className="muted">No runs recorded yet.</p>;
  return (
    <>
      <div className="activity-scale">Peak bucket: {max} {max === 1 ? "run" : "runs"}</div>
      <div className="activity" role="img" aria-label={`Passed and failed runs over time. Largest time bucket: ${max} runs.`}>
        {columns.map((c, i) => (
          <div className="col" key={i} title={`${c.label}: ${c.passed} passed, ${c.failed} failed`}>
            <span className="p" style={{ height: (c.passed / max) * 100 + "%" }} />
            <span className="f" style={{ height: (c.failed / max) * 100 + "%" }} />
          </div>
        ))}
      </div>
      <div className="small muted" style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
        <span>{columns[0].label}</span>
        <span>
          <span style={{ color: "var(--pass)" }}>■</span> passed &nbsp; <span style={{ color: "var(--root)" }}>■</span> failed
        </span>
        <span>{columns[columns.length - 1].label}</span>
      </div>
    </>
  );
}

export default function Runs({ runs, stats, loading, openRun, go, reload }) {
  const [status, setStatus] = useState("all");
  const [family, setFamily] = useState("all");
  const [search, setSearch] = useState("");
  const totals = stats?.totals || {};
  const components = stats?.by_component || [];
  const failedTotal = totals.failed || 0;
  const visible = runs.filter(
    (r) =>
      (status === "all" || r.status === status) &&
      (family === "all" || r.task_family === family) &&
      [r.run_id, r.prompt].join(" ").toLowerCase().includes(search.toLowerCase()),
  );
  const ensemble = stats?.method === "ensemble";

  return (
    <>
      <PageHead
        title="Execution overview"
        actions={
          <>
            <button className="btn" onClick={() => go("break")}>
              <Zap size={16} /> Break a run
            </button>
            <button className="btn primary" onClick={() => go("live")}>
              <Play size={16} /> Start a run
            </button>
          </>
        }
      >
        Your executions, their outcomes, and the evidence behind each failure. Open a run to investigate its next possible path.
      </PageHead>

      <div className="stats">
        <Stat label="Runs recorded" value={totals.runs ?? runs.length} note={`${totals.passed ?? 0} passed`} />
        <Stat
          label="Failed runs"
          value={failedTotal}
          tone={failedTotal ? "tone-root" : ""}
          note={totals.runs ? percent(failedTotal / totals.runs, 0) + " of all runs" : "None yet"}
        />
        <Stat
          label="Time to diagnose"
          value={stats?.diagnosis_latency_ms != null ? Math.round(stats.diagnosis_latency_ms) : "—"}
          unit={stats?.diagnosis_latency_ms != null ? "ms" : ""}
          note={ensemble ? "trained diagnosis · per run" : "rule-based diagnosis · per run"}
        />
        <Stat
          label={ensemble ? "Model localization" : "Baseline localization"}
          value={stats?.top1 != null ? percent(stats.top1, 1) : "—"}
          tone={stats?.top1 != null ? "tone-pass" : ""}
          note={stats?.top1 != null ? <button className="text-link metric-link" onClick={() => go("evaluation")}>Top-1 on held-out test runs · view protocol ↗</button> : "not evaluated yet"}
        />
      </div>

      <div className="grid-2" style={{ marginBottom: 20 }}>
        <Panel title="Leading failure suspects" className="suspects-panel" aside={failedTotal ? `${failedTotal} failed ${failedTotal === 1 ? "run" : "runs"}` : null}>
          {components.length ? (
            <div className="hbars">
              {components.slice(0, 5).map((c, i) => (
                <div className={"hbar" + (i === 0 ? " ours" : "")} key={c.component}>
                  <code style={{ fontSize: 14 }}>{nodeLabel(c.component)}</code>
                  <div className="track">
                    <div className="fill" style={{ width: (c.count / components[0].count) * 100 + "%", background: i === 0 ? "var(--root)" : undefined }} />
                  </div>
                  <span className="val">{percent(c.count / failedTotal, 0)}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">No failures yet. Break a run to see one diagnosed.</p>
          )}
          {components.length > 0 && (
            <p className="small muted" style={{ marginTop: 14 }}>
              {percent(components[0].count / failedTotal, 0)} of failed runs point to <strong>{nodeLabel(components[0].component)}</strong>.
              Tokens recorded {(stats?.total_tokens || 0).toLocaleString()} · OpenAI spend {money(stats?.cost_usd)}.
            </p>
          )}
        </Panel>
        <Panel title="Activity" aside="Runs per time bucket">
          <Activity runs={runs} />
        </Panel>
      </div>

      <section className="panel">
        <div className="panel-head"><h2>Recorded runs</h2><span className="aside">{visible.length} of {runs.length} runs · select a task to investigate</span></div>
        <div className="toolbar">
          <div className="segmented">
            {[
              ["all", "All"],
              ["FAILED", `Failed (${runs.filter((r) => r.status === "FAILED").length})`],
              ["PASSED", "Passed"],
            ].map(([id, label]) => (
              <button key={id} aria-pressed={status === id} className={status === id ? "active" : ""} onClick={() => setStatus(id)}>
                {label}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <input aria-label="Search runs" placeholder="Search task or run ID" value={search} onChange={(e) => setSearch(e.target.value)} />
            <select aria-label="Task family" value={family} onChange={(e) => setFamily(e.target.value)}>
              <option value="all">All task types</option>
              {Object.entries(familyLabels).map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </select>
            <button className="btn small" onClick={reload} aria-label="Refresh runs">
              <RefreshCw size={14} />
            </button>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Task</th>
                <th>Result</th>
                <th>Leading suspect</th>
                <th>Tokens · cost</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((r) => {
                const root = r.root_cause;
                return (
                  <tr key={r.run_id} className="clickable" onClick={() => openRun(r.run_id)}>
                    <td className="task-cell">
                      <button onClick={e => { e.stopPropagation(); openRun(r.run_id); }} aria-label={`Investigate ${r.prompt || r.run_id}`}><strong title={r.prompt}>{r.prompt || "—"}</strong></button>
                      <span>
                        {familyLabels[r.task_family] || r.task_family} · {r.llm_provider === "sandbox" || !r.model ? "sandbox" : r.model} · {short(r.run_id)}
                        {r.fault_type ? " · injected: " + nodeLabel(r.fault_type) : ""}
                        {r.replay ? " · replay" : ""}
                      </span>
                    </td>
                    <td>
                      <StatusBadge status={r.status} />
                    </td>
                    <td>
                      <MiniTape count={r.step_count || 10} suspect={r.status === "FAILED" ? root?.step : null} status={r.status} />
                      <div className="small" style={{ marginTop: 6 }}>
                        {r.status === "FAILED" && root ? (
                          <>
                            <strong>Step {root.step}</strong> · {nodeLabel(root.node)}
                          </>
                        ) : (
                          <span className="muted">{r.status === "PASSED" ? "Checks passed" : "Awaiting diagnosis"}</span>
                        )}
                      </div>
                    </td>
                    <td className="num-cell">
                      {(r.total_tokens || 0).toLocaleString()}
                      <div className="muted">{money(r.cost_usd)}</div>
                    </td>
                    <td className="num-cell muted">{time(r.created_at)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {!visible.length && (
          <Empty
            title={loading ? "Loading runs…" : runs.length ? "No runs match these filters" : "No runs yet"}
            action={
              !loading && !runs.length ? (
                <button className="btn primary" onClick={() => go("live")}>
                  <Play size={16} /> Start a run
                </button>
              ) : null
            }
          >
            {!loading && !runs.length ? "Start a run to record your first trace." : null}
          </Empty>
        )}
      </section>
    </>
  );
}
