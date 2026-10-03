import { useMemo, useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  Box,
  CircleDot,
  Database,
  GitBranch,
  Layers3,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  TriangleAlert,
  Zap,
} from "lucide-react";
import {
  AreaChart,
  Area,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import { diagnosisOf, percent, short, time, familyLabels } from "./api";
import { Badge, Empty, Metric, PageHeader } from "./components";
export default function Overview({
  runs,
  stats,
  loading,
  openRun,
  navigate,
  reload,
}) {
  const [search, setSearch] = useState(""),
    [filter, setFilter] = useState("all"),
    [family, setFamily] = useState("all");
  const totals = stats?.totals || {},
    failed = runs.filter((r) => r.status === "FAILED");
  const filtered = runs.filter(
    (r) =>
      (filter === "all" || r.status === filter) &&
      (family === "all" || r.task_family === family) &&
      [r.run_id, r.prompt, r.task_family]
        .join(" ")
        .toLowerCase()
        .includes(search.toLowerCase()),
  );
  const activity = useMemo(() => {
    const buckets = {};
    [...runs]
      .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
      .forEach((r) => {
        const key = time(r.created_at);
        buckets[key] ||= { time: key, passed: 0, failed: 0 };
        if (r.status === "PASSED") buckets[key].passed++;
        if (r.status === "FAILED") buckets[key].failed++;
      });
    return Object.values(buckets);
  }, [runs]);
  const components = stats?.by_component || [];
  return (
    <>
      <PageHeader
        eyebrow="OBSERVABILITY / OVERVIEW"
        title="Your agents. Understood."
        description="Follow every execution. Find the failure. Prove the fix."
      >
        <button className="button secondary" onClick={() => navigate("break")}>
          <Zap size={15} />
          Break it<span className="mini-tag">DEMO</span>
        </button>
        <button className="button primary" onClick={() => navigate("live")}>
          <Plus size={17} />
          New run
        </button>
      </PageHeader>
      <div className="metrics-grid">
        <Metric
          label="Total runs"
          value={totals.runs ?? runs.length}
          detail="All recorded executions"
          icon={Layers3}
        />
        <Metric
          label="Failed runs"
          value={totals.failed ?? failed.length}
          detail={
            runs.length
              ? percent(failed.length / runs.length) + " of recorded runs"
              : "Awaiting first execution"
          }
          icon={TriangleAlert}
          tone="red-text"
        />
        <Metric
          label="Diagnosis latency"
          value={
            stats?.diagnosis_latency_ms != null ? (
              <>
                {Number(stats.diagnosis_latency_ms).toFixed(1)}
                <small>ms</small>
              </>
            ) : (
              "—"
            )
          }
          detail="Local diagnosis · measured average"
          icon={Zap}
        />
        <Metric
          label="Localization accuracy"
          value={percent(stats?.top1)}
          detail={
            stats?.top1 != null
              ? "Measured held-out Top-1"
              : "Run evaluation to measure Top-1"
          }
          icon={CircleDot}
        />
      </div>
      <div className="chart-grid">
        <section className="panel activity-panel">
          <div className="panel-heading">
            <div>
              <h2>Execution activity</h2>
              <p>A pulse on your agent runs</p>
            </div>
            <div className="legend">
              <span>
                <i className="green-bg" />
                Passed
              </span>
              <span>
                <i className="red-bg" />
                Failed
              </span>
              <span className="subtle-chip">Recorded runs</span>
            </div>
          </div>
          <div className="activity-chart">
            {activity.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={activity}
                  margin={{ left: -25, right: 10, top: 10, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="passFill" x1="0" y1="0" x2="0" y2="1">
                      <stop
                        offset="0%"
                        stopColor="#37a582"
                        stopOpacity={0.18}
                      />
                      <stop offset="100%" stopColor="#37a582" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="failFill" x1="0" y1="0" x2="0" y2="1">
                      <stop
                        offset="0%"
                        stopColor="#e25c4a"
                        stopOpacity={0.15}
                      />
                      <stop offset="100%" stopColor="#e25c4a" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid
                    strokeDasharray="3 5"
                    vertical={false}
                    stroke="#edf0f3"
                  />
                  <XAxis
                    dataKey="time"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 11, fill: "#87909f" }}
                    dy={7}
                  />
                  <YAxis
                    allowDecimals={false}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 11, fill: "#87909f" }}
                  />
                  <Tooltip
                    contentStyle={{
                      border: "1px solid #e5e7eb",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="passed"
                    name="Passed"
                    stroke="#37a582"
                    fill="url(#passFill)"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                  <Area
                    type="monotone"
                    dataKey="failed"
                    name="Failed"
                    stroke="#e25c4a"
                    fill="url(#failFill)"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <Empty title="The recorder is ready">
                Start a run to see execution activity.
              </Empty>
            )}
          </div>
        </section>
        <section className="panel components-panel">
          <div className="panel-heading">
            <div>
              <h2>Where failures begin</h2>
              <p>Ranked by suspected root-cause component</p>
            </div>
            <GitBranch size={17} className="muted" />
          </div>
          <div className="components-list">
            {components.length ? (
              components.slice(0, 4).map((c, i) => (
                <div className="component" key={c.component}>
                  <div>
                    <span className="component-index">0{i + 1}</span>
                    <code>{c.component}</code>
                    <strong>{c.count}</strong>
                  </div>
                  <div className="bar-track">
                    <div
                      style={{
                        width:
                          (c.count /
                            Math.max(...components.map((x) => x.count))) *
                            100 +
                          "%",
                        background: [
                          "#e6725e",
                          "#eaa878",
                          "#d2b596",
                          "#a3b3b2",
                        ][i],
                      }}
                    />
                  </div>
                </div>
              ))
            ) : (
              <Empty icon={ShieldCheck} title="No failure clusters">
                Failed runs will appear here.
              </Empty>
            )}
          </div>
          <div className="panel-note">
            <CircleDot size={12} />
            Attribution uses an evidence-based baseline
          </div>
        </section>
      </div>
      <section className="panel run-panel">
        <div className="panel-heading">
          <div className="inline-title">
            <h2>Recorded runs</h2>
            <span className="count">{runs.length}</span>
          </div>
          <button
            className="icon-button"
            aria-label="Refresh runs"
            onClick={reload}
          >
            <RefreshCw size={15} />
          </button>
        </div>
        <div className="table-toolbar">
          <div className="tabs">
            {[
              ["all", "All runs"],
              ["FAILED", "Failed"],
              ["PASSED", "Passed"],
            ].map(([id, label]) => (
              <button
                key={id}
                className={filter === id ? "active" : ""}
                onClick={() => setFilter(id)}
              >
                {label}
                {id === "FAILED" && <span>{failed.length}</span>}
              </button>
            ))}
          </div>
          <div className="table-filters">
            <div className="search-box">
              <Search size={15} />
              <input
                aria-label="Search runs"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search runs…"
              />
              <kbd>/</kbd>
            </div>
            <label className="family-filter">
              <SlidersHorizontal size={14} />
              <select
                aria-label="Task family"
                value={family}
                onChange={(e) => setFamily(e.target.value)}
              >
                <option value="all">All families</option>
                {Object.entries(familyLabels).map(([v, l]) => (
                  <option value={v} key={v}>
                    {l}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>RUN / TASK</th>
                <th>STATUS</th>
                <th>FAMILY</th>
                <th>TOP SUSPECT</th>
                <th>DURATION</th>
                <th>RECORDED</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => {
                const cause = diagnosisOf(r).root_cause;
                return (
                  <tr key={r.run_id} onClick={() => openRun(r.run_id)}>
                    <td>
                      <button
                        className="run-link"
                        onClick={(e) => {
                          e.stopPropagation();
                          openRun(r.run_id);
                        }}
                      >
                        <span
                          className={
                            "run-icon " +
                            (r.status === "FAILED" ? "failure" : "")
                          }
                        >
                          <GitBranch size={16} />
                        </span>
                        <span>
                          <strong className="mono">{short(r.run_id)}</strong>
                          <span className="task-preview">
                            {r.prompt || "Sandbox execution"}
                          </span>
                        </span>
                      </button>
                    </td>
                    <td>
                      <Badge status={r.status} />
                    </td>
                    <td>
                      <span className="family-label">
                        {familyLabels[r.task_family] || r.task_family}
                      </span>
                    </td>
                    <td>
                      {r.status === "FAILED" && (cause || r.suspect_step) ? (
                        <div className="suspect">
                          <span className="red-dot" />
                          <code>
                            {cause?.node ||
                              r.suspect_node ||
                              "Step " + r.suspect_step}
                          </code>
                          <span>{percent(cause?.confidence ?? r.blame)}</span>
                        </div>
                      ) : (
                        <span className="muted">—</span>
                      )}
                    </td>
                    <td className="mono muted">
                      {r.total_time != null
                        ? (Number(r.total_time) / 1000).toFixed(2) + "s"
                        : "—"}
                    </td>
                    <td className="muted">{time(r.created_at)}</td>
                    <td>
                      <ArrowUpRight size={15} className="muted" />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {!filtered.length && (
          <Empty title={loading ? "Connecting to recorder…" : "No runs found"}>
            {loading
              ? "Loading your local workspace."
              : "Start a new run or adjust the filters."}
          </Empty>
        )}
        <div className="table-bottom">
          <span>
            Showing {filtered.length} of {runs.length} runs
          </span>
          <span>
            <Database size={12} />
            Persisted in local SQLite
          </span>
        </div>
      </section>
      <div className="onboarding-banner">
        <div className="onboarding-icon">
          <Box size={24} />
        </div>
        <div>
          <strong>A wrong answer is only the beginning.</strong>
          <p>
            Inject a fault into a successful run, then follow the evidence back
            to its source.
          </p>
        </div>
        <button onClick={() => navigate("break")}>
          Try the judge challenge
          <ArrowRight size={16} />
        </button>
      </div>
    </>
  );
}
