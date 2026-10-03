import { Inbox } from "lucide-react";
import { familyLabels, nodeLabel, pretty, short } from "./api";

export function StatusBadge({ status }) {
  const label =
    { PASSED: "Passed", FAILED: "Failed", RUNNING: "Running", QUEUED: "Queued", ERROR: "Error" }[status] ||
    status ||
    "—";
  return <span className={"badge " + (status || "").toLowerCase()}>{label}</span>;
}

export function PageHead({ title, children, actions }) {
  return (
    <header className="page-head">
      <div>
        <h1>{title}</h1>
        {children && <p>{children}</p>}
      </div>
      {actions && <div className="actions">{actions}</div>}
    </header>
  );
}

export function Panel({ title, aside, children, className = "", bodyClass = "panel-body" }) {
  return (
    <section className={"panel " + className}>
      {(title || aside) && (
        <div className="panel-head">
          {title && <h2>{title}</h2>}
          {aside && <span className="aside">{aside}</span>}
        </div>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, unit, note, tone = "" }) {
  return (
    <div className={"stat " + tone}>
      <span className="label">{label}</span>
      <div className={"stat-value " + tone}>
        {value}
        {unit && <small>{unit}</small>}
      </div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  );
}

export function Json({ value }) {
  return <pre className="json">{pretty(value)}</pre>;
}

export function Empty({ title, children, action, icon: Icon = Inbox }) {
  return (
    <div className="empty">
      <Icon size={30} />
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="label">{label}</span>
      {children}
      {hint && <div className="hint">{hint}</div>}
    </label>
  );
}

export function Segmented({ options, value, onChange }) {
  return (
    <div className="segmented" role="group" aria-label="Choose an option">
      {options.map(([id, label, disabled, title]) => (
        <button
          key={id}
          type="button"
          aria-pressed={value === id}
          className={value === id ? "active" : ""}
          disabled={disabled}
          title={title}
          onClick={() => onChange(id)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

export function RunPicker({ runs, value, onChange, label = "Run", filter = () => true, hint }) {
  return (
    <Field label={label} hint={hint}>
      <select value={value || ""} onChange={(e) => onChange(e.target.value)}>
        <option value="">Choose a run…</option>
        {runs.filter(filter).map((r) => (
          <option key={r.run_id} value={r.run_id}>
            {r.status === "FAILED" ? "✕" : "✓"} {familyLabels[r.task_family] || r.task_family} ·{" "}
            {(r.prompt || "").slice(0, 60)} · {short(r.run_id)}
          </option>
        ))}
      </select>
    </Field>
  );
}

/* The flight tape: one cell per recorded step. `role(step)` returns a class:
   root | affected | fine | checkpoint | reused | rerun | patched | pending. */
export function Tape({ title, steps, role, flag, selected, onSelect, legend, footer, placeholders = 0 }) {
  return (
    <section className="tape" aria-label={title}>
      <div className="tape-head">
        <h2>{title}</h2>
        {legend && (
          <div className="tape-legend">
            {legend.map(([color, text]) => (
              <span key={text}>
                <i style={{ background: color }} />
                {text}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="tape-track" style={{ "--cols": Math.min(10, Math.max(1, steps.length + placeholders)) }}>
        {steps.map((s) => {
          const kind = role ? role(s) : "fine";
          const tokens = (s.tokens_in || 0) + (s.tokens_out || 0);
          return (
            <button
              key={s.step_id}
              type="button"
              className={"cell " + kind + (selected === s.step_id ? " selected" : "")}
              onClick={() => onSelect?.(s)}
              aria-pressed={selected === s.step_id}
            >
              {flag?.(s) && <span className="flag">{flag(s)}</span>}
              <span className="num">STEP {String(s.step_id).padStart(2, "0")}</span>
              <span className="name">{nodeLabel(s.node_name)}</span>
              <span className="meta">
                {s.tool_error
                  ? "error"
                  : s.llm_call
                    ? "LLM · " + tokens + " tok"
                    : Number(s.latency_ms || 0).toFixed(1) + " ms"}
              </span>
            </button>
          );
        })}
        {Array.from({ length: placeholders }).map((_, i) => (
          <div key={"p" + i} className="cell pending">
            <span className="num">STEP {String(steps.length + i + 1).padStart(2, "0")}</span>
            <span className="name">waiting…</span>
          </div>
        ))}
      </div>
      {footer && <div className="tape-foot">{footer}</div>}
    </section>
  );
}

export function MiniTape({ count = 10, suspect, status }) {
  return (
    <span className={"mini-tape " + (status || "").toLowerCase()} aria-label={suspect ? "Leading suspect at step " + suspect : "Recorded steps"}>
      {Array.from({ length: count }).map((_, i) => (
        <i
          key={i}
          className={status === "FAILED" && suspect && i + 1 === suspect ? "root" : ""}
        />
      ))}
    </span>
  );
}

export function HBars({ rows }) {
  return (
    <div className="hbars">
      {rows.map((r) => (
        <div key={r.label} className={"hbar " + (r.kind || "")}>
          <span>{r.label}</span>
          <div className="track">
            <div className="fill" style={{ width: Math.max(0, Math.min(100, r.value * 100)) + "%" }} />
          </div>
          <span className="val">{r.value == null ? "—" : (r.value * 100).toFixed(1) + "%"}</span>
        </div>
      ))}
    </div>
  );
}

/* Dependency graph: columns by dependency depth, real parent → child edges. */
const ROLE_FILL = {
  root: ["#fdebe7", "#d9381e"],
  affected: ["#fff4e0", "#c27400"],
  fine: ["#e5f4ec", "#1f7a4d"],
  neutral: ["#f6f7f9", "#9aa5b8"],
};
export function DepGraph({ steps, role, selected, onSelect }) {
  const depth = {};
  for (const s of steps) depth[s.step_id] = Math.max(0, ...(s.parent_step_ids || []).map((p) => (depth[p] ?? 0) + 1));
  const columns = {};
  for (const s of steps) (columns[depth[s.step_id]] ||= []).push(s);
  const W = 168, H = 64, BW = 140, BH = 46;
  const pos = {};
  Object.entries(columns).forEach(([d, items]) => items.forEach((s, i) => (pos[s.step_id] = { x: Number(d) * W + 10, y: i * H + 10 })));
  const width = (Math.max(...Object.keys(columns).map(Number)) + 1) * W;
  const height = Math.max(...Object.values(columns).map((c) => c.length)) * H + 10;
  return (
    <svg className="depgraph" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Step dependency graph">
      {steps.flatMap((s) =>
        (s.parent_step_ids || []).map((p) => {
          const a = pos[p], b = pos[s.step_id];
          if (!a || !b) return null;
          const r = role(s);
          const hot = role(steps.find((x) => x.step_id === p)) !== "fine" && r !== "fine";
          return (
            <path
              key={p + "-" + s.step_id}
              d={`M${a.x + BW},${a.y + BH / 2} C${a.x + BW + 24},${a.y + BH / 2} ${b.x - 24},${b.y + BH / 2} ${b.x},${b.y + BH / 2}`}
              fill="none"
              stroke={hot ? "#e0892b" : "#c3cad5"}
              strokeWidth={hot ? 2.5 : 1.5}
            />
          );
        }),
      )}
      {steps.map((s) => {
        const [fill, stroke] = ROLE_FILL[role(s)] || ROLE_FILL.neutral;
        const p = pos[s.step_id];
        return (
          <g key={s.step_id} transform={`translate(${p.x},${p.y})`} onClick={() => onSelect?.(s)} style={{ cursor: "pointer" }}>
            <rect width={BW} height={BH} rx="6" fill={fill} stroke={stroke} strokeWidth={selected === s.step_id ? 3 : 1.5} />
            <text x="10" y="18" fontSize="11" fontWeight="700" fill="#3d4553">STEP {String(s.step_id).padStart(2, "0")}</text>
            <text x="10" y="35" fontSize="14" fontWeight="600" fill="#121722">{nodeLabel(s.node_name).slice(0, 17)}</text>
          </g>
        );
      })}
    </svg>
  );
}

const CAPS = [
  ["record", "Record"],
  ["diagnose", "Diagnose"],
  ["checkpoint", "Checkpoint"],
  ["fork", "Fork"],
  ["resume", "Resume"],
  ["selective_reuse", "Reuse unaffected steps"],
];
export const BUILTIN_CAPS = { record: true, diagnose: true, checkpoint: true, fork: true, resume: true, selective_reuse: true };
export function Capabilities({ caps, adapter }) {
  return (
    <div className="caps" aria-label={"What Black Box can do for " + adapter}>
      <span className="caps-title">{adapter}</span>
      {CAPS.map(([key, label]) => (
        <span key={key} className={"cap " + (caps?.[key] ? "yes" : "no")} title={caps?.[key] ? "Supported" : "Not supported for this adapter"}>
          {caps?.[key] ? "✓" : "✕"} {label}
        </span>
      ))}
    </div>
  );
}
