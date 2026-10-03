import { Layers3 } from "lucide-react";
import { pretty, short, familyLabels } from "./api";
export function Badge({ status }) {
  return (
    <span className={"badge " + (status || "").toLowerCase()}>
      <span />
      {status === "PASSED"
        ? "Passed"
        : status === "FAILED"
          ? "Failed"
          : status || "Pending"}
    </span>
  );
}
export function Empty({ icon: Icon = Layers3, title, children }) {
  return (
    <div className="empty">
      <Icon size={30} />
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
export function Json({ value }) {
  return <pre className="json">{pretty(value)}</pre>;
}
export function Metric({ label, value, detail, icon: Icon, tone }) {
  return (
    <div className="metric">
      <div className="metric-top">
        <span>{label}</span>
        <Icon size={17} />
      </div>
      <div className={"metric-value " + (tone || "")}>{value}</div>
      <div className="metric-detail">{detail}</div>
    </div>
  );
}
export function PageHeader({ eyebrow, title, description, children }) {
  return (
    <div className="page-header">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      <div className="header-actions">{children}</div>
    </div>
  );
}
export function RunSelect({
  runs,
  value,
  onChange,
  label = "Recorded run",
  onlyPassed = false,
}) {
  return (
    <label className="field-label">
      {label}
      <select value={value || ""} onChange={(e) => onChange(e.target.value)}>
        <option value="">Select a run…</option>
        {runs
          .filter((r) => !onlyPassed || r.status === "PASSED")
          .map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {short(r.run_id)} · {familyLabels[r.task_family]} · {r.status}
            </option>
          ))}
      </select>
    </label>
  );
}
