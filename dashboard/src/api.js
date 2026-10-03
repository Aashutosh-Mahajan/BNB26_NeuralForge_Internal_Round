export async function api(path, options = {}) {
  const response = await fetch("/api" + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail || "Request failed"),
    );
  return data;
}
export const post = (path, body) =>
  api(path, { method: "POST", body: JSON.stringify(body) });
export const asRuns = (data) => (Array.isArray(data) ? data : data.runs || []);
export const pretty = (data) => JSON.stringify(data, null, 2);
export const short = (id) => (id ? id.slice(0, 14) : "—");
export const percent = (n, digits = 1) =>
  n == null || Number.isNaN(n) ? "—" : (n * 100).toFixed(digits) + "%";
export const time = (date) =>
  date
    ? new Date(date).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      })
    : "—";
export const money = (n) =>
  n == null
    ? "—"
    : n === 0
      ? "$0"
      : n < 0.01
        ? "$" + n.toFixed(5)
        : "$" + n.toFixed(3);
export const diagnosisOf = (r) => r?.diagnosis || r || {};
export const familyLabels = {
  finance: "Finance",
  sql: "SQL",
  doc_qa: "Document QA",
  math: "Math",
};
export const nodeLabel = (s) => (s || "").replaceAll("_", " ");
export const compactJson = (value, limit = 140) => {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text && text.length > limit ? text.slice(0, limit) + "…" : text;
};
