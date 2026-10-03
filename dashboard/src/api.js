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
export const percent = (n) => (n == null ? "—" : (n * 100).toFixed(1) + "%");
export const time = (date) =>
  date
    ? new Date(date).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      })
    : "—";
export const diagnosisOf = (r) => r?.diagnosis || r || {};
export const familyLabels = {
  finance: "Finance",
  sql: "SQL",
  doc_qa: "Document QA",
  math: "Math",
};
