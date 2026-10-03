import { useEffect, useState } from "react";
import { ArrowRight } from "lucide-react";
import { api, compactJson, nodeLabel } from "../api";
import { Empty, Json, PageHead, Panel, RunPicker, StatusBadge } from "../ui";

function Change({ c }) {
  return (
    <div className="diff-line">
      <span className="muted">{c.path.replace("$.output", "") || "output"}</span>{" "}
      <span className="old">{compactJson(c.before, 70)}</span> <ArrowRight size={12} />{" "}
      <span className="new">{compactJson(c.after, 70)}</span>
    </div>
  );
}

export default function Compare({ runs, initialIds }) {
  const [a, setA] = useState(initialIds?.[0] || "");
  const [b, setB] = useState(initialIds?.[1] || "");
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (initialIds) {
      setA(initialIds[0]);
      setB(initialIds[1]);
    }
  }, [initialIds]);
  useEffect(() => {
    if (!a || !b) return setData(null);
    let live = true;
    Promise.all([api(`/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`), api("/runs/" + a), api("/runs/" + b)])
      .then(([diff, left, right]) => live && (setData({ ...diff, left, right }), setError("")))
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [a, b]);

  return (
    <>
      <PageHead title="Compare two runs">
        Steps are lined up side by side. The first step whose output differs is usually where the story changes.
      </PageHead>
      <div className="grid-2">
        <RunPicker runs={runs} value={a} onChange={setA} label="Original run" />
        <RunPicker runs={runs} value={b} onChange={setB} label="Run to compare (e.g. the replay)" />
      </div>
      {error && <div className="note bad">{error}</div>}
      {!data ? (
        <Panel>
          <Empty title="Choose two runs">After a replay, use “Compare the two runs” to land here automatically.</Empty>
        </Panel>
      ) : (
        <>
          <div className={"verdict " + (data.outcome.after ? "passed" : "failed")}>
            <div>
              <h2>
                {data.first_divergence != null ? <>First difference at <em>step {data.first_divergence}</em></> : "No step output differs"}
              </h2>
              <p>
                Outcome {data.outcome.before ? "passed" : "failed"} → {data.outcome.after ? "passed" : "failed"}. Answer{" "}
                <strong className="mono">{JSON.stringify(data.outcome.answer_before)}</strong> became{" "}
                <strong className="mono">{JSON.stringify(data.outcome.answer_after)}</strong>. {data.changes.length} of{" "}
                {data.aligned_steps.length} steps changed.
              </p>
            </div>
            <div className="gauges">
              <StatusBadge status={data.left.status} />
              <ArrowRight size={18} />
              <StatusBadge status={data.right.status} />
            </div>
          </div>
          <section className="panel">
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Step</th>
                    <th>What changed in the output</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.aligned_steps.map((row, i) => {
                    const outputs = row.changes.filter((c) => c.path.startsWith("$.output"));
                    return (
                      <tr key={i} className={"diff-row" + (row.status !== "unchanged" ? " changed" : "")}>
                        <td style={{ whiteSpace: "nowrap" }}>
                          <strong>{row.a_step ?? row.b_step}</strong> · {nodeLabel(row.node)}
                        </td>
                        <td>
                          {outputs.length ? (
                            outputs.slice(0, 4).map((c, j) => <Change key={j} c={c} />)
                          ) : row.status === "unchanged" ? (
                            <span className="muted">identical</span>
                          ) : (
                            <span className="muted">only inputs or state changed</span>
                          )}
                          {outputs.length > 4 && <div className="small muted">+{outputs.length - 4} more</div>}
                        </td>
                        <td>
                          <span className="tag">{row.status}</span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
          {data.state_diff && (
            <Panel className="section-gap" title="Final agent state" aside={"compared with " + data.state_diff.engine}>
              <Json value={data.state_diff.details} />
            </Panel>
          )}
        </>
      )}
    </>
  );
}
