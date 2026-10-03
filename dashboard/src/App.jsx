import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { api, asRuns, money } from "./api";
import Runs from "./screens/Runs";
import RunDetail from "./screens/RunDetail";
import Live from "./screens/Live";
import Replay from "./screens/Replay";
import Compare from "./screens/Compare";
import BreakRun from "./screens/BreakRun";
import Evaluation from "./screens/Evaluation";

const NAV = [
  ["runs", "Runs"],
  ["live", "Live run"],
  ["break", "Break it"],
  ["replay", "Replay"],
  ["compare", "Compare"],
  ["evaluation", "Evaluation"],
];

export default function App() {
  const [view, setView] = useState("runs");
  const [runs, setRuns] = useState([]);
  const [stats, setStats] = useState(null);
  const [llm, setLlm] = useState(null);
  const [selected, setSelected] = useState(null);
  const [compareIds, setCompareIds] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const reload = useCallback(async () => {
    try {
      const [r, s, l] = await Promise.all([api("/runs?limit=300"), api("/stats"), api("/llm/status")]);
      setRuns(asRuns(r));
      setStats(s);
      setLlm(l);
      setError("");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  const go = (target) => {
    setView(target);
    setError("");
    window.scrollTo({ top: 0 });
  };
  const openRun = async (id, target = "run") => {
    try {
      setSelected(await api("/runs/" + id));
      go(target);
    } catch (e) {
      setError(e.message);
    }
  };
  const compare = (a, b) => {
    setCompareIds([a, b]);
    go("compare");
    reload();
  };
  const active = view === "run" ? "runs" : view;

  return (
    <>
      <header className="topbar">
        <button className="brand" onClick={() => go("runs")} aria-label="Black Box, go to runs">
          <span className="brand-mark" aria-hidden="true" />
          BLACK BOX
        </button>
        <nav className="nav" aria-label="Main">
          {NAV.map(([id, label]) => (
            <button key={id} className={active === id ? "active" : ""} onClick={() => go(id)} aria-current={active === id ? "page" : undefined}>
              {label}
            </button>
          ))}
        </nav>
        {llm && (
          <div className="agent-pill" title="Default agent model and OpenAI spend so far">
            <span className="dot" />
            Agent <strong>{llm.active_provider === "sandbox" ? "Offline sandbox" : llm.active_model}</strong>
            {llm.openai_key_configured && <> · GPT spend <strong>{money(llm.spent_usd)}</strong></>}
          </div>
        )}
      </header>
      <main>
        {error && (
          <div className="banner-error" role="alert">
            <AlertTriangle size={18} />
            {error}. Check that the server is running on port 8010.
            <button className="btn small" onClick={reload}>
              <RefreshCw size={14} /> Try again
            </button>
          </div>
        )}
        {view === "runs" && <Runs runs={runs} stats={stats} loading={loading} openRun={openRun} go={go} reload={reload} />}
        {view === "run" && selected && (
          <RunDetail
            run={selected}
            onBack={() => go("runs")}
            onReplay={() => go("replay")}
            onBreak={() => go("break")}
          />
        )}
        {view === "live" && <Live llm={llm} onDone={reload} openRun={openRun} />}
        {view === "break" && (
          <BreakRun
            runs={runs}
            selected={selected}
            onDone={(id) => {
              reload();
              openRun(id);
            }}
          />
        )}
        {view === "replay" && (
          <Replay runs={runs} selected={selected} setSelected={setSelected} onCompare={compare} onUpdate={reload} />
        )}
        {view === "compare" && <Compare runs={runs} initialIds={compareIds} />}
        {view === "evaluation" && <Evaluation />}
      </main>
    </>
  );
}
