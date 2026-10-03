import { useCallback, useEffect, useState } from "react";
import {
  ArrowUpRight,
  Box,
  ChevronRight,
  FlaskConical,
  GitBranch,
  GitCompareArrows,
  Layers3,
  Radio,
  RefreshCw,
  Terminal,
  TriangleAlert,
} from "lucide-react";
import { api, asRuns } from "./api";
import Overview from "./Overview";
import {
  Diagnosis,
  LiveRun,
  Replay,
  Compare,
  BreakRun,
  Evaluation,
} from "./Investigation";
const navigation = [
  { id: "runs", label: "Runs", icon: Layers3 },
  { id: "live", label: "Live", icon: Radio },
  { id: "replay", label: "Replay", icon: GitBranch },
  { id: "compare", label: "Compare", icon: GitCompareArrows },
  { id: "evaluation", label: "Evaluation", icon: FlaskConical },
];
export default function App() {
  const [view, setView] = useState("runs"),
    [runs, setRuns] = useState([]),
    [stats, setStats] = useState(null),
    [selected, setSelected] = useState(null),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(true),
    [refresh, setRefresh] = useState(0),
    [compareIds, setCompareIds] = useState(null);
  const reload = useCallback(async () => {
    try {
      const [r, s] = await Promise.all([api("/runs"), api("/stats")]);
      setRuns(asRuns(r));
      setStats(s);
      setError("");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    reload();
  }, [reload, refresh]);
  const openRun = async (id, target = "diagnosis") => {
    try {
      setSelected(await api("/runs/" + id));
      setView(target);
    } catch (e) {
      setError(e.message);
    }
  };
  const navigate = (v) => {
    setView(v);
    setError("");
  };
  const compare = (a, b) => {
    setCompareIds([a, b]);
    setView("compare");
    setRefresh((x) => x + 1);
  };
  return (
    <div className="app-shell">
      <header className="topbar">
        <button
          className="brand"
          onClick={() => navigate("runs")}
          aria-label="Black Box home"
        >
          <span className="brand-icon">
            <Box size={23} strokeWidth={1.6} />
          </span>
          BLACK BOX<span className="beta">BETA</span>
        </button>
        <nav>
          {navigation.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={
                view === id ||
                (id === "runs" && ["diagnosis", "break"].includes(view))
                  ? "nav-active"
                  : ""
              }
              onClick={() => navigate(id)}
            >
              <Icon size={15} />
              {label}
              {id === "live" && <i className="live-dot" />}
            </button>
          ))}
        </nav>
        <div className="workspace">
          <span className="workspace-dot" />
          <span>Local workspace</span>
          <span className="avatar">BB</span>
        </div>
      </header>
      <div className="workspace-bar">
        <div>
          <span>Workspace</span>
          <ChevronRight size={12} />
          <strong>Sandbox agents</strong>
        </div>
        <span className="environment">
          <Terminal size={12} /> DEVELOPMENT <span className="v-divider" />{" "}
          v0.1.0
        </span>
      </div>
      <main>
        {error && (
          <div className="error-banner" role="alert">
            <TriangleAlert size={17} />
            <span>{error}. Check that the API is running on port 8010.</span>
            <button
              onClick={() => {
                setError("");
                reload();
              }}
            >
              <RefreshCw size={15} /> Retry
            </button>
          </div>
        )}
        {view === "runs" && (
          <Overview
            runs={runs}
            stats={stats}
            loading={loading}
            openRun={openRun}
            navigate={navigate}
            reload={() => setRefresh((x) => x + 1)}
          />
        )}
        {view === "live" && (
          <LiveRun onDone={() => setRefresh((x) => x + 1)} openRun={openRun} />
        )}
        {view === "diagnosis" && selected && (
          <Diagnosis
            run={selected}
            onBack={() => navigate("runs")}
            onReplay={() => navigate("replay")}
            onBreak={() => navigate("break")}
          />
        )}
        {view === "replay" && (
          <Replay
            runs={runs}
            selected={selected}
            setSelected={setSelected}
            onCompare={compare}
            onUpdate={() => setRefresh((x) => x + 1)}
          />
        )}
        {view === "compare" && <Compare runs={runs} initialIds={compareIds} />}
        {view === "break" && (
          <BreakRun
            runs={runs}
            selected={selected}
            onDone={(id) => {
              setRefresh((x) => x + 1);
              openRun(id);
            }}
          />
        )}
        {view === "evaluation" && <Evaluation />}
      </main>
      <footer>
        <span>
          <span className="status-dot" />
          All tools sandboxed
        </span>
        <span>Recorded. Diagnosed. Replayed.</span>
        <a href="/api/docs" target="_blank" rel="noreferrer">
          API reference <ArrowUpRight size={12} />
        </a>
      </footer>
    </div>
  );
}
