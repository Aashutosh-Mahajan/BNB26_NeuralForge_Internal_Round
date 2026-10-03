import { useCallback, useEffect, useState } from "react";
import { Activity, AlertTriangle, ArrowLeft, ArrowUpRight, Box, ChevronRight, FlaskConical, GitCompareArrows, GitFork, LayoutDashboard, RefreshCw, Radio, ShieldCheck } from "lucide-react";
import { api, asRuns } from "./api";
import Home from "./screens/Home";
import Runs from "./screens/Runs";
import RunDetail from "./screens/RunDetail";
import Live from "./screens/Live";
import Replay from "./screens/Replay";
import Compare from "./screens/Compare";
import BreakRun from "./screens/BreakRun";
import Evaluation from "./screens/Evaluation";

const NAV = [
  ["runs", "Overview", LayoutDashboard],
  ["live", "Live execution", Radio],
  ["replay", "Replay & alternatives", GitFork],
  ["compare", "Trace comparison", GitCompareArrows],
  ["break", "Failure lab", FlaskConical],
  ["evaluation", "Model evaluation", Activity],
];
const routeFromHash = () => window.location.hash.slice(1) || "home";

export function Brand({ onClick }) {
  return <button className="identity" onClick={onClick} aria-label="Black Box home"><span className="identity-icon"><Box size={22} strokeWidth={1.7} /></span><span>blackbox<span className="identity-period">.</span><small>AGENT FLIGHT RECORDER</small></span></button>;
}

export default function App() {
  const [route, setRoute] = useState(routeFromHash);
  const requestedView = route.split("/")[0];
  const view = ["home", "run", ...NAV.map(([id]) => id)].includes(requestedView) ? requestedView : "home";
  const [runs, setRuns] = useState([]);
  const [stats, setStats] = useState(null);
  const [llm, setLlm] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [selected, setSelected] = useState(null);
  const [compareIds, setCompareIds] = useState(null);
  const [repairCandidate, setRepairCandidate] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const reload = useCallback(async () => {
    try {
      const [r, s, l] = await Promise.all([api("/runs?limit=300"), api("/stats"), api("/llm/status")]);
      setRuns(asRuns(r)); setStats(s); setLlm(l); setError("");
      setModelStatus(await api("/models/status").catch(() => null));
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { reload(); }, [reload]);
  useEffect(() => {
    const update = () => { setRoute(routeFromHash()); window.scrollTo({ top: 0 }); };
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  useEffect(() => {
    if (view !== "run" || !route.includes("/")) return;
    let current = true;
    setSelected(null);
    api("/runs/" + encodeURIComponent(route.split("/")[1])).then(r => { if (current) setSelected(r); }).catch(e => { if (current) setError(e.message); });
    return () => { current = false; };
  }, [route, view]);

  const go = target => { window.location.hash = target; window.scrollTo({ top: 0 }); };
  const openRun = async (id, target = "run") => {
    try { setSelected(await api("/runs/" + id)); go(target === "run" ? "run/" + id : target); }
    catch (e) { setError(e.message); }
  };
  const compare = (a, b) => { setCompareIds([a, b]); go("compare"); reload(); };
  const active = view === "run" ? "runs" : view;
  const trained = modelStatus?.available === true;

  if (view === "home") return <Home go={go} Brand={Brand} />;

  return <div className="workspace">
    <a className="skip-link" href="#workspace-content" onClick={e => { e.preventDefault(); document.getElementById("workspace-content")?.focus(); }}>Skip to content</a>
    <aside className="sidebar">
      <Brand onClick={() => go("home")} />
      <div className="workspace-label"><span className="workspace-avatar">B</span><div>Black Box workspace<small>Local environment</small></div><ChevronRight size={15} /></div>
      <span className="nav-caption">WORKSPACE</span>
      <nav className="workspace-nav" aria-label="Workspace">{NAV.map(([id, label, Icon]) => <button key={id} className={active === id ? "active" : ""} onClick={() => go(id)} aria-current={active === id ? "page" : undefined}><Icon size={18} /><span>{label}</span></button>)}</nav>
      <div className="sidebar-guide"><span className="guide-icon"><ShieldCheck size={20} /></span><h3>Make every step explainable.</h3><p>Inspect a failure. Test an alternative. Keep the evidence.</p><button onClick={() => go("break")}>Explore the failure lab <ArrowUpRight size={15} /></button></div>
      <div className="sidebar-bottom"><button onClick={() => go("home")}><ArrowLeft size={16} /> Product homepage</button><span><span className={"status-dot " + (error ? "offline" : loading ? "pending" : "")} />{error ? "Backend unavailable" : loading ? "Connecting…" : "Connected to local backend"}</span></div>
    </aside>
    <div className="workspace-body">
      <header className="workspace-header"><div className="breadcrumb">Workspace <ChevronRight size={14} /><strong>{view === "run" ? "Run investigation" : NAV.find(([id]) => id === view)?.[1]}</strong></div><div className="header-status"><span className="environment-tag"><span className={"status-dot " + (error ? "offline" : loading ? "pending" : "")} />{error ? "Disconnected" : loading ? "Connecting" : "Local workspace"}</span><button className="icon-button" onClick={reload} aria-label="Refresh workspace"><RefreshCw size={16} /></button><span className="user-avatar" aria-label="Black Box workspace">BB</span></div></header>
      <main id="workspace-content" className="workspace-content" tabIndex={-1}>
        {error && <div className="banner-error" role="alert"><AlertTriangle size={18} /><span>{error}. Check the local backend on port 8010.</span><button className="btn small" onClick={reload}><RefreshCw size={14} /> Retry</button></div>}
        {view === "runs" && <><div className="workspace-eyebrow"><span className="status-dot" /> EXECUTION INTELLIGENCE</div><Runs runs={runs} stats={stats} loading={loading} openRun={openRun} go={go} reload={reload} /><div className="workspace-footnote"><span><ShieldCheck size={15} />{loading ? "Checking diagnosis engine…" : trained ? "Trained diagnosis model available" : "Rule-based diagnosis · trained model not loaded"}</span><span>Agent: {llm ? llm.active_provider === "sandbox" ? "Offline sandbox" : llm.active_model || llm.active_provider : "Unavailable"}</span></div></>}
        {view === "run" && (selected ? <RunDetail run={selected} onBack={() => go("runs")} onReplay={candidate => { setRepairCandidate(candidate ? { ...candidate, runId: selected.run_id } : null); go("replay"); }} onBreak={() => go("break")} /> : <div className="empty"><h3>Loading investigation…</h3><button className="btn" onClick={() => go("runs")}>Return to overview</button></div>)}
        {view === "live" && <Live llm={llm} onDone={reload} openRun={openRun} />}
        {view === "break" && <BreakRun runs={runs} selected={selected} onDone={id => { reload(); openRun(id); }} />}
        {view === "replay" && <Replay runs={runs} selected={selected} setSelected={setSelected} initialCandidate={repairCandidate} onCompare={compare} onUpdate={reload} />}
        {view === "compare" && <Compare runs={runs} initialIds={compareIds} />}
        {view === "evaluation" && <Evaluation />}
      </main>
      <footer className="workspace-footer"><span>blackbox. <span>Every execution tells a story.</span></span><span>Record / Investigate / Replay</span></footer>
    </div>
  </div>;
}
