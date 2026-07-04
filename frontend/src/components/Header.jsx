import { NavLink } from 'react-router-dom'

const STAGES = [
  { id: 'FEED',      color: 'var(--feed)' },
  { id: 'BARS',      color: 'var(--bars)' },
  { id: 'GRAPH',     color: 'var(--graph)' },
  { id: 'THESIS',    color: 'var(--thesis)' },
  { id: 'READINESS', color: 'var(--readiness-ok)' },
  { id: 'TWIN',      color: 'var(--twin)' },
  { id: 'LEDGER',    color: 'var(--ledger)' },
  { id: 'TRUST',     color: 'var(--trust)' },
  { id: 'OPTIMIZER', color: 'var(--optimizer)' },
]

function stageValue(id, state, graph) {
  if (!state) return '—'
  const live = state.live || {}
  switch (id) {
    case 'FEED':      return live.price ? `$${Number(live.price).toFixed(2)}` : '—'
    case 'BARS':      return state.bars_ready ? `${state.bars_ready}/200` : '—'
    case 'GRAPH':     return state.graph_version ? `v${state.graph_version}` : '—'
    case 'THESIS':    return live.selected_chain || (live.n_hypotheses ? `${live.n_hypotheses} hyp` : '—')
    case 'READINESS': {
      const vec = state.active_do?.algo_health_vector
      if (Array.isArray(vec)) return vec[0] > 0.6 ? 'NORMAL' : 'STRESSED'
      return '—'
    }
    case 'TWIN':      return state.active_do ? 'running' : '—'
    case 'LEDGER':    return state.active_do?.status || state.latest_closed_do?.status || '—'
    case 'TRUST':     return '—'
    case 'OPTIMIZER': return state.regime || '—'
    default:          return '—'
  }
}

export default function Header({ state, graph, activeStage, onStageClick }) {
  const live = state?.live || {}
  const phase = state?.phase || 'bootstrap'
  const ev = state?.escape_valve || {}
  const graphVersion = state?.graph_version
  const feedOk = live.price && Date.now() - new Date(live.ts).getTime() < 30000

  return (
    <>
      {/* Top nav */}
      <nav className="top-nav">
        <span className="nav-brand">ARIVU</span>
        <div className="nav-links">
          <NavLink to="/"          className={({isActive}) => 'nav-link' + (isActive ? ' active' : '')}>Cockpit</NavLink>
          <NavLink to="/compare"   className={({isActive}) => 'nav-link' + (isActive ? ' active' : '')}>Comparison</NavLink>
          <NavLink to="/ledger"    className={({isActive}) => 'nav-link' + (isActive ? ' active' : '')}>Ledger</NavLink>
          <NavLink to="/research"  className={({isActive}) => 'nav-link' + (isActive ? ' active' : '')}>Research</NavLink>
        </div>
        <div className="nav-spacer" />
        <div className="nav-status">
          <span className={`pulse-dot ${feedOk ? '' : 'stale'}`} />
          {feedOk ? 'LIVE' : 'STALE'}
        </div>
      </nav>

      {/* Pipeline ribbon */}
      <div className="pipeline-ribbon">
        {STAGES.map((s, i) => (
          <div key={s.id} style={{ display: 'flex', alignItems: 'center' }}>
            <div
              className={`stage-pill ${activeStage === i ? 'active' : ''}`}
              onClick={() => onStageClick?.(i)}
              style={{ borderColor: activeStage === i ? s.color : 'transparent' }}
            >
              <span style={{ color: s.color, fontWeight: 700, fontSize: '0.70rem', letterSpacing: '0.08em' }}>{s.id}</span>
              <span className="stage-value">{stageValue(s.id, state, graph)}</span>
            </div>
            {i < STAGES.length - 1 && <span className="stage-arrow">›</span>}
          </div>
        ))}
      </div>

      {/* Context chips */}
      <div className="context-chips">
        {ev.fired
          ? <span className="chip escape">⚡ ESCAPE VALVE · FIRED · size=1%</span>
          : ev.armed
          ? <span className="chip escape">⚡ ESCAPE VALVE · {ev.consecutive_holds}/30 HOLDS · ARMED</span>
          : <span className="chip">ESCAPE VALVE · {ev.consecutive_holds ?? 0}/30 HOLDS · INACTIVE</span>}

        <span className={`chip ${phase === 'trained' ? 'ok' : 'warn'}`}>
          AGENT PHASE · {phase.toUpperCase()}
        </span>

        {graphVersion && (
          <span className="chip regime">GRAPH · v{graphVersion} · {graph?.algorithm?.toUpperCase() || '—'}</span>
        )}

        {live.hold_reason && (
          <span className="chip state mono">
            {live.hold_reason === 'traded'
              ? `ACTION · BUY · ${live.selected_chain || '—'}`
              : `HOLD · ${live.hold_reason?.toUpperCase()}`}
          </span>
        )}
      </div>
    </>
  )
}
