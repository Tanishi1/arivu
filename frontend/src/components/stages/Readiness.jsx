export default function Readiness({ activeDo, live }) {
  const vec = activeDo?.algo_health_vector || []
  const assumptions = activeDo?.assumptions || []

  const pNormal   = Array.isArray(vec) ? vec[0] : null
  const pStressed = Array.isArray(vec) ? vec[1] : null
  const pDegraded = Array.isArray(vec) ? vec[2] : null

  const verdict = pNormal == null ? null
    : pNormal > 0.6 ? 'NORMAL'
    : pNormal > 0.35 ? 'STRESSED'
    : 'DEGRADED'

  const verdictColor = verdict === 'NORMAL' ? 'var(--readiness-ok)'
    : verdict === 'STRESSED' ? 'var(--readiness-warn)'
    : 'var(--readiness-bad)'

  // ML2 breach risk — average across assumptions
  const avgBreachRisk = assumptions.length > 0
    ? assumptions.reduce((s, a) => s + (a.breach_risk || 0), 0) / assumptions.length
    : null

  const ml2Color = avgBreachRisk == null ? 'var(--text-secondary)'
    : avgBreachRisk < 0.3 ? 'var(--readiness-ok)'
    : avgBreachRisk < 0.6 ? 'var(--readiness-warn)'
    : 'var(--readiness-bad)'

  const gateOpen = verdict === 'NORMAL' && (avgBreachRisk == null || avgBreachRisk < 0.6)

  return (
    <div className="stage-card fade-in">
      <div className="stage-header">
        <span className="stage-tag tag-ready">READINESS</span>
        <span className="stage-title">Agent Readiness</span>
        <span className="stage-status" style={{ color: verdictColor }}>
          {verdict || 'PENDING'}
        </span>
      </div>
      <div className="stage-body">
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14 }}>

          {/* ML1 */}
          <div>
            <div className="text-xs text-secondary" style={{ marginBottom:8, letterSpacing:'0.06em', fontWeight:600 }}>ML1 HEALTH VECTOR</div>
            {vec.length === 3 ? (
              <>
                <ProbRow label="normal"   val={pNormal}   color="var(--readiness-ok)" />
                <ProbRow label="stressed" val={pStressed} color="var(--readiness-warn)" />
                <ProbRow label="degraded" val={pDegraded} color="var(--readiness-bad)" />
                <div style={{ marginTop:8 }}>
                  <span style={{
                    fontFamily:'var(--font-mono)', fontSize:'0.78rem', fontWeight:700,
                    color: verdictColor, padding:'2px 8px',
                    background: verdictColor + '18', borderRadius:4,
                  }}>VERDICT: {verdict}</span>
                </div>
              </>
            ) : (
              <p className="text-xs text-secondary">No ML1 data yet</p>
            )}
          </div>

          {/* ML2 */}
          <div>
            <div className="text-xs text-secondary" style={{ marginBottom:8, letterSpacing:'0.06em', fontWeight:600 }}>ML2 BREACH RISK</div>
            {avgBreachRisk != null ? (
              <>
                <div style={{ fontSize:'2rem', fontFamily:'var(--font-mono)', fontWeight:700, color: ml2Color, lineHeight:1 }}>
                  {avgBreachRisk.toFixed(3)}
                </div>
                <div className="text-xs text-secondary" style={{ marginTop:5 }}>
                  {avgBreachRisk < 0.3 ? 'acceptable' : avgBreachRisk < 0.6 ? 'elevated' : 'HIGH RISK'}
                </div>
                <div style={{ marginTop:8 }}>
                  {assumptions.slice(0,2).map((a,i) => (
                    <div key={i} className="text-xs mono text-secondary" style={{ marginBottom:2 }}>
                      {a.name?.split('|').slice(1,3).join('→')} · {a.breach_risk?.toFixed(3)}
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-xs text-secondary">No active trade</p>
            )}
          </div>
        </div>

        {/* Gate visual */}
        <div style={{ marginTop: 14, padding:'8px 12px', borderRadius:6,
          background: gateOpen ? '#e8f5f3' : '#fdf6ee',
          border: `1px solid ${gateOpen ? 'var(--readiness-ok)' : 'var(--readiness-warn)'}`,
          display:'flex', alignItems:'center', gap:10 }}>
          <span style={{ fontSize:'1.2rem' }}>{gateOpen ? '✓' : '⊘'}</span>
          <span className="mono text-xs" style={{ fontWeight:600, color: gateOpen ? 'var(--readiness-ok)' : 'var(--readiness-warn)' }}>
            GATE {gateOpen ? 'OPEN — agent may act' : 'NARROWED — awaiting better conditions'}
          </span>
        </div>
      </div>
    </div>
  )
}

function ProbRow({ label, val, color }) {
  const pct = Math.round((val || 0) * 100)
  return (
    <div className="prob-strip">
      <span className="prob-label">{label}</span>
      <div className="prob-bar-wrap">
        <div className="prob-bar" style={{ width:`${pct}%`, background: color }} />
      </div>
      <span className="prob-value mono">{pct}%</span>
    </div>
  )
}
