export default function Trust({ trust }) {
  const scores = trust?.scores || []
  const history = trust?.history || []

  // Most recent trust update
  const latest = history[0]

  return (
    <div className="stage-card fade-in">
      <div className="stage-header">
        <span className="stage-tag tag-trust">TRUST</span>
        <span className="stage-title">Layer 2 Trust</span>
        <span className="stage-status">{scores.length} edges tracked</span>
      </div>
      <div className="stage-body">
        {/* Latest update */}
        {latest && (
          <div style={{ marginBottom:14, padding:'10px 12px', borderRadius:6,
            background: latest.outcome_correct ? 'var(--trust-light)' : '#fceaea',
            border: `1px solid ${latest.outcome_correct ? 'var(--trust)' : 'var(--readiness-bad)'}` }}>
            <div className="text-xs text-secondary" style={{ marginBottom:5, fontWeight:600, letterSpacing:'0.06em' }}>LAST TRUST UPDATE</div>
            <div className="mono" style={{ fontWeight:600, fontSize:'0.82rem', marginBottom:6 }}>
              {latest.edge_key?.split('|').slice(0,2).join(' → ')}
            </div>
            <div style={{ display:'flex', gap:16, alignItems:'center' }}>
              <span className="trust-delta-val" style={{ fontFamily:'var(--font-mono)', color: latest.outcome_correct ? 'var(--trust)' : 'var(--readiness-bad)' }}>
                trust after: {latest.trust_after?.toFixed(4)}
              </span>
              <span className="text-xs mono text-secondary">n={latest.n_observations}</span>
              <span className="text-xs mono" style={{ color: latest.outcome_correct ? 'var(--trust)' : 'var(--readiness-bad)' }}>
                {latest.outcome_correct ? '✓ direction held' : '✗ wrong direction'}
              </span>
            </div>
          </div>
        )}

        {/* All edge scores */}
        {scores.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">◌</div>
            <p>No trust scores yet — requires at least one closed trade</p>
          </div>
        ) : (
          <div style={{ display:'flex', flexDirection:'column', gap:6 }}>
            {scores.slice(0,8).map(s => (
              <div key={s.edge_key} style={{ display:'flex', alignItems:'center', gap:10 }}>
                <span className="mono text-xs" style={{ width:220, flexShrink:0, color:'var(--text-secondary)' }}>
                  {s.source} → {s.target} (lag {s.lag})
                </span>
                <div className="trust-bar-wrap">
                  <div className="trust-bar" style={{ width:`${Math.round(s.trust*100)}%` }} />
                </div>
                <span className="mono" style={{ fontSize:'0.78rem', fontWeight:600, width:44, textAlign:'right',
                  color: s.trust > 0.6 ? 'var(--trust)' : s.trust < 0.4 ? 'var(--readiness-bad)' : 'var(--text-primary)' }}>
                  {s.trust.toFixed(3)}
                </span>
                <span className="text-xs text-secondary mono">n={s.n_observations}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
