export default function Ledger({ ledger }) {
  const decisions = ledger?.decisions || []

  return (
    <div className="stage-card fade-in">
      <div className="stage-header">
        <span className="stage-tag tag-ledger">LEDGER</span>
        <span className="stage-title">Decision Ledger</span>
        <span className="stage-status">{decisions.length} entries</span>
      </div>
      <div className="stage-body">
        {decisions.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">◫</div>
            <p>No committed decisions yet — agent is in warm-up</p>
          </div>
        ) : (
          <div style={{ overflowX:'auto' }}>
            <table className="ledger-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Time</th>
                  <th>Action</th>
                  <th>Chain</th>
                  <th>Status</th>
                  <th>Phase</th>
                  <th>PnL</th>
                  <th>Close Reason</th>
                </tr>
              </thead>
              <tbody>
                {decisions.map(d => {
                  const tp = d.tuned_params || {}
                  const outcome = d.outcome
                  const action = tp.predicted_direction === 'up' ? 'BUY' : tp.predicted_direction === 'down' ? 'SELL' : '—'
                  return (
                    <tr key={d.id}>
                      <td>{d.id?.slice(0,8)}</td>
                      <td>{d.timestamp_committed ? new Date(d.timestamp_committed).toLocaleTimeString() : '—'}</td>
                      <td style={{ color: action==='BUY' ? 'var(--readiness-ok)' : action==='SELL' ? 'var(--readiness-bad)' : 'inherit', fontWeight:600 }}>{action}</td>
                      <td style={{ maxWidth:160, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{tp.chain_summary || '—'}</td>
                      <td><span className={`status-badge ${d.status}`}>{d.status}</span></td>
                      <td>{d.phase}</td>
                      <td style={{ color: (outcome?.actual_pnl||0) >= 0 ? 'var(--readiness-ok)' : 'var(--readiness-bad)', fontWeight:600 }}>
                        {outcome ? `$${outcome.actual_pnl?.toFixed(2)}` : '—'}
                      </td>
                      <td style={{ color:'var(--text-secondary)' }}>{outcome?.close_reason || '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Active DO detail */}
        {decisions[0]?.status === 'ACTIVE' && (
          <div style={{ marginTop:12, padding:'10px 12px', background:'var(--feed-light)',
            borderRadius:6, border:'1px solid var(--feed)' }}>
            <div className="text-xs" style={{ fontWeight:600, color:'var(--feed)', letterSpacing:'0.06em', marginBottom:6 }}>ACTIVE TRADE</div>
            {(decisions[0].assumptions || []).map((a, i) => (
              <div key={i} className="mono text-xs" style={{ marginBottom:3, color:'var(--text-secondary)' }}>
                {i+1}. {a.name?.split('|').slice(1).join(' → ')} · breach_risk={a.breach_risk?.toFixed(3)}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
