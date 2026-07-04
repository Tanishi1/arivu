// GraphMeta.jsx — shown in right panel when GRAPH stage is active
// The actual graph is on the left. This shows metadata + edge table.
export default function GraphMeta({ graph, live }) {
  if (!graph) return (
    <div className="empty">
      <div className="empty-icon">◇</div>
      <p>No graph yet — waiting for 40+ bars for Granger, 80+ for PCMCI</p>
    </div>
  )

  const validated = graph.edges?.filter(e => e.validated) || []
  const toTarget  = graph.edges?.filter(e => e.target === 'price_return') || []
  const escapeEd  = graph.edges?.filter(e => e.is_escape_valve) || []

  return (
    <div style={{display:'flex',flexDirection:'column',gap:14}}>
      {/* Summary */}
      <div className="metric-grid">
        <M label="Version"   val={graph.version_id?.slice(0,10) || '—'} />
        <M label="Algorithm" val={graph.algorithm || '—'} />
        <M label="Total edges"    val={graph.edge_count ?? '—'} />
        <M label="Validated"      val={validated.length} />
        <M label="→ price_return" val={toTarget.length} />
        <M label="Escape valve"   val={escapeEd.length} cls={escapeEd.length>0?'warn':''} />
        <M label="Bars used"      val={graph.n_bars_used ?? '—'} />
        <M label="Graph nodes"    val={graph.variables?.length ?? '—'} />
      </div>

      {/* Edges to price_return — the most important ones */}
      <div>
        <div style={{fontSize:'0.62rem',fontWeight:700,letterSpacing:'0.08em',color:'var(--text-secondary)',marginBottom:8}}>
          DIRECT CAUSAL PATHS TO price_return
        </div>
        {toTarget.length === 0 ? (
          <div style={{fontSize:'0.70rem',color:'var(--text-secondary)',fontStyle:'italic'}}>No direct edges to price_return yet</div>
        ) : (
          <table className="ltable" style={{width:'100%'}}>
            <thead>
              <tr>
                <th>Source</th>
                <th>Lag</th>
                <th>p-val</th>
                <th>Strength</th>
                <th>Valid</th>
              </tr>
            </thead>
            <tbody>
              {toTarget.sort((a,b)=>(b.strength||0)-(a.strength||0)).map((e,i) => (
                <tr key={i}>
                  <td style={{fontFamily:'var(--font-mono)',fontSize:'0.68rem'}}>{e.source}</td>
                  <td style={{fontFamily:'var(--font-mono)'}}>{e.lag ?? '—'}</td>
                  <td style={{fontFamily:'var(--font-mono)',color:(e.p_value||1)<0.05?'var(--ok)':'var(--text-secondary)'}}>
                    {e.p_value?.toFixed(4) ?? '—'}
                  </td>
                  <td style={{fontFamily:'var(--font-mono)',fontWeight:600}}>{e.strength?.toFixed(4) ?? '—'}</td>
                  <td>{e.validated ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Escape valve edges */}
      {escapeEd.length > 0 && (
        <div>
          <div style={{fontSize:'0.62rem',fontWeight:700,letterSpacing:'0.08em',color:'var(--escape)',marginBottom:6}}>
            ⚡ ESCAPE VALVE EDGES (not in discovered graph)
          </div>
          {escapeEd.map((e,i) => (
            <div key={i} style={{fontFamily:'var(--font-mono)',fontSize:'0.68rem',color:'var(--escape)',marginBottom:3}}>
              {e.source} → {e.target} (lag {e.lag})
            </div>
          ))}
        </div>
      )}

      {/* Algorithm note */}
      <div style={{padding:'8px 10px',background:'var(--graph-bg)',borderRadius:5,border:'1px solid var(--graph)',fontSize:'0.62rem',color:'var(--text-secondary)',lineHeight:1.6}}>
        <strong style={{color:'var(--graph)'}}>PCMCI+</strong> (Peter & Clark Momentary Conditional Independence) discovers causal links
        by testing conditional independence at each lag. Only validated edges (p &lt; threshold) appear in hypothesis generation.
        Granger fallback used below 80 bars.
      </div>
    </div>
  )
}

function M({ label, val, cls }) {
  return (
    <div className="metric-cell">
      <div className="metric-label">{label}</div>
      <div className={`metric-val ${cls||''}`}>{val}</div>
    </div>
  )
}
