// Thesis.jsx — Causal hypothesis list with multi-hop chain rendering
// Shows each hypothesis as a chain of nodes connected by edge arrows,
// with per-edge lag, coefficient, and breach risk.

export default function Thesis({ live, activeDo }) {
  const hypotheses = live?.hypotheses || []
  const selected   = live?.selected_id || live?.selected_chain
  const tp         = activeDo?.tuned_params || {}

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:14 }}>

      {/* ── Header counts ── */}
      <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
        <MetricChip label="Hypotheses" val={hypotheses.length} />
        <MetricChip label="Multi-hop"  val={hypotheses.filter(h => (h.hop_count||h.chain_edges?.length||0) > 1).length} color="#7c3aed" />
        <MetricChip label="Selected"   val={hypotheses.filter(h => h.id===selected||h.chain===selected).length > 0 ? '1' : '—'} color="var(--thesis)" />
      </div>

      {/* ── Hypothesis list ── */}
      {hypotheses.length === 0 ? (
        <div className="empty">
          <div className="empty-icon">⊕</div>
          <p>No qualifying causal chains in this cycle</p>
        </div>
      ) : (
        <div style={{display:'flex',flexDirection:'column',gap:8}}>
          {hypotheses.map((h, i) => {
            const isSelected  = h.id === selected || live?.selected_chain === h.chain
            const isEV        = h.is_escape_valve
            const hops        = h.hop_count || h.chain_edges?.length || (h.chain ? (h.chain.split('→').length - 1) : 1)
            const breachRisk  = h.avg_breach_risk ?? null

            return (
              <div
                key={i}
                style={{
                  borderRadius:7,
                  border: `1px solid ${isSelected ? 'var(--thesis)' : isEV ? 'var(--escape)' : 'var(--border-light)'}`,
                  background: isSelected ? 'var(--thesis-light,#f0f4ff)' : isEV ? 'var(--escape-bg)' : 'var(--bg-panel)',
                  overflow:'hidden',
                  transition:'all 0.15s',
                  opacity: isSelected ? 1 : Math.max(0.55, 1 - i * 0.12),
                }}
              >
                {/* Row header */}
                <div style={{
                  padding:'7px 10px',
                  display:'flex',alignItems:'center',gap:8,
                  borderBottom: isSelected ? '1px solid var(--thesis)33' : '1px solid transparent',
                }}>
                  <span style={{
                    fontFamily:'var(--font-mono)',fontSize:'0.62rem',
                    color:'var(--text-secondary)',width:18,flexShrink:0,
                  }}>H{i+1}</span>

                  {/* Multi-hop badge */}
                  {hops > 1 && (
                    <span style={{
                      fontSize:'0.52rem',fontWeight:700,padding:'1px 5px',borderRadius:3,
                      background:'#7c3aed18',color:'#7c3aed',border:'1px solid #7c3aed30',
                      letterSpacing:'0.04em',
                    }}>{hops}-HOP</span>
                  )}

                  {/* Chain text */}
                  <span style={{
                    fontFamily:'var(--font-mono)',fontSize:'0.72rem',flex:1,
                    color: isSelected ? 'var(--thesis)' : 'var(--text-primary)',
                    overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',
                  }}>{h.chain}</span>

                  {/* Score */}
                  <span style={{
                    fontFamily:'var(--font-mono)',fontSize:'0.75rem',fontWeight:700,
                    color: isSelected ? 'var(--thesis)' : 'var(--text-secondary)',
                    flexShrink:0,
                  }}>{h.composite_score?.toFixed(4)}</span>

                  {/* Direction */}
                  <span style={{
                    fontFamily:'var(--font-mono)',fontSize:'0.65rem',fontWeight:700,flexShrink:0,
                    color: h.predicted_direction==='up' ? 'var(--ok)' : h.predicted_direction==='down' ? 'var(--bad)' : 'var(--text-secondary)',
                  }}>{h.predicted_direction?.toUpperCase() || '—'}</span>

                  {/* Badges */}
                  {isSelected && <span style={{fontSize:'0.58rem',fontWeight:700,padding:'2px 6px',borderRadius:3,background:'var(--thesis)',color:'#fff',flexShrink:0}}>SEL</span>}
                  {isEV       && <span style={{fontSize:'0.58rem',fontWeight:700,padding:'2px 6px',borderRadius:3,background:'var(--escape)',color:'#fff',flexShrink:0}}>ESC</span>}
                </div>

                {/* Expanded detail for selected hypothesis */}
                {isSelected && (
                  <div style={{padding:'10px 12px',display:'flex',flexDirection:'column',gap:10}}>

                    {/* Visual chain */}
                    <ChainVisual chain={h.chain} edges={h.chain_edges} />

                    {/* Score breakdown */}
                    <div style={{display:'flex',gap:12,flexWrap:'wrap'}}>
                      {tp.layer1_score != null && <ScorePair label="L1 Stability" val={tp.layer1_score?.toFixed(4)} color="var(--graph)" />}
                      {tp.layer2_score != null && <ScorePair label="L2 Trust"     val={tp.layer2_score?.toFixed(4)} color="var(--trust)" />}
                      {tp.composite_score != null && <ScorePair label="Composite" val={tp.composite_score?.toFixed(4)} color="var(--thesis)" />}
                      {breachRisk != null && (
                        <ScorePair
                          label="Breach Risk"
                          val={breachRisk.toFixed(4)}
                          color={breachRisk < 0.3 ? 'var(--ok)' : breachRisk < 0.6 ? 'var(--warn)' : 'var(--bad)'}
                        />
                      )}
                    </div>

                    {/* Breach risk bar */}
                    {breachRisk != null && (
                      <div>
                        <div style={{display:'flex',justifyContent:'space-between',marginBottom:3}}>
                          <span style={{fontSize:'0.60rem',color:'var(--text-secondary)'}}>ML2 Avg Breach Risk</span>
                          <span style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',
                            color: breachRisk < 0.3 ? 'var(--ok)' : breachRisk < 0.6 ? 'var(--warn)' : 'var(--bad)',
                          }}>{(breachRisk*100).toFixed(1)}%</span>
                        </div>
                        <div style={{height:4,background:'var(--bg-panel-2)',borderRadius:2,overflow:'hidden'}}>
                          <div style={{
                            height:'100%',borderRadius:2,
                            background: breachRisk < 0.3 ? 'var(--ok)' : breachRisk < 0.6 ? 'var(--warn)' : 'var(--bad)',
                            width:`${Math.min(100,breachRisk*100)}%`,transition:'width 0.5s',
                          }}/>
                        </div>
                        <div style={{fontSize:'0.57rem',color:'var(--text-secondary)',marginTop:3}}>
                          Proximity-based: 1/(1+|driver_z|) — near zero-crossing = high risk
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Visual causal chain: node → edge → node → edge → node ──
function ChainVisual({ chain, edges }) {
  if (!chain) return null

  // Parse chain string: "btc_return→(lag=1)→price_return" or "a→b→c"
  const parts  = chain.split(/\s*[→>]\s*/).map(p => p.trim()).filter(Boolean)
  const labels = parts.map(p => p.replace(/\(.*?\)/g, '').trim())

  if (labels.length < 2) {
    return (
      <div style={{fontFamily:'var(--font-mono)',fontSize:'0.70rem',color:'var(--thesis)',padding:'6px 0'}}>
        {chain}
      </div>
    )
  }

  return (
    <div style={{
      display:'flex',alignItems:'center',flexWrap:'wrap',gap:2,
      padding:'8px 10px',background:'var(--bg-panel-2)',borderRadius:5,
    }}>
      {labels.map((label, i) => {
        const edgeInfo = edges?.[i - 1]
        return (
          <div key={i} style={{display:'flex',alignItems:'center',gap:2}}>
            {i > 0 && (
              <div style={{display:'flex',flexDirection:'column',alignItems:'center',margin:'0 4px'}}>
                <span style={{fontFamily:'var(--font-mono)',fontSize:'0.55rem',color:'var(--text-secondary)',whiteSpace:'nowrap'}}>
                  {edgeInfo ? `lag=${edgeInfo.lag} c=${edgeInfo.coeff?.toFixed(2)}` : '→'}
                </span>
                <span style={{color:'var(--thesis)',fontSize:'0.9rem',lineHeight:1}}>→</span>
              </div>
            )}
            <div style={{
              padding:'3px 8px',borderRadius:4,
              background: i === labels.length - 1 ? 'var(--thesis)' : i === 0 ? '#7c3aed18' : 'var(--bg-panel)',
              border: i === labels.length - 1 ? 'none' : '1px solid var(--border-light)',
              color: i === labels.length - 1 ? '#fff' : 'var(--text-primary)',
              fontFamily:'var(--font-mono)',fontSize:'0.63rem',fontWeight: i===0||i===labels.length-1?700:500,
              maxWidth:120,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',
            }}
            title={label}
            >
              {label.length > 14 ? label.slice(0,6)+'…'+label.slice(-4) : label}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function ScorePair({ label, val, color }) {
  return (
    <div>
      <div style={{fontSize:'0.58rem',color:'var(--text-secondary)',marginBottom:2}}>{label}</div>
      <div style={{fontFamily:'var(--font-mono)',fontSize:'0.82rem',fontWeight:700,color: color||'var(--text-primary)'}}>{val}</div>
    </div>
  )
}

function MetricChip({ label, val, color }) {
  return (
    <div style={{
      padding:'4px 10px',borderRadius:5,background:'var(--bg-panel-2)',
      border:'1px solid var(--border-light)',display:'flex',gap:7,alignItems:'center',
    }}>
      <span style={{fontSize:'0.60rem',color:'var(--text-secondary)'}}>{label}</span>
      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.75rem',fontWeight:700,color: color||'var(--text-primary)'}}>{val}</span>
    </div>
  )
}
