export default function ThesisSidebar({ live, activeDo, trust, state }) {
  const hypotheses = live?.hypotheses || []
  const tp = activeDo?.tuned_params || {}
  const vec = activeDo?.algo_health_vector || state?.active_do?.algo_health_vector || []
  const ev = state?.escape_valve || {}
  const scores = trust?.scores || []

  const verdict = Array.isArray(vec) && vec.length === 3
    ? vec[0] > 0.6 ? 'NORMAL' : vec[0] > 0.35 ? 'STRESSED' : 'DEGRADED'
    : null
  const verdictColor = verdict === 'NORMAL' ? 'var(--ok)' : verdict === 'STRESSED' ? 'var(--warn)' : verdict === 'DEGRADED' ? 'var(--bad)' : 'var(--text-secondary)'

  return (
    <div style={{
      width: 260, flexShrink: 0,
      background: 'var(--bg-panel)',
      borderLeft: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Agent Decision State */}
      <div style={{padding:'8px 12px',borderBottom:'1px solid var(--border-light)',flexShrink:0}}>
        <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.1em',color:'var(--text-secondary)',marginBottom:7}}>AGENT STATE</div>
        <div style={{
          display:'flex',alignItems:'center',gap:8,
          padding:'6px 10px',
          borderRadius:6,
          background: live?.hold_reason==='traded' ? 'var(--ok-bg)' : 'var(--bg-panel-2)',
          border:`1px solid ${live?.hold_reason==='traded' ? 'var(--ok)' : 'var(--border-light)'}`,
        }}>
          <span style={{
            fontFamily:'var(--font-mono)',fontSize:'1.1rem',fontWeight:800,
            color: live?.hold_reason==='traded' ? 'var(--ok)' : 'var(--text-secondary)',
          }}>
            {live?.hold_reason === 'traded' ? tp.predicted_direction==='up' ? 'BUY' : 'SELL' : 'HOLD'}
          </span>
          <div style={{flex:1}}>
            <div style={{fontSize:'0.60rem',color:'var(--text-secondary)',fontFamily:'var(--font-mono)'}}>
              {live?.hold_reason === 'traded' ? tp.chain_summary?.slice(0,28) : live?.hold_reason?.slice(0,28) || '—'}
            </div>
            {tp.composite_score != null && (
              <div style={{fontSize:'0.62rem',fontWeight:700,fontFamily:'var(--font-mono)',color:'var(--thesis)'}}>
                score: {tp.composite_score?.toFixed(4)}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Hypothesis ranking */}
      <div style={{padding:'8px 12px',borderBottom:'1px solid var(--border-light)',flexShrink:0}}>
        <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.1em',color:'var(--text-secondary)',marginBottom:7}}>
          HYPOTHESES ({hypotheses.length})
        </div>
        {hypotheses.length === 0 ? (
          <div style={{fontSize:'0.68rem',color:'var(--text-secondary)',fontStyle:'italic'}}>No qualifying chains this cycle</div>
        ) : (
          <div style={{display:'flex',flexDirection:'column',gap:4}}>
            {hypotheses.slice(0,5).map((h,i) => {
              const isSel = live?.selected_chain === h.chain || live?.selected_id === h.id
              return (
                <div key={i} style={{
                  display:'flex',alignItems:'center',gap:6,
                  padding:'5px 8px',borderRadius:4,
                  background: isSel ? 'var(--thesis-bg)' : 'var(--bg-panel-2)',
                  border:`1px solid ${isSel ? 'var(--thesis)' : 'transparent'}`,
                }}>
                  <span style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',color:'var(--text-secondary)',width:14}}>H{i+1}</span>
                  <span style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',flex:1,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{h.chain}</span>
                  <span style={{fontFamily:'var(--font-mono)',fontSize:'0.68rem',fontWeight:700,color:isSel?'var(--thesis)':'var(--text-secondary)'}}>
                    {h.composite_score?.toFixed(3)}
                  </span>
                  {isSel && <span style={{fontSize:'0.55rem',fontWeight:700,background:'var(--thesis)',color:'#fff',padding:'1px 4px',borderRadius:2}}>▶</span>}
                  {h.is_escape_valve && <span style={{fontSize:'0.55rem',fontWeight:700,background:'var(--escape)',color:'#fff',padding:'1px 4px',borderRadius:2}}>EV</span>}
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Selected chain detail */}
      {(tp.chain_summary || live?.selected_chain) && (
        <div style={{padding:'8px 12px',borderBottom:'1px solid var(--border-light)',flexShrink:0}}>
          <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.1em',color:'var(--thesis)',marginBottom:6}}>SELECTED CHAIN</div>
          <div style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',fontWeight:700,color:'var(--thesis)',marginBottom:6,lineHeight:1.4}}>
            {tp.chain_summary || live?.selected_chain}
          </div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:5}}>
            <Kv k="L1 stability" v={tp.layer1_score?.toFixed(4) ?? '—'} />
            <Kv k="L2 trust"     v={tp.layer2_score?.toFixed(4) ?? '—'} />
            <Kv k="Composite"    v={tp.composite_score?.toFixed(4) ?? '—'} />
            <Kv k="Direction"    v={tp.predicted_direction?.toUpperCase() ?? '—'} />
          </div>
        </div>
      )}

      {/* ML1 Readiness */}
      <div style={{padding:'8px 12px',borderBottom:'1px solid var(--border-light)',flexShrink:0}}>
        <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.1em',color:'var(--text-secondary)',marginBottom:7}}>
          ML1 READINESS
        </div>
        {Array.isArray(vec) && vec.length===3 ? (
          <>
            <Pb label="normal"   val={vec[0]} color="var(--ok)"   />
            <Pb label="stressed" val={vec[1]} color="var(--warn)" />
            <Pb label="degraded" val={vec[2]} color="var(--bad)"  />
            <div style={{marginTop:6}}>
              <span style={{
                fontFamily:'var(--font-mono)',fontSize:'0.65rem',fontWeight:700,
                color:verdictColor,padding:'2px 7px',
                background:verdictColor+'18',borderRadius:3,
              }}>VERDICT: {verdict}</span>
            </div>
          </>
        ) : <span style={{fontSize:'0.68rem',color:'var(--text-secondary)'}}>No health vector yet</span>}
      </div>

      {/* Escape Valve */}
      <div style={{padding:'8px 12px',borderBottom:'1px solid var(--border-light)',flexShrink:0}}>
        <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.1em',color:'var(--text-secondary)',marginBottom:5}}>ESCAPE VALVE</div>
        <div style={{
          padding:'5px 8px',borderRadius:4,
          background: ev.fired ? 'var(--escape-bg)' : ev.armed ? '#fff7ed88' : 'var(--bg-panel-2)',
          border:`1px solid ${ev.fired||ev.armed ? 'var(--escape)' : 'var(--border-light)'}`,
        }}>
          <div style={{fontFamily:'var(--font-mono)',fontSize:'0.68rem',fontWeight:ev.fired||ev.armed?700:400,color:ev.fired||ev.armed?'var(--escape)':'var(--text-secondary)'}}>
            {ev.fired ? '⚡ FIRED' : ev.armed ? `⚡ ARMED · ${ev.consecutive_holds}/${ev.threshold??25}` : `${ev.consecutive_holds??0}/${ev.threshold??25} holds · inactive`}
          </div>
        </div>
      </div>

      {/* Trust scores (top 3) */}
      <div style={{flex:1,overflow:'auto',padding:'8px 12px'}}>
        <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.1em',color:'var(--text-secondary)',marginBottom:7}}>TOP TRUST SCORES</div>
        {scores.length === 0
          ? <span style={{fontSize:'0.65rem',color:'var(--text-secondary)'}}>No scores yet</span>
          : scores.slice(0,5).map(s=>(
          <div key={s.edge_key} style={{marginBottom:5}}>
            <div style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:'var(--text-secondary)',marginBottom:2,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>
              {s.source} → {s.target}
            </div>
            <div style={{display:'flex',alignItems:'center',gap:5}}>
              <div style={{flex:1,height:4,background:'var(--bg-panel-2)',borderRadius:2,overflow:'hidden'}}>
                <div style={{height:'100%',background:'var(--trust)',borderRadius:2,width:`${s.trust*100}%`,transition:'width 0.5s'}}/>
              </div>
              <span style={{fontFamily:'var(--font-mono)',fontSize:'0.62rem',fontWeight:600,width:36,textAlign:'right',color:s.trust>0.6?'var(--ok)':s.trust<0.4?'var(--bad)':'var(--text-primary)'}}>{s.trust.toFixed(3)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function Kv({ k, v }) {
  return (
    <div>
      <div style={{fontSize:'0.58rem',color:'var(--text-secondary)'}}>{k}</div>
      <div style={{fontFamily:'var(--font-mono)',fontSize:'0.68rem',fontWeight:600}}>{v}</div>
    </div>
  )
}

function Pb({ label, val, color }) {
  const pct = Math.round((val||0)*100)
  return (
    <div style={{display:'flex',alignItems:'center',gap:6,marginBottom:4}}>
      <span style={{fontSize:'0.60rem',color:'var(--text-secondary)',width:52,flexShrink:0}}>{label}</span>
      <div style={{flex:1,height:5,background:'var(--bg-panel-2)',borderRadius:2,overflow:'hidden'}}>
        <div style={{height:'100%',background:color,borderRadius:2,width:`${pct}%`,transition:'width 0.5s'}}/>
      </div>
      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:'var(--text-secondary)',width:28,textAlign:'right'}}>{pct}%</span>
    </div>
  )
}
