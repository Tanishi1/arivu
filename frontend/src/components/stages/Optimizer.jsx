import { useEffect, useState } from 'react'
import React from 'react'

// ── friendly labels for the 5 optimizer dimensions ──
const PARAM_META = {
  k_runs:      { label:'k',      unit:'',    desc:'Roll window (≥15)',  group:'search' },
  min_runs:    { label:'min_r',  unit:'',    desc:'Min samples',        group:'search' },
  threshold:   { label:'thresh', unit:'',    desc:'Edge threshold',     group:'search' },
  tau_max:     { label:'τ_max',  unit:'bars',desc:'PCMCI max lag',      group:'discovery' },
  pcmci_alpha: { label:'α',      unit:'',    desc:'PCMCI p-cutoff',    group:'discovery' },
}

// Colour for search vs discovery params
const GROUP_COLOR = {
  search:    'var(--optimizer)',
  discovery: '#7c3aed',
}

function ParamRow({ k, v }) {
  const meta = PARAM_META[k]
  if (!meta) return null   // skip step_*/regime/etc
  const col = GROUP_COLOR[meta.group]
  return (
    <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:3,gap:4}}>
      <div style={{display:'flex',alignItems:'center',gap:5}}>
        <span style={{
          fontSize:'0.52rem',padding:'1px 4px',borderRadius:2,fontWeight:700,letterSpacing:'0.05em',
          background: col+'18', color: col, fontFamily:'var(--font-mono)',
        }}>{meta.group==='discovery'?'DISC':'SRCH'}</span>
        <span style={{fontSize:'0.62rem',color:'var(--text-secondary)'}}>{meta.label}</span>
        <span style={{fontSize:'0.56rem',color:'var(--text-tertiary)',display:'none'}}>{meta.desc}</span>
      </div>
      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',fontWeight:700,color:'var(--text-primary)'}}>
        {typeof v === 'number'
          ? (Number.isInteger(v) ? v : v.toFixed(4))
          : v}
        {meta.unit ? <span style={{fontSize:'0.55rem',color:'var(--text-secondary)',marginLeft:2}}>{meta.unit}</span> : null}
      </span>
    </div>
  )
}

const DISPLAY_KEYS = ['k_runs','min_runs','threshold','tau_max','pcmci_alpha']

export default function Optimizer({ optimizer, escapeValve = {}, state }) {
  const regimes = optimizer?.regimes || []
  const [restartFlash, setRestartFlash] = useState(false)

  const prevCands = React.useRef(null)
  useEffect(() => {
    if (!regimes.length) return
    const cur = JSON.stringify(regimes[0]?.population)
    if (prevCands.current && prevCands.current !== cur) {
      setRestartFlash(true)
      setTimeout(() => setRestartFlash(false), 1500)
    }
    prevCands.current = cur
  }, [regimes])

  const ev = escapeValve
  const phase = state?.phase || 'bootstrap'
  const regime = state?.regime || regimes[0]?.regime || '—'

  return (
    <div style={{display:'flex',flexDirection:'column',gap:14}}>

      {/* ── 5-dimension banner ── */}
      <div style={{
        padding:'9px 13px',borderRadius:6,
        background:'linear-gradient(135deg,#4f46e508,#7c3aed08)',
        border:'1px solid var(--border-light)',
        display:'flex',alignItems:'center',gap:10,flexWrap:'wrap',
      }}>
        <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.08em',color:'var(--text-secondary)'}}>
          5-DIM HILL CLIMB
        </div>
        {[
          {label:'k_runs',     color:'var(--optimizer)', desc:'window'},
          {label:'min_runs',   color:'var(--optimizer)', desc:'min obs'},
          {label:'threshold',  color:'var(--optimizer)', desc:'edge cutoff'},
          {label:'τ_max',      color:'#7c3aed',          desc:'PCMCI lag'},
          {label:'α-cutoff',   color:'#7c3aed',          desc:'p-value'},
        ].map(d => (
          <div key={d.label} style={{
            display:'flex',alignItems:'center',gap:4,
            padding:'2px 8px',borderRadius:4,
            background: d.color+'14', border:`1px solid ${d.color}30`,
          }}>
            <span style={{fontFamily:'var(--font-mono)',fontSize:'0.63rem',fontWeight:700,color:d.color}}>{d.label}</span>
            <span style={{fontSize:'0.56rem',color:'var(--text-secondary)'}}>{d.desc}</span>
          </div>
        ))}
        <div style={{marginLeft:'auto',fontSize:'0.58rem',color:'var(--text-secondary)'}}>
          <span style={{color:'var(--optimizer)',fontWeight:700}}>●</span> search params &nbsp;
          <span style={{color:'#7c3aed',fontWeight:700}}>●</span> discovery params
        </div>
      </div>

      {/* ── Escape Valve ── */}
      <div>
        <div style={{fontSize:'0.62rem',fontWeight:700,letterSpacing:'0.08em',color:'var(--text-secondary)',marginBottom:8}}>
          ESCAPE VALVE
        </div>
        <div style={{
          padding:'10px 12px',borderRadius:6,
          background: ev.fired ? 'var(--escape-bg)' : ev.armed ? '#fff7ed' : 'var(--bg-panel)',
          border:`1px solid ${ev.fired||ev.armed ? 'var(--escape)' : 'var(--border-light)'}`,
          display:'flex',alignItems:'center',gap:12,
        }}>
          <div style={{fontSize:'1.4rem',flexShrink:0}}>
            {ev.fired ? '⚡' : ev.armed ? '⚠️' : '○'}
          </div>
          <div style={{flex:1}}>
            <div style={{fontFamily:'var(--font-mono)',fontSize:'0.78rem',fontWeight:700,color:ev.fired||ev.armed?'var(--escape)':'var(--text-secondary)'}}>
              {ev.fired
                ? 'ESCAPE VALVE FIRED — forcing non-graph hypothesis'
                : ev.armed
                ? `ARMED — ${ev.consecutive_holds}/25 consecutive holds`
                : `Inactive — ${ev.consecutive_holds??0}/25 holds`}
            </div>
            <div style={{marginTop:5}}>
              <div style={{height:5,background:'var(--bg-panel-2)',borderRadius:3,overflow:'hidden',width:'100%'}}>
                <div style={{
                  height:'100%',borderRadius:3,
                  background: ev.fired ? 'var(--escape)' : ev.armed ? 'var(--warn)' : 'var(--border)',
                  width:`${Math.min(100,((ev.consecutive_holds??0)/25)*100)}%`,
                  transition:'width 0.5s',
                }}/>
              </div>
              <div style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:'var(--text-secondary)',marginTop:3}}>
                Fires at 25 consecutive HOLDs → forces an unconstrained hypothesis from graph neighbourhood
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Current regime ── */}
      <div style={{display:'flex',gap:8}}>
        <div className="metric-cell" style={{flex:1}}>
          <div className="metric-label">Current Regime</div>
          <div className="metric-val" style={{color:'var(--optimizer)'}}>{regime}</div>
        </div>
        <div className="metric-cell" style={{flex:1}}>
          <div className="metric-label">Phase</div>
          <div className={`metric-val ${phase==='trained'?'ok':'warn'}`}>{phase}</div>
        </div>
        <div className="metric-cell" style={{flex:1}}>
          <div className="metric-label">Holds streak</div>
          <div className={`metric-val ${(ev.consecutive_holds||0)>18?'bad':(ev.consecutive_holds||0)>12?'warn':''}`}>
            {ev.consecutive_holds??0}
          </div>
        </div>
      </div>

      {/* ── Hill climbing population per regime ── */}
      {regimes.length === 0 ? (
        <div className="empty">
          <div className="empty-icon">◈</div>
          <p>No optimizer state yet — requires completed decision cycles with outcomes</p>
        </div>
      ) : regimes.map(r => (
        <div key={r.regime} style={{border:'1px solid var(--border-light)',borderRadius:8,overflow:'hidden'}}>
          <div style={{
            padding:'7px 12px',
            background: r.regime===regime ? 'var(--optimizer-bg)' : 'var(--bg-panel)',
            borderBottom:'1px solid var(--border-light)',
            display:'flex',alignItems:'center',gap:10,
          }}>
            <span style={{fontFamily:'var(--font-mono)',fontSize:'0.62rem',fontWeight:700,letterSpacing:'0.06em',color:'var(--optimizer)'}}>
              REGIME: {r.regime?.toUpperCase()}
            </span>
            {r.regime===regime && (
              <span style={{fontSize:'0.58rem',background:'var(--optimizer)',color:'#fff',padding:'1px 6px',borderRadius:3,fontWeight:700}}>
                ACTIVE
              </span>
            )}
            <span style={{marginLeft:'auto',fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:'var(--text-secondary)'}}>
              {r.history_count} steps · avg_score={r.recent_avg_score}
            </span>
          </div>

          <div style={{padding:'10px 12px'}}>
            {/* Discovery params highlight strip */}
            {r.population?.[0] && (
              <div style={{
                marginBottom:10,padding:'7px 10px',borderRadius:5,
                background:'linear-gradient(90deg,#7c3aed08,transparent)',
                border:'1px solid #7c3aed22',
                display:'flex',gap:16,alignItems:'center',
              }}>
                <span style={{fontSize:'0.58rem',fontWeight:700,letterSpacing:'0.07em',color:'#7c3aed'}}>DISCOVERY</span>
                <span style={{fontFamily:'var(--font-mono)',fontSize:'0.68rem',color:'#7c3aed',fontWeight:700}}>
                  τ={r.population[0].tau_max ?? '—'} bars
                </span>
                <span style={{fontFamily:'var(--font-mono)',fontSize:'0.68rem',color:'#7c3aed',fontWeight:700}}>
                  α={r.population[0].pcmci_alpha != null ? r.population[0].pcmci_alpha.toFixed(3) : '—'}
                </span>
                <span style={{fontSize:'0.58rem',color:'var(--text-secondary)',marginLeft:'auto'}}>
                  best config · being hill-climbed
                </span>
              </div>
            )}

            {/* Candidate population */}
            <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.08em',color:'var(--text-secondary)',marginBottom:8}}>
              CANDIDATE POPULATION (5-dim CEM hill-climb)
            </div>

            <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:8,marginBottom:10,
              animation: restartFlash && r.regime===regime ? 'flash-restart 1.5s ease' : 'none',
            }}>
              {(r.population || []).slice(0,3).map((c,i) => (
                <div key={i} style={{
                  border:`1px solid ${i===0?'var(--optimizer)':'var(--border-light)'}`,
                  borderRadius:6,padding:'8px 10px',
                  background: i===0 ? 'var(--optimizer-bg)' : 'var(--bg-panel)',
                }}>
                  <div style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',fontWeight:700,color:i===0?'var(--optimizer)':'var(--text-secondary)',marginBottom:6}}>
                    {i===0?'● BEST':'  CAND '} {String.fromCharCode(65+i)}
                  </div>
                  {/* Search params block */}
                  <div style={{marginBottom:5,paddingBottom:5,borderBottom:'1px dashed var(--border-light)'}}>
                    {['k_runs','min_runs','threshold'].map(k =>
                      c[k] != null ? <ParamRow key={k} k={k} v={c[k]} /> : null
                    )}
                  </div>
                  {/* Discovery params block */}
                  <div>
                    {['tau_max','pcmci_alpha'].map(k =>
                      c[k] != null ? <ParamRow key={k} k={k} v={c[k]} /> : null
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Stagnation bar */}
            <div style={{marginBottom:6}}>
              <div style={{display:'flex',justifyContent:'space-between',marginBottom:3}}>
                <span style={{fontSize:'0.60rem',color:'var(--text-secondary)'}}>Stagnation counter</span>
                <span style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:r.stagnation_count>3?'var(--warn)':'var(--text-secondary)'}}>
                  {r.stagnation_count||0} / 5 → random restart
                </span>
              </div>
              <div style={{height:5,background:'var(--bg-panel-2)',borderRadius:3,overflow:'hidden'}}>
                <div style={{
                  height:'100%',borderRadius:3,
                  background:r.stagnation_count>=4?'var(--bad)':r.stagnation_count>=2?'var(--warn)':'var(--ok)',
                  width:`${Math.min(100,((r.stagnation_count||0)/5)*100)}%`,
                  transition:'width 0.5s',
                }}/>
              </div>
              <div style={{fontSize:'0.58rem',color:'var(--text-secondary)',marginTop:3}}>
                When stagnation hits 5 → random restart: new population sampled from prior · ensures exploration
              </div>
            </div>

            {/* Recent scores sparkline */}
            {r.history_scores?.length > 0 && (
              <div style={{marginTop:8}}>
                <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.06em',color:'var(--text-secondary)',marginBottom:5}}>
                  SCORE HISTORY
                </div>
                <div style={{display:'flex',alignItems:'flex-end',gap:2,height:30}}>
                  {r.history_scores.slice(-20).map((s,i) => {
                    const max = Math.max(...r.history_scores.slice(-20))
                    const h = max > 0 ? Math.max(3,(s/max)*28) : 3
                    return (
                      <div key={i} style={{flex:1,height:h,background:'var(--optimizer)',opacity:0.4+i*0.03,borderRadius:1,transition:'height 0.3s'}}/>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      ))}

      {/* ── Algorithm explanation ── */}
      <div style={{padding:'9px 12px',background:'var(--optimizer-bg)',borderRadius:5,border:'1px solid var(--optimizer)',fontSize:'0.62rem',lineHeight:1.7}}>
        <strong style={{color:'var(--optimizer)'}}>Meta Optimizer (5-dim Hill-Climb):</strong>
        {' '}Hill-climbs 5 parameters per market regime.
        <br/>
        <span style={{color:'var(--optimizer)'}}>Search (3):</span> k_runs (floor=15), min_runs, threshold — control edge validation strictness.
        <br/>
        <span style={{color:'#7c3aed'}}>Discovery (2):</span> τ_max (PCMCI lag window, up to 12 bars), α (p-value cutoff) — control causal timescale found.
        <br/>
        <span style={{color:'var(--text-secondary)'}}>Trade gates:</span> fee floor (0.15% min return) · same-graph breach block · top-30 hypothesis pool.
        After 5 steps without improvement → <strong>random restart</strong>: full population resampled from prior.
        Separate populations for trending, volatile, and quiet regimes.
      </div>
    </div>
  )
}
