import { useState, useEffect, useRef } from 'react'
import { NavLink } from 'react-router-dom'

const NAV = [
  { to: '/',         label: 'Cockpit' },
  { to: '/compare',  label: 'Comparison' },
  { to: '/ledger',   label: 'Ledger' },
  { to: '/research', label: 'Research' },
]

export default function ResearchPage() {
  const [trust, setTrust]         = useState(null)
  const [optimizer, setOptimizer] = useState(null)
  const [ledger, setLedger]       = useState(null)
  const [allArms, setAllArms]     = useState({})

  useEffect(() => {
    const load = async () => {
      const [t, o, l, ca, ps, pc] = await Promise.all([
        fetch('/api/trust').then(r => r.json()).catch(() => null),
        fetch('/api/optimizer').then(r => r.json()).catch(() => null),
        fetch('/api/ledger?limit=100&strategy=CausalAgent').then(r => r.json()).catch(() => null),
        fetch('/api/ledger?limit=50&strategy=CausalAgent').then(r => r.json()).catch(() => null),
        fetch('/api/ledger?limit=50&strategy=ppo_standard').then(r => r.json()).catch(() => null),
        fetch('/api/ledger?limit=50&strategy=ppo_causal_feature').then(r => r.json()).catch(() => null),
      ])
      setTrust(t)
      setOptimizer(o)
      setLedger(l)
      setAllArms({ CausalAgent: ca, ppo_standard: ps, ppo_causal_feature: pc })
    }
    load()
  }, [])

  const decisions = ledger?.decisions || []
  const scores    = trust?.scores || []
  const regimes   = optimizer?.regimes || []

  const closedTrades = decisions.filter(d => d.outcome)
  const wins         = closedTrades.filter(d => (d.outcome?.actual_pnl || 0) > 0)
  const winRate      = closedTrades.length > 0 ? (wins.length / closedTrades.length * 100).toFixed(1) : null
  const totalPnl     = closedTrades.reduce((s, d) => s + (d.outcome?.actual_pnl || 0), 0)
  const avgPnl       = closedTrades.length > 0 ? totalPnl / closedTrades.length : null
  const bestTrade    = closedTrades.reduce((b, d) => (d.outcome?.actual_pnl||0) > (b?.outcome?.actual_pnl||0) ? d : b, null)
  const worstTrade   = closedTrades.reduce((b, d) => (d.outcome?.actual_pnl||0) < (b?.outcome?.actual_pnl||0) ? d : b, null)

  // Cross-arm stats
  const armStats = Object.entries(allArms).map(([arm, data]) => {
    const ds = data?.decisions || []
    const cl = ds.filter(d => d.outcome)
    const ws = cl.filter(d => (d.outcome?.actual_pnl||0) > 0)
    const pnl = cl.reduce((s,d) => s+(d.outcome?.actual_pnl||0),0)
    return { arm, total: ds.length, closed: cl.length, wins: ws.length, pnl,
             winRate: cl.length > 0 ? (ws.length/cl.length*100) : null }
  })

  const ARM_COLORS = { CausalAgent:'#2563eb', ppo_standard:'#7c3aed', ppo_causal_feature:'#9333ea' }
  const ARM_LABELS = { CausalAgent:'Causal', ppo_standard:'PPO-S', ppo_causal_feature:'PPO-C' }

  return (
    <div className="shell" style={{background:'var(--bg-base)'}}>
      <nav className="top-nav">
        <span className="nav-brand">ARIVU</span>
        <div className="nav-tabs">
          {NAV.map(({to,label}) => (
            <NavLink key={to} to={to} end={to==='/'} className={({isActive})=>`nav-tab${isActive?' active':''}`}>{label}</NavLink>
          ))}
        </div>
        <div className="nav-right">
          <span style={{fontFamily:'var(--font-mono)',fontSize:'0.64rem',color:'var(--text-secondary)'}}>
            RESEARCH METRICS
          </span>
        </div>
      </nav>

      <div style={{flex:1,overflowY:'auto',padding:'16px 20px 24px'}}>

        {/* ── Header ── */}
        <div style={{display:'flex',alignItems:'baseline',gap:10,marginBottom:18}}>
          <h2 style={{fontSize:'0.95rem',fontWeight:700,letterSpacing:'0.03em'}}>Research Dashboard</h2>
          <span style={{fontSize:'0.68rem',color:'var(--text-secondary)',fontFamily:'var(--font-mono)'}}>
            Triple-arm causal experiment · CausalAgent vs PPO baselines
          </span>
        </div>

        {/* ── Row 1: KPI cards ── */}
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(140px,1fr))',gap:10,marginBottom:16}}>
          <KpiCard label="CausalAgent Trades" val={decisions.length} sub="total decisions" />
          <KpiCard label="Closed Trades" val={closedTrades.length} sub="with outcome" />
          <KpiCard label="Win Rate" val={winRate != null ? `${winRate}%` : '—'} sub="of closed trades"
            color={winRate > 50 ? 'var(--ok)' : winRate < 40 ? 'var(--bad)' : null} />
          <KpiCard label="Total PnL" val={`$${totalPnl.toFixed(2)}`} sub="causal arm"
            color={totalPnl >= 0 ? 'var(--ok)' : 'var(--bad)'} />
          <KpiCard label="Avg PnL/Trade" val={avgPnl != null ? `$${avgPnl.toFixed(2)}` : '—'} sub="per closed trade"
            color={avgPnl != null ? (avgPnl >= 0 ? 'var(--ok)' : 'var(--bad)') : null} />
          <KpiCard label="Trust Scores" val={scores.length} sub="Bayesian edges tracked" />
        </div>

        {/* ── Row 2: Main grid ── */}
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:14,marginBottom:14}}>

          {/* Cross-arm comparison */}
          <ResearchCard title="Triple-Arm PnL Comparison" accent="#2563eb">
            {armStats.every(a => a.closed === 0) ? (
              <EmptyState icon="⚖" text="No closed trades yet across any arm. Awaiting bar accumulation and first trade execution." />
            ) : (
              <div style={{display:'flex',flexDirection:'column',gap:8}}>
                {armStats.map(a => (
                  <div key={a.arm} style={{display:'flex',alignItems:'center',gap:10}}>
                    <span style={{
                      background: ARM_COLORS[a.arm],color:'#fff',
                      fontFamily:'var(--font-mono)',fontSize:'0.58rem',fontWeight:700,
                      padding:'2px 7px',borderRadius:3,width:52,textAlign:'center',flexShrink:0,
                    }}>{ARM_LABELS[a.arm]}</span>
                    <div style={{flex:1,display:'flex',flexDirection:'column',gap:2}}>
                      <div style={{display:'flex',justifyContent:'space-between',fontSize:'0.62rem',color:'var(--text-secondary)'}}>
                        <span>{a.closed} closed · {a.winRate != null ? `${a.winRate.toFixed(0)}% win` : '—'}</span>
                        <span style={{fontFamily:'var(--font-mono)',fontWeight:700,color:a.pnl>=0?'var(--ok)':'var(--bad)'}}>
                          ${a.pnl.toFixed(2)}
                        </span>
                      </div>
                      <div style={{height:5,background:'var(--bg-panel-2)',borderRadius:3,overflow:'hidden'}}>
                        <div style={{
                          height:'100%',background:ARM_COLORS[a.arm],borderRadius:3,
                          width:`${Math.min(100,Math.abs(a.pnl/Math.max(1,...armStats.map(x=>Math.abs(x.pnl))))*100)}%`,
                          transition:'width 0.6s',
                        }}/>
                      </div>
                    </div>
                  </div>
                ))}
                <div style={{marginTop:8,padding:'7px 10px',background:'var(--graph-bg)',borderRadius:6,border:'1px solid var(--graph)22'}}>
                  <span style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:'var(--graph)'}}>
                    Causal edge: evidence-based directional reasoning vs learned policy bias
                  </span>
                </div>
              </div>
            )}
          </ResearchCard>

          {/* Trade outcomes timeline */}
          <ResearchCard title="Recent Closed Trades" accent="#16a34a">
            {closedTrades.length === 0 ? (
              <EmptyState icon="📊" text="No closed trades yet. CausalAgent will trade when causal score threshold is met." />
            ) : (
              <div style={{display:'flex',flexDirection:'column',gap:3}}>
                {closedTrades.slice(0,10).map((d,i) => {
                  const pnl = d.outcome?.actual_pnl || 0
                  const tp  = d.tuned_params || {}
                  return (
                    <div key={d.id} style={{
                      display:'flex',alignItems:'center',gap:8,
                      padding:'5px 8px',borderRadius:5,
                      background: pnl > 0 ? 'var(--ok-bg)' : pnl < 0 ? 'var(--bad-bg)' : 'var(--bg-panel)',
                      borderLeft: `3px solid ${pnl > 0 ? 'var(--ok)' : pnl < 0 ? 'var(--bad)' : 'var(--border)'}`,
                    }}>
                      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',color:'var(--text-secondary)',width:50,flexShrink:0}}>
                        {d.timestamp_committed ? new Date(d.timestamp_committed).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) : '—'}
                      </span>
                      <span style={{flex:1,fontFamily:'var(--font-mono)',fontSize:'0.60rem',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',color:'var(--text-secondary)'}}>
                        {tp.chain_summary || '—'}
                      </span>
                      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.68rem',fontWeight:700,color:pnl>=0?'var(--ok)':'var(--bad)',width:52,textAlign:'right'}}>
                        {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                      </span>
                      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',color:'var(--text-secondary)',width:90,textAlign:'right',overflow:'hidden',textOverflow:'ellipsis'}}>
                        {d.outcome?.close_reason || '—'}
                      </span>
                    </div>
                  )
                })}
                {closedTrades.length > 10 && (
                  <div style={{textAlign:'center',fontSize:'0.62rem',color:'var(--text-secondary)',padding:'4px 0'}}>
                    +{closedTrades.length-10} more on Ledger page
                  </div>
                )}
              </div>
            )}
          </ResearchCard>
        </div>

        {/* ── Row 3: Trust + Close reasons ── */}
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:14,marginBottom:14}}>

          {/* Trust scores */}
          <ResearchCard title="Layer 2 — Bayesian Edge Trust" accent="#9333ea">
            {scores.length === 0 ? (
              <EmptyState icon="◌" text="No trust scores yet. Updates after each closed trade via Beta posterior." />
            ) : (
              <>
                <div style={{marginBottom:10,display:'flex',gap:8,flexWrap:'wrap'}}>
                  <Chip label={`${scores.length} edges tracked`} color="#9333ea" />
                  <Chip label={`${scores.filter(s=>s.trust>0.6).length} high confidence`} color="var(--ok)" />
                  <Chip label={`${scores.filter(s=>s.trust<0.4).length} weak`} color="var(--bad)" />
                </div>
                <div style={{display:'flex',flexDirection:'column',gap:5,maxHeight:220,overflowY:'auto'}}>
                  {scores.map(s => (
                    <div key={s.edge_key} style={{display:'flex',alignItems:'center',gap:8}}>
                      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',color:'var(--text-secondary)',width:170,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',flexShrink:0}}>
                        {s.source} → {s.target} <span style={{color:'var(--border)'}}>L{s.lag}</span>
                      </span>
                      <div style={{flex:1,height:5,background:'var(--bg-panel-2)',borderRadius:3,overflow:'hidden'}}>
                        <div style={{height:'100%',borderRadius:3,transition:'width 0.6s',
                          background: s.trust > 0.6 ? 'var(--ok)' : s.trust < 0.4 ? 'var(--bad)' : '#ca8a04',
                          width:`${s.trust*100}%`
                        }}/>
                      </div>
                      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',fontWeight:700,
                        width:38,textAlign:'right',
                        color:s.trust>0.6?'var(--ok)':s.trust<0.4?'var(--bad)':'var(--warn)'}}>
                        {s.trust.toFixed(3)}
                      </span>
                      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',color:'var(--text-secondary)',width:28,textAlign:'right'}}>
                        n={s.n_observations}
                      </span>
                    </div>
                  ))}
                </div>
                <div style={{marginTop:8,fontSize:'0.62rem',color:'var(--text-secondary)',fontStyle:'italic'}}>
                  Beta(α, β) posterior — meaningful after ~15 observations
                </div>
              </>
            )}
          </ResearchCard>

          {/* Close reason distribution */}
          <ResearchCard title="Close Reason Distribution" accent="#475569">
            {closedTrades.length === 0 ? (
              <EmptyState icon="◫" text="No closed trades yet." />
            ) : (() => {
              const reasons = {}
              closedTrades.forEach(d => {
                const r = d.outcome?.close_reason || 'unknown'
                reasons[r] = (reasons[r] || 0) + 1
              })
              const sorted = Object.entries(reasons).sort((a,b) => b[1]-a[1])
              const REASON_COLORS = {
                assumption_breach: 'var(--warn)',
                horizon_expired:   'var(--graph)',
                system_shutdown:   'var(--text-secondary)',
                manual:            'var(--text-secondary)',
              }
              return (
                <div style={{display:'flex',flexDirection:'column',gap:7}}>
                  {sorted.map(([reason, count]) => (
                    <div key={reason} style={{display:'flex',alignItems:'center',gap:8}}>
                      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.62rem',width:180,flexShrink:0,
                        color:REASON_COLORS[reason]||'var(--text-secondary)'}}>
                        {reason}
                      </span>
                      <div style={{flex:1,height:8,background:'var(--bg-panel-2)',borderRadius:4,overflow:'hidden'}}>
                        <div style={{height:'100%',borderRadius:4,
                          background:REASON_COLORS[reason]||'var(--graph)',
                          width:`${(count/closedTrades.length)*100}%`,
                          transition:'width 0.6s',
                        }}/>
                      </div>
                      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.70rem',fontWeight:600,width:24,textAlign:'right'}}>
                        {count}
                      </span>
                      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:'var(--text-secondary)',width:36,textAlign:'right'}}>
                        {(count/closedTrades.length*100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                  <div style={{marginTop:6,fontSize:'0.62rem',color:'var(--text-secondary)',fontStyle:'italic',lineHeight:1.5}}>
                    assumption_breach = trajectory exited predicted causal band.
                    Natural during early training when hypothesis calibration is loose.
                  </div>
                </div>
              )
            })()}
          </ResearchCard>
        </div>

        {/* ── Row 4: Meta Optimizer ── */}
        <ResearchCard title="Meta Parameter Optimizer — Regime Populations" accent="#9333ea">
          {regimes.length === 0 ? (
            <EmptyState icon="◈" text="No optimizer state yet — populates after first causal discovery run." />
          ) : (
            <div style={{display:'flex',flexDirection:'column',gap:18}}>
              {regimes.map(regime => (
                <div key={regime.regime}>
                  <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:8}}>
                    <span style={{
                      fontFamily:'var(--font-mono)',fontSize:'0.62rem',fontWeight:700,
                      letterSpacing:'0.08em',color:'var(--optimizer)',
                    }}>REGIME: {regime.regime?.toUpperCase()}</span>
                    <span style={{fontSize:'0.60rem',color:'var(--text-secondary)',fontFamily:'var(--font-mono)'}}>
                      {regime.history_count} updates
                    </span>
                    {regime.recent_avg_score != null && (
                      <span style={{fontSize:'0.60rem',fontFamily:'var(--font-mono)',
                        color:regime.recent_avg_score>0?'var(--ok)':regime.recent_avg_score<0?'var(--bad)':'var(--text-secondary)'}}>
                        avg={regime.recent_avg_score}
                      </span>
                    )}
                  </div>
                  <div className="cand-grid">
                    {(regime.population || []).map((c, i) => (
                      <div key={i} className={`cand-card ${i===0?'best':''}`}>
                        <div className="cand-label">CAND {['A','B','C'][i]}{i===0?' ●':''}</div>
                        <div className="cand-row"><span className="cand-key">k_runs</span><span className="cand-val">{c.k_runs}</span></div>
                        <div className="cand-row"><span className="cand-key">min_runs</span><span className="cand-val">{c.min_runs}</span></div>
                        <div className="cand-row"><span className="cand-key">threshold</span><span className="cand-val">{c.threshold?.toFixed(3)}</span></div>
                        {c.recent_score != null && (
                          <div style={{marginTop:5,height:3,background:'var(--bg-panel-2)',borderRadius:2,overflow:'hidden'}}>
                            <div style={{height:'100%',background:'var(--optimizer)',borderRadius:2,
                              width:`${Math.min(100,Math.abs(c.recent_score)*100)}%`}}/>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
              <div style={{padding:'8px 12px',background:'var(--optimizer-bg)',borderRadius:6,border:'1px solid var(--optimizer)22'}}>
                <span style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:'var(--optimizer)'}}>
                  Hill-climbing on PCMCI threshold params per market regime.
                  Population of 3 candidates evolves via score-weighted replacement.
                </span>
              </div>
            </div>
          )}
        </ResearchCard>

      </div>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────

function ResearchCard({ title, accent, children }) {
  return (
    <div style={{
      background:'#fff',
      border:'1px solid var(--border-light)',
      borderRadius:10,
      overflow:'hidden',
      borderTop:`3px solid ${accent}`,
    }}>
      <div style={{
        padding:'10px 14px',
        borderBottom:'1px solid var(--border-light)',
        display:'flex',alignItems:'center',gap:8,
      }}>
        <span style={{
          width:6,height:6,borderRadius:'50%',
          background:accent,flexShrink:0,
        }}/>
        <h3 style={{fontSize:'0.78rem',fontWeight:700,letterSpacing:'0.02em',color:'var(--text-primary)'}}>
          {title}
        </h3>
      </div>
      <div style={{padding:'12px 14px'}}>
        {children}
      </div>
    </div>
  )
}

function KpiCard({ label, val, sub, color }) {
  return (
    <div style={{
      background:'#fff',
      border:'1px solid var(--border-light)',
      borderRadius:8,
      padding:'10px 12px',
    }}>
      <div style={{fontSize:'0.60rem',color:'var(--text-secondary)',marginBottom:4,letterSpacing:'0.06em',fontWeight:600}}>
        {label.toUpperCase()}
      </div>
      <div style={{fontFamily:'var(--font-mono)',fontSize:'1.1rem',fontWeight:700,color:color||'var(--text-primary)',marginBottom:2}}>
        {val}
      </div>
      <div style={{fontSize:'0.60rem',color:'var(--text-secondary)'}}>
        {sub}
      </div>
    </div>
  )
}

function Chip({ label, color }) {
  return (
    <span style={{
      display:'inline-flex',alignItems:'center',gap:4,
      padding:'2px 7px',borderRadius:3,
      fontFamily:'var(--font-mono)',fontSize:'0.60rem',fontWeight:600,
      background:color+'18',color,border:`1px solid ${color}33`,
    }}>
      {label}
    </span>
  )
}

function EmptyState({ icon, text }) {
  return (
    <div style={{
      display:'flex',flexDirection:'column',alignItems:'center',
      justifyContent:'center',padding:'32px 20px',gap:8,
      color:'var(--text-secondary)',
    }}>
      <span style={{fontSize:'1.5rem',opacity:0.2}}>{icon}</span>
      <p style={{fontSize:'0.72rem',textAlign:'center',maxWidth:260,lineHeight:1.6}}>{text}</p>
    </div>
  )
}
