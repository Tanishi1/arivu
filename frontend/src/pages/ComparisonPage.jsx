import { useState, useEffect, useCallback } from 'react'
import { NavLink } from 'react-router-dom'
import { useLiveFeed } from '../hooks/useSystemState'

// ── Arm definitions ───────────────────────────────────
const ARMS = {
  ppo_standard: {
    label:    'PPO Standard',
    tag:      'PPO-S',
    color:    '#7c3aed',
    bg:       '#f5f3ff',
    border:   '#7c3aed',
    subtitle: 'Policy-based · directional bias · bearish learned policy',
    mechanism:'Policy Network',
    steps: [
      { id:'OBS',    label:'Observation Vector',      desc:'14 standard market features (price, volume, RSI, spread, momentum…)' },
      { id:'NET',    label:'PPO Neural Network',       desc:'3-layer MLP — maps obs → action logits via learned weights' },
      { id:'ACTION', label:'Action Output',            desc:'Outputs BUY / SELL / HOLD — currently persistent SELL (bearish policy)' },
      { id:'EXEC',   label:'Execution Gate',           desc:'SELL with no open long → converted to HOLD (no-op)' },
    ]
  },
  ppo_causal_feature: {
    label:    'PPO + Causal Features',
    tag:      'PPO-C',
    color:    '#9333ea',
    bg:       '#faf5ff',
    border:   '#9333ea',
    subtitle: 'Policy-based · enriched obs · shares PCMCI graph features',
    mechanism:'Policy Network + Causal Obs',
    steps: [
      { id:'OBS',    label:'Observation Vector',      desc:'Standard features + PCMCI-derived causal features (graph-aware obs)' },
      { id:'CAUSAL', label:'Causal Feature Injection', desc:'ML1 graph features injected into obs space — edge strengths, lag values from PCMCI' },
      { id:'NET',    label:'PPO Neural Network',       desc:'Same PPO architecture — but sees causal structure in its input space' },
      { id:'ACTION', label:'Action Output',            desc:'Outputs BUY / SELL / HOLD — still a learned policy, same bearish tendency' },
      { id:'EXEC',   label:'Execution Gate',           desc:'Same SELL → HOLD conversion when no long position open' },
    ]
  },
  CausalAgent: {
    label:    'Causal Agent',
    tag:      'ARIVU',
    color:    '#2563eb',
    bg:       '#eff6ff',
    border:   '#2563eb',
    subtitle: 'Evidence-based · no directional bias · statistically validated',
    mechanism:'Causal Reasoning Loop',
    steps: [
      { id:'FEED',      label:'Market Feed',            desc:'Raw Binance ticks — price, spread, volume, order book depth' },
      { id:'BARS',      label:'Feature Bar Construction', desc:'Ticks compressed into OHLCV + derived features (RSI, volatility, momentum…)' },
      { id:'GRAPH',     label:'PCMCI Causal Discovery', desc:'Discovers causal DAG from feature time series — which variable Granger-causes price_return?' },
      { id:'THESIS',    label:'Hypothesis Generation',  desc:'Enumerates all directed paths to price_return, scores each by causal strength' },
      { id:'READINESS', label:'Agent Readiness Gate',   desc:'ML1 health vector (normal/stressed/degraded) — must be normal to trade' },
      { id:'TWIN',      label:'Response Twin Simulation', desc:'Simulates each hypothesis trajectory — checks if predicted return exceeds noise band' },
      { id:'LEDGER',    label:'Commit Decision',         desc:'Writes decision object to tamper-proof ledger with full audit trail' },
      { id:'TRUST',     label:'Trust Update',           desc:'After trade closes — Bayesian Beta update on edge trust scores' },
      { id:'OPTIMIZER', label:'Meta Optimizer',         desc:'Hill-climbing on causal threshold params per market regime' },
    ]
  }
}

const ARM_ORDER = ['ppo_standard','ppo_causal_feature','CausalAgent']

export default function ComparisonPage() {
  const [armData, setArmData]   = useState({})
  const [graph, setGraph]       = useState(null)
  const [activeHyp, setActiveHyp] = useState(null)
  const [ticks, setTicks]       = useState([])
  const [causalStep, setCausalStep] = useState(0)

  // Load per-arm ledger stats
  useEffect(() => {
    const load = async () => {
      const results = {}
      await Promise.all(ARM_ORDER.map(async arm => {
        try {
          const r = await fetch(`/api/ledger?limit=20&strategy=${arm}`).then(r => r.json())
          results[arm] = r
        } catch {}
      }))
      const g = await fetch('/api/graph/latest').then(r => r.json()).catch(()=>({}))
      const state = await fetch('/api/state').then(r => r.json()).catch(()=>({}))
      setArmData(results)
      setGraph(g.graph)
      if (state.live?.selected_chain) setActiveHyp(state.live.selected_chain)
      if (state.live?.hypotheses) setActiveHyp(state.live.hypotheses[0]?.chain)
    }
    load()
    const id = setInterval(load, 10000)
    return () => clearInterval(id)
  }, [])

  // WebSocket for live ticks + causal agent step animation
  const onFeedMsg = useCallback(msg => {
    setTicks(prev => [msg, ...prev].slice(0, 5))
    // Animate causal step forward
    setCausalStep(s => (s + 1) % 9)
  }, [])
  useLiveFeed(onFeedMsg)

  const latestTick = ticks[0]

  return (
    <div className="shell">
      {/* Nav */}
      <nav className="top-nav">
        <span className="nav-brand">ARIVU</span>
        <div className="nav-tabs">
          {[['/', 'Cockpit'], ['/compare', 'Comparison'], ['/ledger', 'Ledger'], ['/research', 'Research']].map(([to, label]) => (
            <NavLink key={to} to={to} end={to==='/'} className={({isActive})=>`nav-tab${isActive?' active':''}`}>{label}</NavLink>
          ))}
        </div>
        <div className="nav-right live-ticker">
          <span style={{fontFamily:'var(--font-mono)',fontSize:'0.64rem',color:'var(--text-secondary)',letterSpacing:'0.06em'}}>
            SAME MARKET · THREE MECHANISMS
          </span>
          {latestTick?.price && (
            <>
              <span style={{color:'var(--border)'}}>·</span>
              <span style={{fontFamily:'var(--font-mono)',fontSize:'0.80rem',fontWeight:700}}>
                ${Number(latestTick.price).toFixed(4)}
              </span>
              <span style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:'var(--text-secondary)'}}>SOL/USDT</span>
            </>
          )}
        </div>
      </nav>

      {/* Explanation bar */}
      <div style={{padding:'7px 18px',background:'var(--bg-panel)',borderBottom:'1px solid var(--border-light)',display:'flex',gap:24,flexWrap:'wrap'}}>
        {ARM_ORDER.map(arm => {
          const cfg = ARMS[arm]
          return (
            <div key={arm} style={{display:'flex',alignItems:'center',gap:7}}>
              <span style={{background:cfg.color,color:'#fff',fontFamily:'var(--font-mono)',fontSize:'0.60rem',fontWeight:700,padding:'2px 7px',borderRadius:3,letterSpacing:'0.06em'}}>
                {cfg.tag}
              </span>
              <span style={{fontSize:'0.68rem',color:'var(--text-secondary)'}}>{cfg.subtitle}</span>
            </div>
          )
        })}
      </div>

      {/* 3 vertical panels */}
      <div style={{display:'flex',flex:1,minHeight:0,overflow:'hidden'}}>
        {ARM_ORDER.map((armKey, idx) => {
          const cfg  = ARMS[armKey]
          const data = armData[armKey]
          const decisions = data?.decisions || []
          const closed = decisions.filter(d => d.outcome)
          const wins = closed.filter(d => (d.outcome?.actual_pnl||0) > 0)
          const winRate = closed.length > 0 ? (wins.length/closed.length*100).toFixed(0) : null
          const totalPnl = closed.reduce((s,d)=>s+(d.outcome?.actual_pnl||0),0)
          const latest = decisions[0]
          const tp = latest?.tuned_params || {}

          // Current action
          const action = armKey === 'CausalAgent'
            ? (latest ? (tp.predicted_direction==='up'?'BUY':tp.predicted_direction==='down'?'SELL':'HOLD') : 'HOLD')
            : (latest?.status==='ACTIVE' ? (tp.predicted_direction==='up'?'BUY':'SELL') : 'HOLD')

          const actionColor = action==='BUY'?'var(--ok)':action==='SELL'?'var(--bad)':'var(--text-secondary)'

          return (
            <div key={armKey} style={{
              flex:1,
              borderRight: idx < 2 ? '1px solid var(--border)' : 'none',
              display:'flex',flexDirection:'column',
              overflow:'hidden',
              background: '#fff',
            }}>
              {/* Panel header */}
              <div style={{
                padding:'10px 14px',
                borderBottom:'1px solid var(--border-light)',
                background: cfg.bg,
                flexShrink:0,
              }}>
                <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:4}}>
                  <span style={{background:cfg.color,color:'#fff',fontFamily:'var(--font-mono)',fontSize:'0.65rem',fontWeight:700,padding:'3px 9px',borderRadius:4,letterSpacing:'0.06em'}}>
                    {cfg.tag}
                  </span>
                  <span style={{fontWeight:700,fontSize:'0.88rem'}}>{cfg.label}</span>
                </div>
                <div style={{fontSize:'0.65rem',color:'var(--text-secondary)'}}>{cfg.mechanism}</div>
              </div>

              {/* Current action */}
              <div style={{
                padding:'10px 14px',
                borderBottom:'1px solid var(--border-light)',
                display:'flex',alignItems:'center',gap:12,
                flexShrink:0,
                background: action==='BUY'?'var(--ok-bg)':action==='SELL'?'var(--bad-bg)':'var(--bg-panel)',
              }}>
                <div>
                  <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.08em',color:'var(--text-secondary)',marginBottom:3}}>CURRENT ACTION</div>
                  <div style={{fontFamily:'var(--font-mono)',fontSize:'1.5rem',fontWeight:800,color:actionColor,lineHeight:1}}>{action}</div>
                </div>
                <div style={{flex:1,borderLeft:'1px solid var(--border-light)',paddingLeft:12}}>
                  <div style={{fontSize:'0.60rem',color:'var(--text-secondary)',marginBottom:3}}>WHY</div>
                  <div style={{fontFamily:'var(--font-mono)',fontSize:'0.62rem',color:'var(--text-secondary)',lineHeight:1.5}}>
                    {armKey==='CausalAgent'
                      ? (action==='HOLD'
                          ? (latestTick?.hold_reason || 'no qualifying causal chain')
                          : tp.chain_summary || activeHyp || '—')
                      : action==='HOLD'
                        ? 'SELL with no open long → converted to HOLD'
                        : 'Policy network outputs directional signal'}
                  </div>
                </div>
              </div>

              {/* Stats row */}
              <div style={{
                display:'grid',gridTemplateColumns:'repeat(3,1fr)',
                borderBottom:'1px solid var(--border-light)',
                flexShrink:0,
              }}>
                <Stat label="Decisions" val={decisions.length} />
                <Stat label="Win Rate"  val={winRate!=null?`${winRate}%`:'—'} color={winRate>50?'var(--ok)':winRate<40?'var(--bad)':null} />
                <Stat label="PnL"       val={closed.length?`$${totalPnl.toFixed(2)}`:'—'} color={totalPnl>=0?'var(--ok)':'var(--bad)'} />
              </div>

              {/* Decision pipeline steps */}
              <div style={{flex:1,overflow:'auto',padding:'12px 14px'}}>
                <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.1em',color:'var(--text-secondary)',marginBottom:10}}>
                  DECISION PIPELINE
                </div>
                <div style={{display:'flex',flexDirection:'column',gap:0}}>
                  {cfg.steps.map((step,i) => {
                    const isActive = armKey==='CausalAgent'
                      ? i === causalStep
                      : i === cfg.steps.length-1 // RL: always show last step as active

                    return (
                      <div key={step.id}>
                        <div style={{
                          display:'flex',gap:10,
                          padding:'8px 10px',
                          borderRadius:6,
                          background: isActive ? cfg.bg : 'transparent',
                          border: `1px solid ${isActive ? cfg.border : 'transparent'}`,
                          transition:'all 0.3s',
                          marginBottom:2,
                        }}>
                          {/* Step badge */}
                          <div style={{
                            flexShrink:0,width:22,height:22,borderRadius:'50%',
                            display:'flex',alignItems:'center',justifyContent:'center',
                            background: isActive ? cfg.color : 'var(--bg-panel-2)',
                            color: isActive ? '#fff' : 'var(--text-secondary)',
                            fontSize:'0.60rem',fontWeight:700,
                            transition:'all 0.3s',
                          }}>{i+1}</div>

                          <div style={{flex:1,minWidth:0}}>
                            <div style={{
                              fontFamily:'var(--font-mono)',fontSize:'0.65rem',fontWeight:700,
                              color: isActive ? cfg.color : 'var(--text-primary)',
                              marginBottom:2,
                            }}>
                              {step.label}
                            </div>
                            <div style={{fontSize:'0.62rem',color:'var(--text-secondary)',lineHeight:1.5}}>
                              {step.desc}
                            </div>

                            {/* Extra data for active causal agent steps */}
                            {armKey==='CausalAgent' && isActive && (
                              <StepData step={step.id} graph={graph} tick={latestTick} activeHyp={activeHyp} cfg={cfg} />
                            )}
                          </div>
                        </div>

                        {/* Arrow connector */}
                        {i < cfg.steps.length-1 && (
                          <div style={{
                            marginLeft:20,width:2,height:14,
                            background: i < (armKey==='CausalAgent'?causalStep:cfg.steps.length-1)
                              ? cfg.color : 'var(--border-light)',
                            transition:'background 0.3s',
                          }}/>
                        )}
                      </div>
                    )
                  })}
                </div>

                {/* Architecture distinction callout */}
                <div style={{
                  marginTop:14,padding:'9px 12px',
                  background:'var(--bg-panel)',
                  borderRadius:6,border:'1px solid var(--border-light)',
                }}>
                  <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.08em',color:'var(--text-secondary)',marginBottom:5}}>
                    ARCHITECTURE NOTE
                  </div>
                  <div style={{fontSize:'0.62rem',color:'var(--text-secondary)',lineHeight:1.6}}>
                    {armKey==='ppo_standard' && 'No causal awareness. Policy learned entirely from price history. Bearish bias is a learned statistical artifact — not a causal claim.'}
                    {armKey==='ppo_causal_feature' && 'Knows WHAT the causal graph says (features in obs). But does not REASON about it — the neural net decides how to use that information.'}
                    {armKey==='CausalAgent' && 'Does not have a directional bias. HOLD ≠ bearish. HOLD = "no statistically valid causal chain found." Will fire in either direction when evidence exists.'}
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Per-step live data ────────────────────────────────
function StepData({ step, graph, tick, activeHyp, cfg }) {
  if (step === 'FEED' && tick) return (
    <div style={{marginTop:5,fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:cfg.color}}>
      ${Number(tick.price).toFixed(4)} · spread={tick.spread?.toFixed(5)} · rsi={tick.rsi?.toFixed(1)}
    </div>
  )
  if (step === 'BARS') return (
    <div style={{marginTop:5,fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:cfg.color}}>
      200/200 bars ready · {graph?.n_bars_used ?? '—'} bars used in latest graph
    </div>
  )
  if (step === 'GRAPH' && graph) return (
    <div style={{marginTop:5,fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:cfg.color}}>
      v{graph.version_id?.slice(0,8)} · {graph.edge_count} edges · {graph.edges?.filter(e=>e.validated).length ?? 0} validated · {graph.algorithm}
    </div>
  )
  if (step === 'THESIS') return (
    <div style={{marginTop:5,fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:cfg.color}}>
      {activeHyp ? `Selected: ${activeHyp}` : 'score_too_low — no chain meets threshold'}
    </div>
  )
  if (step === 'READINESS') return (
    <div style={{marginTop:5,fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:cfg.color}}>
      {tick?.hold_reason === 'ml1_degraded' ? '⊘ ML1 degraded — gate closed' : '✓ gate open'}
    </div>
  )
  return null
}

// ── Stat cell ─────────────────────────────────────────
function Stat({ label, val, color }) {
  return (
    <div style={{padding:'7px 10px',borderRight:'1px solid var(--border-light)'}}>
      <div style={{fontSize:'0.58rem',color:'var(--text-secondary)',marginBottom:2}}>{label}</div>
      <div style={{fontFamily:'var(--font-mono)',fontSize:'0.82rem',fontWeight:700,color:color||'var(--text-primary)'}}>{val ?? '—'}</div>
    </div>
  )
}
