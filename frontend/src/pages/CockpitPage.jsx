import { useState, useCallback, useRef, useEffect } from 'react'
import { NavLink } from 'react-router-dom'
import { useSystemState, useLiveFeed } from '../hooks/useSystemState'
import { useReplay } from '../hooks/useReplay'
import LiveGraph from '../components/LiveGraph'
import ThesisSidebar from '../components/ThesisSidebar'

// Pipeline stages match VSTAGES in LiveGraph.jsx
const STAGES = [
  { id:'STREAMS',   label:'Streams' },
  { id:'BARS',      label:'Bars' },
  { id:'PCMCI',     label:'PCMCI' },
  { id:'HYPOTHESIS',label:'Hypothesis' },
  { id:'OPTIMIZER', label:'Optimizer' },
  { id:'TWIN',      label:'Twin' },
  { id:'DECISION',  label:'Decision' },
  { id:'LEARNING',  label:'Learning' },
]

const NAV_LINKS = [
  { to: '/',         label: 'Cockpit' },
  { to: '/compare',  label: 'Comparison' },
  { to: '/ledger',   label: 'Ledger' },
  { to: '/research', label: 'Research' },
]

export default function CockpitPage() {
  const { state, graph, trust, optimizer, loading } = useSystemState()
  const [ticks, setTicks]       = useState([])
  const [price, setPrice]       = useState(null)
  const [flash, setFlash]       = useState('')
  const [liveData, setLiveData] = useState(null)
  const [stage, setStage]       = useState('STREAMS')
  const [playing, setPlaying]   = useState(true)
  const prevP   = useRef(null)
  const tkCount = useRef(0)
  const replay  = useReplay()

  // Sync replay stage -> local stage when replaying (proper effect, not in-body)
  const REPLAY_MAP = {
    FEED:'STREAMS', STREAMS:'STREAMS',
    BARS:'BARS',
    GRAPH:'PCMCI', PCMCI:'PCMCI',
    THESIS:'HYPOTHESIS', HYPOTHESIS:'HYPOTHESIS',
    OPTIMIZER:'OPTIMIZER',
    TWIN:'TWIN',
    READINESS:'DECISION', DECISION:'DECISION', LEDGER:'DECISION',
    TRUST:'LEARNING', LEARNING:'LEARNING',
  }
  // useEffect to follow replay.activeStage when replay is playing
  useEffect(() => {
    if(!replay.replay || !replay.playing) return
    const raw = replay.STAGES[replay.activeStage]
    const mapped = REPLAY_MAP[raw] || 'STREAMS'
    setStage(mapped)
  }, [replay.activeStage, replay.playing, replay.replay])


  const onTick = useCallback(msg => {
    tkCount.current++
    const p = msg.price
    if (prevP.current !== null && p !== prevP.current) {
      setFlash(p > prevP.current ? 'flash-up' : 'flash-dn')
      setTimeout(() => setFlash(''), 400)
    }
    prevP.current = p
    setPrice(p)
    setLiveData(msg)
    setTicks(prev => [msg, ...prev].slice(0, 200))
  }, [])

  useLiveFeed(onTick)

  const ev = state?.escape_valve || {}
  const feedOk = liveData?.ts && (Date.now() - new Date(liveData.ts).getTime()) < 30000

  // ML1 health verdict
  const vec = state?.active_do?.algo_health_vector || []
  const verdict = Array.isArray(vec) && vec.length === 3
    ? vec[0] > 0.6 ? 'NORMAL' : vec[0] > 0.35 ? 'STRESSED' : 'DEGRADED'
    : null
  const verdictColor = verdict === 'NORMAL' ? 'var(--ok)' : verdict === 'STRESSED' ? 'var(--warn)' : verdict === 'DEGRADED' ? 'var(--bad)' : null

  if (loading) return (
    <div style={{display:'flex',alignItems:'center',justifyContent:'center',height:'100vh',flexDirection:'column',gap:10,background:'var(--bg-base)'}}>
      <div style={{width:28,height:28,border:'2px solid #e2e8f0',borderTopColor:'#4f46e5',borderRadius:'50%',animation:'spin 0.8s linear infinite'}}/>
      <p style={{fontFamily:'monospace',fontSize:'0.75rem',color:'#64748b'}}>Connecting to Arivu…</p>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  )

  return (
    <div style={{display:'flex',flexDirection:'column',height:'100vh',overflow:'hidden',background:'var(--bg-base)'}}>

      {/* ── Top Nav ── */}
      <nav className="top-nav" style={{paddingRight:16}}>
        <span className="nav-brand">ARIVU</span>
        <div className="nav-tabs">
          {NAV_LINKS.map(({to, label}) => (
            <NavLink key={to} to={to} end={to==='/'} className={({isActive}) => `nav-tab${isActive?' active':''}`}>
              {label}
            </NavLink>
          ))}
        </div>
        <div className="nav-right live-ticker">
          {/* Bars counter */}
          <span style={{fontSize:'0.64rem',fontFamily:'var(--font-mono)',color:'var(--text-secondary)'}}>
            BARS <b style={{color:'var(--text-primary)'}}>{state?.bars_ready??0}</b>/200
          </span>
          <span style={{color:'var(--border)'}}>·</span>
          {/* Phase */}
          <span style={{fontSize:'0.64rem',fontFamily:'var(--font-mono)',color:'var(--text-secondary)'}}>
            {(state?.phase||'bootstrap').toUpperCase()}
          </span>
          {/* ML1 verdict chip */}
          {verdict && (
            <>
              <span style={{color:'var(--border)'}}>·</span>
              <span style={{
                fontSize:'0.58rem',fontFamily:'var(--font-mono)',fontWeight:700,
                padding:'2px 7px',borderRadius:3,
                background: verdictColor+'18', color: verdictColor,
                border:`1px solid ${verdictColor}44`,
              }}>ML1:{verdict}</span>
            </>
          )}
          <span style={{color:'var(--border)'}}>·</span>
          {/* Price */}
          <span className={`price-val ${flash}`} style={{fontFamily:'var(--font-mono)',fontSize:'0.82rem',fontWeight:700}}>
            {price ? `$${Number(price).toFixed(4)}` : '—'}
          </span>
          {/* Live dot */}
          <span className={`pulse-dot ${feedOk ? '' : 'stale'}`}/>
          <span style={{fontSize:'0.64rem',fontFamily:'var(--font-mono)',color:'var(--text-secondary)'}}>
            {feedOk ? `LIVE · ${tkCount.current}t` : 'STALE'}
          </span>
          {ev.fired && (
            <span style={{fontSize:'0.60rem',fontFamily:'var(--font-mono)',fontWeight:700,color:'var(--escape)',background:'var(--escape-bg)',padding:'2px 6px',borderRadius:3,border:'1px solid var(--escape)44'}}>⚡ ESC</span>
          )}
        </div>
      </nav>

      {/* ── Stage pills + transport ── */}
      <div style={{
        display:'flex',alignItems:'center',gap:4,padding:'5px 14px',
        background:'#fff',borderBottom:'1px solid var(--border-light)',
        flexShrink:0,flexWrap:'wrap',
      }}>
        <button onClick={() => setPlaying(p => !p)} style={{
          padding:'3px 12px',borderRadius:4,border:'1px solid var(--border)',cursor:'pointer',
          background:playing?'var(--graph)':'#fff',color:playing?'#fff':'var(--graph)',
          fontSize:'0.68rem',fontWeight:700,marginRight:4,flexShrink:0,transition:'all 0.15s',
        }}>{playing ? '⏸ Pause' : '▶ Play'}</button>
        <button onClick={() => setStage(s => { const i = STAGES.findIndex(x => x.id===s); return STAGES[(i+1)%STAGES.length].id })} style={{
          padding:'3px 10px',borderRadius:4,border:'1px solid var(--border)',cursor:'pointer',
          background:'#fff',color:'var(--text-secondary)',fontSize:'0.68rem',fontWeight:600,
          marginRight:8,flexShrink:0,transition:'all 0.12s',
        }}>⏭ Next</button>
        <div style={{width:1,height:18,background:'var(--border-light)',marginRight:4,flexShrink:0}}/>
        {/* Replay button */}
        <button
          onClick={()=>{ replay.load(); setPlaying(false) }}
          title="Load & replay last completed experiment"
          style={{
            padding:'3px 10px',borderRadius:4,cursor:'pointer',flexShrink:0,
            border:'1px solid #4f46e544',transition:'all 0.15s',
            background: replay.replay ? '#4f46e5' : '#f8f7ff',
            color: replay.replay ? '#fff' : '#4f46e5',
            fontSize:'0.65rem',fontWeight:700,
          }}>
          {replay.loading ? '…' : '⟳ Replay'}
        </button>
        {replay.error && <span style={{fontSize:'0.60rem',color:'var(--bad)',fontFamily:'var(--font-mono)'}}>{replay.error}</span>}
        {replay.replay && (
          <>
            <button onClick={replay.playing ? replay.pause : ()=>{replay.play(); setPlaying(false)}} style={{padding:'3px 8px',borderRadius:4,border:'1px solid var(--border)',cursor:'pointer',fontSize:'0.62rem',fontWeight:700,background:'#fff',color:'var(--graph)'}}>
              {replay.playing ? '⏸' : '▶'}
            </button>
            <span style={{fontSize:'0.60rem',fontFamily:'var(--font-mono)',color:'var(--text-secondary)',minWidth:68}}>
              {REPLAY_MAP[replay.STAGES[replay.activeStage]]||replay.STAGES[replay.activeStage]} ({replay.activeStage+1}/{replay.STAGES.length})
            </span>
          </>
        )}
        <div style={{width:1,height:18,background:'var(--border-light)',marginLeft:4,marginRight:4,flexShrink:0}}/>
        {STAGES.map((s, i) => (
          <div key={s.id} style={{display:'flex',alignItems:'center',gap:0}}>
            <button onClick={() => setStage(s.id)} style={{
              padding:'3px 10px',borderRadius:4,border:'none',cursor:'pointer',
              background: stage===s.id ? 'var(--graph)' : 'transparent',
              color: stage===s.id ? '#fff' : 'var(--text-secondary)',
              fontSize:'0.65rem',fontWeight:stage===s.id?700:500,
              transition:'all 0.15s',
            }}>{s.label}</button>
            {i < STAGES.length-1 && <span style={{color:'var(--border-light)',fontSize:'0.7rem',margin:'0 1px'}}>›</span>}
          </div>
        ))}
      </div>

      {/* ── Main area: Canvas + Sidebar ── */}
      <div style={{flex:1,display:'flex',minHeight:0,overflow:'hidden'}}>
        {/* Canvas */}
        <div style={{flex:1,display:'flex',flexDirection:'column',overflow:'hidden',position:'relative'}}>
          <LiveGraph
            graph={graph}
            ticks={ticks}
            selectedChain={liveData?.selected_chain}
            hypotheses={liveData?.hypotheses||[]}
            activeStage={stage}
            playing={playing}
            liveData={liveData}
            optimizer={optimizer}
            agentState={state}
          />
        </div>
        {/* Right sidebar — ThesisSidebar */}
        <ThesisSidebar
          live={liveData}
          activeDo={state?.active_do}
          trust={trust}
          state={state}
        />
      </div>

    </div>
  )
}
