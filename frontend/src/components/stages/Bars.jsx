import { useState, useEffect, useRef } from 'react'

const TICKS_PER_BAR = 10

export default function Bars({ barsReady, live, ticks = [] }) {
  const total = 200
  const pct = barsReady ? Math.min(100, Math.round((barsReady / total) * 100)) : 0

  const [dots, setDots]           = useState([])   // accumulated tick dots
  const [compressing, setCompressing] = useState(false)
  const [barFlash, setBarFlash]   = useState(false)
  const [recentBars, setRecentBars] = useState([])
  const prevTickLen = useRef(0)
  const dotCount    = useRef(0)
  const idRef       = useRef(0)

  useEffect(() => {
    if (ticks.length === prevTickLen.current) return
    prevTickLen.current = ticks.length
    dotCount.current++

    idRef.current++
    setDots(prev => [{ id: idRef.current }, ...prev].slice(0, TICKS_PER_BAR))

    if (dotCount.current >= TICKS_PER_BAR) {
      dotCount.current = 0
      setCompressing(true)
      setTimeout(() => {
        setCompressing(false)
        setBarFlash(true)
        setDots([])
        setRecentBars(prev => [{
          id: idRef.current,
          price: ticks[0]?.price,
          rsi: ticks[0]?.rsi,
          vol: ticks[0]?.volatility,
        }, ...prev].slice(0, 6))
        setTimeout(() => setBarFlash(false), 800)
      }, 450)
    }
  }, [ticks.length, ticks])

  const FEATURE_GROUPS = [
    { label:'Market',    feats:['volatility','spread','volume','rsi'], color:'#4f46e5' },
    { label:'Order Book',feats:['ob_imbalance','depth_imb','ema_sp'], color:'#2563eb' },
    { label:'Macro',     feats:['btc_ret','eth_ret','session_sin','session_cos'], color:'#7c3aed' },
    { label:'Momentum',  feats:['price_mom','ema_vol','vwap_dev'], color:'#0891b2' },
  ]

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:14 }}>

      {/* Progress */}
      <div>
        <div style={{ display:'flex', justifyContent:'space-between', marginBottom:5 }}>
          <span style={{ fontSize:'0.62rem', color:'var(--text-secondary)' }}>Warm-up progress</span>
          <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.65rem', fontWeight:700 }}>{barsReady ?? 0} / {total} bars ({pct}%)</span>
        </div>
        <div style={{ height:6, background:'var(--bg-panel-2)', borderRadius:3, overflow:'hidden' }}>
          <div style={{
            height:'100%', borderRadius:3, transition:'width 0.6s',
            background: pct >= 40 ? 'var(--ok)' : 'var(--warn)',
            width:`${pct}%`,
          }}/>
        </div>
        <div style={{ display:'flex', gap:20, marginTop:5 }}>
          <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.58rem', color: barsReady >= 40 ? 'var(--ok)' : 'var(--text-secondary)' }}>
            {barsReady >= 40 ? '✓' : '⊘'} Granger @ 40
          </span>
          <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.58rem', color: barsReady >= 80 ? 'var(--ok)' : 'var(--text-secondary)' }}>
            {barsReady >= 80 ? '✓' : '⊘'} PCMCI @ 80
          </span>
          <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.58rem', color: barsReady >= 200 ? 'var(--ok)' : 'var(--text-secondary)' }}>
            {barsReady >= 200 ? '✓' : '⊘'} Full @ 200
          </span>
        </div>
      </div>

      {/* Adaptive PCMCI Refresh */}
      <div style={{
        padding:'8px 11px', borderRadius:6,
        background:'linear-gradient(90deg,#7c3aed08,transparent)',
        border:'1px solid #7c3aed22',
        display:'flex', gap:14, alignItems:'center', flexWrap:'wrap',
      }}>
        <div style={{fontSize:'0.58rem',fontWeight:700,letterSpacing:'0.07em',color:'#7c3aed',flexShrink:0}}>
          PCMCI REFRESH
        </div>
        <div style={{display:'flex',gap:10,flexWrap:'wrap',flex:1}}>
          <KV label="τ_max"   val={live?.tau_max       != null ? `${live.tau_max} bars` : '— bars'} color="#7c3aed" />
          <KV label="α"       val={live?.pcmci_alpha   != null ? live.pcmci_alpha.toFixed(3)    : '—'}    color="#7c3aed" />
          <KV label="interval" val={live?.effective_refresh != null ? `${live.effective_refresh}s` : '300s'} />
          <div style={{display:'flex',alignItems:'center',gap:5}}>
            <span style={{fontSize:'0.58rem',color:'var(--text-secondary)'}}>mode</span>
            <span style={{
              fontFamily:'var(--font-mono)',fontSize:'0.62rem',fontWeight:700,padding:'1px 6px',borderRadius:3,
              background: live?.adaptive_fast ? 'var(--warn-bg,#fff7ed)' : 'var(--ok-bg,#f0fdf4)',
              color:      live?.adaptive_fast ? 'var(--warn,#d97706)'    : 'var(--ok,#16a34a)',
              border:     live?.adaptive_fast ? '1px solid #d9770633'    : '1px solid #16a34a33',
            }}>
              {live?.adaptive_fast ? 'FAST (2×)' : 'NORMAL'}
            </span>
          </div>
        </div>
        <div style={{fontSize:'0.57rem',color:'var(--text-secondary)',flexShrink:0}}>
          {live?.adaptive_fast
            ? 'Vol > median → refresh halved'
            : 'Vol ≤ median → full interval'}
        </div>
      </div>

      {/* Live tick → bar compression */}
      <div>
        <div style={{ fontSize:'0.60rem', fontWeight:700, letterSpacing:'0.08em', color:'var(--text-secondary)', marginBottom:8 }}>
          TICK → BAR COMPRESSION
        </div>

        {/* Tick dot buffer */}
        <div style={{
          display:'flex', flexWrap:'wrap', gap:4, minHeight:32,
          padding:'8px', background:'var(--bg-panel-2)',
          border:'1px solid var(--border-light)', borderRadius:6,
          marginBottom:8,
        }}>
          {Array.from({ length: TICKS_PER_BAR }).map((_, i) => {
            const filled = i < dots.length
            return (
              <div key={i} style={{
                width:10, height:10, borderRadius:'50%', flexShrink:0,
                background: filled ? '#4f46e5' : 'var(--border-light)',
                boxShadow: filled ? '0 0 6px #4f46e588' : 'none',
                transform: compressing ? 'scale(0.1)' : 'scale(1)',
                opacity: compressing ? 0 : filled ? 1 : 0.25,
                transition: `all 0.4s ease ${i * 0.03}s`,
              }}/>
            )
          })}
        </div>

        {/* Status */}
        <div style={{
          padding:'5px 10px', borderRadius:5, textAlign:'center',
          fontFamily:'var(--font-mono)', fontSize:'0.65rem',
          background: barFlash ? '#dbeafe' : compressing ? '#fef9c3' : 'var(--bg-panel)',
          border:`1px solid ${barFlash ? '#2563eb' : compressing ? '#ca8a04' : 'var(--border-light)'}`,
          color: barFlash ? '#1d4ed8' : compressing ? '#92400e' : 'var(--text-secondary)',
          fontWeight: barFlash || compressing ? 700 : 400,
          transition:'all 0.3s',
        }}>
          {barFlash
            ? '◉ BAR FORMED → INJECTING INTO GRAPH'
            : compressing
            ? '▶▶ COMPRESSING TICKS → FEATURE BAR'
            : `${dots.length} / ${TICKS_PER_BAR} ticks buffered`}
        </div>
      </div>

      {/* Recent bars */}
      {recentBars.length > 0 && (
        <div>
          <div style={{ fontSize:'0.60rem', fontWeight:700, letterSpacing:'0.08em', color:'var(--text-secondary)', marginBottom:7 }}>
            RECENT BARS
          </div>
          {recentBars.map((b, i) => (
            <div key={b.id} style={{
              display:'flex', alignItems:'center', gap:8,
              padding:'4px 0', borderBottom:'1px solid var(--border-light)',
              opacity: 1 - i * 0.14,
            }}>
              <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.60rem', fontWeight:700, color:'var(--bars, #4f46e5)', width:28 }}>B{barsReady - i}</span>
              <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.65rem', flex:1 }}>${b.price ? Number(b.price).toFixed(3) : '—'}</span>
              <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.60rem', color:'var(--text-secondary)' }}>RSI={b.rsi?.toFixed(0) ?? '—'}</span>
              <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.60rem', color:'var(--text-secondary)' }}>σ={b.vol?.toFixed(5) ?? '—'}</span>
            </div>
          ))}
        </div>
      )}

      {/* Feature groups — what goes into each bar */}
      <div>
        <div style={{ fontSize:'0.60rem', fontWeight:700, letterSpacing:'0.08em', color:'var(--text-secondary)', marginBottom:7 }}>
          FEATURE VECTOR (per bar)
        </div>
        <div style={{ display:'flex', flexWrap:'wrap', gap:5 }}>
          {FEATURE_GROUPS.map(g => (
            <div key={g.label} style={{
              padding:'4px 9px', borderRadius:4,
              border:`1px solid ${g.color}44`, background:`${g.color}10`,
              fontSize:'0.60rem',
            }}>
              <span style={{ fontWeight:700, color:g.color }}>{g.label}</span>
              <span style={{ color:'var(--text-secondary)', marginLeft:5 }}>({g.feats.length} feats)</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function KV({ label, val, color }) {
  return (
    <div style={{display:'flex',alignItems:'center',gap:4}}>
      <span style={{fontSize:'0.58rem',color:'var(--text-secondary)'}}>{label}</span>
      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',fontWeight:700,color: color||'var(--text-primary)'}}>{val}</span>
    </div>
  )
}
