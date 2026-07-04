import { useRef, useEffect, useState } from 'react'

const BAR_EVERY_N_TICKS = 10

export default function TickStream({ ticks=[], barsReady=0, live }) {
  const [dots, setDots] = useState([])
  const [compressing, setCompressing] = useState(false)
  const [barFlash, setBarFlash] = useState(false)
  const [bars, setBars] = useState([]) // mini bar history
  const prevLen = useRef(0)
  const dotCount = useRef(0)
  const idSeq = useRef(0)

  useEffect(() => {
    if (ticks.length === prevLen.current) return
    prevLen.current = ticks.length
    dotCount.current += 1
    idSeq.current += 1

    // Add a dot
    const newDot = { id: idSeq.current, color: '#4f46e5' }
    setDots(prev => {
      const next = [newDot, ...prev].slice(0, BAR_EVERY_N_TICKS)
      return next
    })

    // Every N ticks → compress into bar
    if (dotCount.current >= BAR_EVERY_N_TICKS) {
      dotCount.current = 0
      setCompressing(true)
      setTimeout(() => {
        setCompressing(false)
        setBarFlash(true)
        setDots([])
        const tick = ticks[0]
        setBars(prev => [{
          price: tick?.price,
          vol: tick?.volatility,
          rsi: tick?.rsi,
          id: idSeq.current,
        }, ...prev].slice(0, 8))
        setTimeout(() => setBarFlash(false), 600)
      }, 400)
    }
  }, [ticks.length])

  const ok = live?.ts && (Date.now() - new Date(live.ts).getTime()) < 30000

  return (
    <div style={{
      width: 230, flexShrink: 0,
      background: 'var(--bg-panel)',
      borderRight: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{padding:'8px 12px',borderBottom:'1px solid var(--border-light)',display:'flex',alignItems:'center',gap:6}}>
        <span style={{fontFamily:'var(--font-mono)',fontSize:'0.62rem',fontWeight:700,letterSpacing:'0.1em',color:'var(--feed)'}}>FEED</span>
        <span className={`pulse-dot ${ok?'':'stale'}`} />
        <span style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',color:'var(--text-secondary)',marginLeft:'auto'}}>
          {ticks.length}t
        </span>
      </div>

      {/* Price */}
      <div style={{padding:'10px 12px',borderBottom:'1px solid var(--border-light)'}}>
        <div style={{fontFamily:'var(--font-mono)',fontSize:'1.3rem',fontWeight:700,letterSpacing:'-0.01em'}}>
          {live?.price ? `$${Number(live.price).toFixed(4)}` : '—'}
        </div>
        <div style={{display:'flex',gap:10,marginTop:5}}>
          <Kv k="RSI"   v={live?.rsi?.toFixed(1) ?? '—'} />
          <Kv k="Vol"   v={live?.volatility?.toFixed(5) ?? '—'} />
        </div>
        <div style={{display:'flex',gap:10,marginTop:2}}>
          <Kv k="Spr"   v={live?.spread?.toFixed(5) ?? '—'} />
          <Kv k="Edges" v={live?.graph_edges ?? '—'} />
        </div>
      </div>

      {/* Tick → Bar accumulator */}
      <div style={{padding:'10px 12px',borderBottom:'1px solid var(--border-light)'}}>
        <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.08em',color:'var(--text-secondary)',marginBottom:7}}>
          TICK → BAR COMPRESSION
        </div>

        {/* Dot buffer */}
        <div style={{
          display:'flex',flexWrap:'wrap',gap:3,minHeight:28,
          padding:'6px',background:'var(--bg-base)',
          borderRadius:5,border:'1px solid var(--border-light)',
          marginBottom:8,
          transition:'all 0.3s',
        }}>
          {Array.from({length: BAR_EVERY_N_TICKS}).map((_,i) => {
            const hasDot = i < dots.length
            return (
              <div key={i} style={{
                width:8,height:8,borderRadius:'50%',flexShrink:0,
                background: hasDot ? '#4f46e5' : 'var(--border)',
                opacity: compressing ? 0 : hasDot ? 1 : 0.2,
                transform: compressing ? 'scale(0.2)' : 'scale(1)',
                transition: `all 0.35s ease ${i*0.02}s`,
              }}/>
            )
          })}
        </div>

        {/* Compress arrow */}
        <div style={{
          textAlign:'center',
          fontFamily:'var(--font-mono)',fontSize:'0.62rem',
          color: compressing ? 'var(--feed)' : 'var(--text-secondary)',
          fontWeight: compressing ? 700 : 400,
          transition:'color 0.3s',
          marginBottom:6,
        }}>
          {compressing ? '▶ COMPRESSING ▶▶' : `▷ ${dots.length}/${BAR_EVERY_N_TICKS} ticks`}
        </div>

        {/* Bar flash */}
        <div style={{
          padding:'4px 8px',
          background: barFlash ? 'var(--graph-bg)' : 'var(--bg-panel)',
          border:`1px solid ${barFlash ? 'var(--graph)' : 'var(--border-light)'}`,
          borderRadius:4,
          transition:'all 0.3s',
          fontSize:'0.62rem',
          fontFamily:'var(--font-mono)',
          color: barFlash ? 'var(--graph)' : 'var(--text-secondary)',
          fontWeight: barFlash ? 700 : 400,
        }}>
          {barFlash ? '◉ BAR FORMED → INJECTING TO GRAPH' : `${barsReady} bars ready`}
        </div>
      </div>

      {/* Bar history */}
      <div style={{flex:1,overflow:'hidden',padding:'8px 12px'}}>
        <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.08em',color:'var(--text-secondary)',marginBottom:6}}>
          BARS · {barsReady}/200
        </div>

        {/* Progress bar */}
        <div style={{height:4,background:'var(--bg-panel-2)',borderRadius:2,overflow:'hidden',marginBottom:8}}>
          <div style={{
            height:'100%',borderRadius:2,
            background: barsReady >= 80 ? 'var(--ok)' : barsReady >= 40 ? 'var(--warn)' : 'var(--bad)',
            width:`${Math.min(100,(barsReady/200)*100)}%`,
            transition:'width 0.5s',
          }}/>
        </div>

        <div style={{fontSize:'0.58rem',color:'var(--text-secondary)',fontFamily:'var(--font-mono)',marginBottom:8}}>
          {barsReady < 40 ? '⊘ Granger not ready (40)' :
           barsReady < 80 ? '✓ Granger · ⊘ PCMCI (80)' :
                            '✓ Granger · ✓ PCMCI · full'}
        </div>

        {/* Recent bars */}
        {bars.map((b,i) => (
          <div key={b.id} style={{
            display:'flex',alignItems:'center',gap:6,
            padding:'3px 0',
            borderBottom:'1px solid var(--border-light)',
            opacity: 1 - i*0.12,
          }}>
            <span style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',fontWeight:600,color:'var(--graph)'}}>BAR</span>
            <span style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:'var(--text-secondary)',flex:1}}>
              ${b.price ? Number(b.price).toFixed(2) : '—'}
            </span>
            <span style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',color:'var(--text-secondary)'}}>
              r={b.rsi?.toFixed(0) ?? '—'}
            </span>
          </div>
        ))}
      </div>

      {/* Hold reason */}
      <div style={{padding:'8px 12px',borderTop:'1px solid var(--border-light)',flexShrink:0}}>
        <div style={{
          fontFamily:'var(--font-mono)',fontSize:'0.62rem',
          color: live?.hold_reason === 'traded' ? 'var(--ok)' : 'var(--text-secondary)',
          fontWeight: live?.hold_reason === 'traded' ? 700 : 400,
        }}>
          {live?.hold_reason === 'traded'
            ? `▶ ACTION: ${live.predicted_direction?.toUpperCase() || 'TRADE'}`
            : `⊘ ${live?.hold_reason || 'HOLD'}`}
        </div>
      </div>
    </div>
  )
}

function Kv({ k, v }) {
  return (
    <div style={{display:'flex',gap:3,alignItems:'baseline'}}>
      <span style={{fontSize:'0.58rem',color:'var(--text-secondary)',fontFamily:'var(--font-mono)'}}>{k}</span>
      <span style={{fontSize:'0.65rem',fontWeight:600,fontFamily:'var(--font-mono)'}}>{v}</span>
    </div>
  )
}
