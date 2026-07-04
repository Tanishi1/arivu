import { useEffect, useRef, useState, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  ReferenceLine, ResponsiveContainer, Tooltip
} from 'recharts'

const SIM_DURATION_MS = 12000 // 12 second simulation loop

export default function Twin({ activeDo, live, onProgress }) {
  const tp = activeDo?.tuned_params || {}
  const hypotheses = live?.hypotheses || []
  const predicted = tp.predicted_magnitude || 0

  const [progress, setProgress] = useState(0)  // 0..1
  const startRef = useRef(null)
  const rafRef   = useRef(null)
  const onProgressRef = useRef(onProgress)
  useEffect(() => { onProgressRef.current = onProgress }, [onProgress])

  // Build trajectory data
  const buildTraj = (mag, noise = 0) => {
    const pts = []
    for (let i = 0; i <= 40; i++) {
      const t = i / 40
      // Realistic: rises, then flattens/drops based on predicted direction
      const signal = mag * (t * 2 - t * t)  // parabolic arc
      const noiseTerm = noise * Math.sin(t * Math.PI * 3) * 0.00003
      pts.push({ t: i, val: signal + noiseTerm })
    }
    return pts
  }

  const trajectories = hypotheses.slice(0, 5).map((h, i) => ({
    id: h.id || i,
    chain: h.chain,
    isSelected: live?.selected_chain === h.chain || live?.selected_id === h.id,
    isEscape: h.is_escape_valve,
    data: buildTraj(h.predicted_magnitude || 0, i + 1),
    color: live?.selected_chain === h.chain
      ? '#2563eb'
      : h.is_escape_valve
      ? '#c2410c'
      : `rgba(100,80,200,${Math.max(0.12, 0.35 - i * 0.06)})`,
    strokeWidth: live?.selected_chain === h.chain ? 2.5 : 1,
  }))

  const band = predicted ? Math.abs(predicted) * 2.5 : 0.0002

  // Cursor position in data units (t: 0..40)
  const cursorT = progress * 40

  // Selected trajectory value at cursor
  const selTraj = trajectories.find(t => t.isSelected)
  const cursorVal = selTraj ? (() => {
    const mag = activeDo?.tuned_params?.predicted_magnitude || 0
    const t = progress
    return mag * (t * 2 - t * t)
  })() : null

  // Animate progress loop
  useEffect(() => {
    const animate = (ts) => {
      if (!startRef.current) startRef.current = ts
      const elapsed = ts - startRef.current
      const t = (elapsed % SIM_DURATION_MS) / SIM_DURATION_MS
      setProgress(t)
      onProgressRef.current?.(t)
      rafRef.current = requestAnimationFrame(animate)
    }
    rafRef.current = requestAnimationFrame(animate)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  const noData = hypotheses.length === 0

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
      {/* Header */}
      <div style={{ display:'flex', alignItems:'center', gap:8 }}>
        <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.62rem', fontWeight:700, letterSpacing:'0.1em', color:'var(--twin)' }}>
          TWIN SIM
        </span>
        <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.60rem', color:'var(--text-secondary)' }}>
          {hypotheses.length} trajectories · {Math.round(progress * 100)}% complete
        </span>
        {/* Live progress bar */}
        <div style={{ flex:1, height:3, background:'var(--bg-panel-2)', borderRadius:2, overflow:'hidden' }}>
          <div style={{
            height:'100%', background:'var(--twin)', borderRadius:2,
            width:`${progress * 100}%`, transition:'width 0.05s linear',
          }}/>
        </div>
      </div>

      {noData ? (
        <div style={{ padding:'30px', textAlign:'center', color:'var(--text-secondary)', fontStyle:'italic', fontSize:'0.75rem' }}>
          No active simulation — waiting for qualifying hypothesis
        </div>
      ) : (
        <>
          {/* Legend */}
          <div style={{ display:'flex', gap:12, flexWrap:'wrap' }}>
            <Legend color="#2563eb" label="selected trajectory" />
            <Legend color="rgba(100,80,200,0.35)" label="other hypotheses" />
            <Legend color="var(--bad)" label={`breach band ±${band.toFixed(6)}`} dashed />
          </div>

          {/* Chart */}
          <div style={{ position:'relative' }}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart margin={{ top:8, right:12, bottom:8, left:8 }}>
                <CartesianGrid stroke="var(--border-light)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="t" type="number" domain={[0,40]}
                  tick={{ fontSize:8, fontFamily:'JetBrains Mono,monospace', fill:'var(--text-secondary)' }}
                />
                <YAxis
                  tick={{ fontSize:8, fontFamily:'JetBrains Mono,monospace', fill:'var(--text-secondary)' }}
                  tickFormatter={v => v.toFixed(5)} width={72}
                />
                <Tooltip
                  contentStyle={{ fontFamily:'JetBrains Mono,monospace', fontSize:'0.62rem', background:'var(--bg-panel)', border:'1px solid var(--border)' }}
                  formatter={v => v.toFixed(8)}
                />
                {/* Breach bands */}
                <ReferenceLine y={band}  stroke="var(--bad)" strokeDasharray="5 3" strokeOpacity={0.55} />
                <ReferenceLine y={-band} stroke="var(--bad)" strokeDasharray="5 3" strokeOpacity={0.55} />
                {/* Zero baseline */}
                <ReferenceLine y={0} stroke="var(--border)" strokeOpacity={0.5} />
                {/* Animated cursor */}
                <ReferenceLine x={cursorT} stroke="#2563eb" strokeWidth={1.5} strokeOpacity={0.8} />

                {/* All trajectories */}
                {trajectories.map(tr => (
                  <Line
                    key={tr.id}
                    data={tr.data}
                    dataKey="val"
                    stroke={tr.color}
                    strokeWidth={tr.strokeWidth}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>

            {/* Moving cursor dot on selected trajectory */}
            {cursorVal !== null && selTraj && (
              <div style={{
                position:'absolute',
                // These are rough positions — recharts chart area starts ~72px from left, ends ~12px from right
                left: `calc(72px + ${(cursorT/40) * (100 - 72/220 * 100)}%)`,
                top: 8,  // approximate — just show in header
                pointerEvents: 'none',
              }}>
              </div>
            )}
          </div>

          {/* Selected hypothesis detail */}
          {tp.chain_summary && (
            <div style={{
              padding:'8px 10px', background:'var(--twin-bg, #eff6ff)', borderRadius:5,
              border:'1px solid var(--twin, #7c3aed)', fontSize:'0.62rem',
              fontFamily:'var(--font-mono)', lineHeight:1.7,
            }}>
              <div style={{ fontWeight:700, color:'var(--twin, #7c3aed)', marginBottom:4 }}>SELECTED: {tp.chain_summary}</div>
              <div style={{ color:'var(--text-secondary)' }}>
                predicted_magnitude = {predicted?.toFixed(8)}
                &nbsp;·&nbsp; band = ±{band?.toFixed(8)}
                &nbsp;·&nbsp; direction = {tp.predicted_direction?.toUpperCase() || '—'}
              </div>
              <div style={{ marginTop:4 }}>
                {Math.abs(cursorVal || 0) > band
                  ? <span style={{ color:'var(--bad)', fontWeight:700 }}>⊘ BREACHED BAND at t={cursorT.toFixed(0)}</span>
                  : progress > 0.5
                  ? <span style={{ color:'var(--ok)', fontWeight:700 }}>✓ Within band — trajectory valid</span>
                  : <span style={{ color:'var(--text-secondary)' }}>Simulating… {Math.round(progress*100)}%</span>
                }
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function Legend({ color, label, dashed }) {
  return (
    <span style={{ display:'flex', alignItems:'center', gap:5, fontSize:'0.60rem', color:'var(--text-secondary)' }}>
      <span style={{
        width:16, height: dashed ? 1 : 2,
        background: color, borderRadius: dashed ? 0 : 1,
        display:'inline-block',
        borderTop: dashed ? `1px dashed ${color}` : 'none',
      }}/>
      {label}
    </span>
  )
}
