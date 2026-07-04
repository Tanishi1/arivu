import { useRef, useEffect, useState } from 'react'

export default function Feed({ live, ticks = [], tickCount = 0 }) {
  const streamRef = useRef(null)
  const [latestFlash, setLatestFlash] = useState(false)

  // Flash on new tick
  const prevCount = useRef(tickCount)
  useEffect(() => {
    if (tickCount !== prevCount.current) {
      prevCount.current = tickCount
      setLatestFlash(true)
      setTimeout(() => setLatestFlash(false), 300)
    }
  }, [tickCount])

  // Auto-scroll tick stream
  useEffect(() => {
    if (streamRef.current) streamRef.current.scrollTop = 0
  }, [ticks.length])

  const ok = live?.ts && (Date.now() - new Date(live.ts).getTime()) < 30000

  return (
    <div className="stage-panel fade-in">
      <div className="stage-hdr">
        <span className="stage-tag t-feed">FEED</span>
        <span className="stage-title">Market Feed — Binance SOL/USDT</span>
        <span className="stage-meta" style={{ display:'flex', alignItems:'center', gap:6 }}>
          <span className={`pulse-dot ${ok ? '' : 'stale'}`} />
          {ok ? `LIVE · ${tickCount} ticks received` : 'STALE'}
        </span>
      </div>

      {!live ? (
        <div className="empty"><div className="empty-icon">○</div><p>Waiting for first feed tick from JSONL WebSocket…</p></div>
      ) : (
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14 }}>
          {/* Current snapshot */}
          <div>
            <div style={{ fontSize:'0.65rem', fontWeight:700, letterSpacing:'0.08em', color:'var(--text-secondary)', marginBottom:8 }}>CURRENT SNAPSHOT</div>
            <div className="metric-grid">
              <M label="SOL/USDT"   val={live.price ? `$${Number(live.price).toFixed(4)}` : '—'} />
              <M label="Volatility" val={live.volatility?.toFixed(6) ?? '—'} />
              <M label="Spread"     val={live.spread?.toFixed(6) ?? '—'} />
              <M label="RSI"        val={live.rsi?.toFixed(2) ?? '—'}
                 cls={live.rsi > 70 ? 'warn' : live.rsi < 30 ? 'ok' : ''} />
              <M label="Graph edges" val={live.graph_edges ?? '—'} />
              <M label="Hypotheses" val={live.n_hypotheses ?? '—'} />
              <M label="Best score" val={live.best_score?.toFixed(4) ?? '—'} />
              <M label="Hold reason" val={live.hold_reason ?? '—'} />
            </div>
          </div>

          {/* Live tick stream */}
          <div>
            <div style={{ fontSize:'0.65rem', fontWeight:700, letterSpacing:'0.08em', color:'var(--text-secondary)', marginBottom:8 }}>
              INCOMING TICKS (last {ticks.length})
            </div>
            <div
              ref={streamRef}
              style={{ height:240, overflowY:'auto', border:'1px solid var(--border-light)', borderRadius:6, background:'var(--bg-panel)' }}
            >
              {ticks.length === 0 ? (
                <div style={{ padding:'20px', color:'var(--text-secondary)', fontSize:'0.72rem', textAlign:'center' }}>
                  Waiting for WebSocket ticks…
                </div>
              ) : ticks.map((t, i) => (
                <div
                  key={i}
                  className={i === 0 && latestFlash ? 'tick-flash' : ''}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    padding: '4px 10px',
                    borderBottom: '1px solid var(--border-light)',
                    opacity: Math.max(0.3, 1 - i * 0.04),
                    background: i === 0 ? 'var(--feed-bg)' : 'transparent',
                    transition: 'background 0.3s',
                  }}
                >
                  <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.62rem', color:'var(--text-secondary)', width:68, flexShrink:0 }}>
                    {t.ts ? new Date(t.ts).toLocaleTimeString() : '—'}
                  </span>
                  <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.70rem', fontWeight:600, width:72, flexShrink:0 }}>
                    ${t.price ? Number(t.price).toFixed(4) : '—'}
                  </span>
                  <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.62rem', color:'var(--text-secondary)', flex:1, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                    {t.hold_reason === 'traded'
                      ? <span style={{ color:'var(--ok)', fontWeight:700 }}>🟢 TRADE · {t.selected_chain}</span>
                      : t.hold_reason || '—'}
                  </span>
                  {t.n_hypotheses > 0 && (
                    <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.60rem', color:'var(--thesis)' }}>
                      {t.n_hypotheses}h
                    </span>
                  )}
                </div>
              ))}
            </div>
            {live.ts && (
              <p style={{ fontSize:'0.62rem', color:'var(--text-secondary)', fontFamily:'var(--font-mono)', marginTop:5 }}>
                Last tick: {new Date(live.ts).toLocaleTimeString()}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function M({ label, val, cls }) {
  return (
    <div className="metric-cell">
      <div className="metric-label">{label}</div>
      <div className={`metric-val ${cls || ''}`}>{val}</div>
    </div>
  )
}
