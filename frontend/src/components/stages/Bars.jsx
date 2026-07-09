import { useState, useEffect, useRef } from 'react'

// Exact 19 variable names from feature_bar.py — in pipeline order
const VARIABLE_NAMES = [
  'price_return', 'volume', 'spread', 'volatility', 'rsi',
  'trade_intensity', 'order_book_imbalance', 'ema_spread',
  'bollinger_width', 'price_in_band', 'regime_volatile',
  'regime_trending', 'btc_return', 'eth_return',
  'algo_health_p_normal', 'algo_health_p_stressed', 'algo_health_p_degraded',
  'session_sin', 'session_cos',
]

const VAR_GROUPS = [
  { label: 'Target',  color: '#16a34a', vars: ['price_return'] },
  { label: 'Market',  color: '#4f46e5', vars: ['volume','spread','volatility','rsi','trade_intensity','ema_spread','bollinger_width','price_in_band'] },
  { label: 'Micro',   color: '#2563eb', vars: ['order_book_imbalance'] },
  { label: 'Regime',  color: '#0891b2', vars: ['regime_volatile','regime_trending'] },
  { label: 'Macro',   color: '#7c3aed', vars: ['btc_return','eth_return','session_sin','session_cos'] },
  { label: 'Health',  color: '#475569', vars: ['algo_health_p_normal','algo_health_p_stressed','algo_health_p_degraded'] },
]

// Pretty display names for the 19 variables
const VAR_LABEL = {
  price_return:          'price_ret',
  volume:                'volume',
  spread:                'spread',
  volatility:            'volatility',
  rsi:                   'RSI',
  trade_intensity:       'trade_int',
  order_book_imbalance:  'ob_imb',
  ema_spread:            'ema_sp',
  bollinger_width:       'bol_width',
  price_in_band:         'in_band',
  regime_volatile:       'reg_vol',
  regime_trending:       'reg_trend',
  btc_return:            'btc_ret',
  eth_return:            'eth_ret',
  algo_health_p_normal:  'hlth_norm',
  algo_health_p_stressed:'hlth_stress',
  algo_health_p_degraded:'hlth_deg',
  session_sin:           'sess_sin',
  session_cos:           'sess_cos',
}

// Milestones in bar warmup
const MILESTONES = [
  { at: 40,  label: 'Granger @ 40',  desc: 'Granger pre-screen unlocked' },
  { at: 60,  label: 'PCMCI @ 60',    desc: 'First PCMCI run possible' },
  { at: 200, label: 'Full @ 200',     desc: 'Full rolling window' },
]

export default function Bars({ barsReady, live }) {
  const total = 200
  const pct   = barsReady ? Math.min(100, Math.round((barsReady / total) * 100)) : 0

  // Rolling price sparkline — updated each time live changes
  const [priceHistory, setPriceHistory] = useState([])
  const [barEvents,    setBarEvents]    = useState([])  // timestamped bar-close events
  const prevBars  = useRef(barsReady ?? 0)
  const prevPrice = useRef(null)

  useEffect(() => {
    if (!live?.price) return
    const p = Number(live.price)

    // Add to sparkline (keep last 40 points)
    setPriceHistory(h => [...h, p].slice(-40))

    // Detect a new bar closed: barsReady incremented
    const curBars = barsReady ?? 0
    if (curBars > prevBars.current) {
      prevBars.current = curBars
      setBarEvents(ev => [{
        barN:  curBars,
        price: p,
        ret:   prevPrice.current != null ? ((p - prevPrice.current) / prevPrice.current) : null,
        rsi:   live.rsi,
        vol:   live.volatility,
        ts:    new Date().toLocaleTimeString(),
      }, ...ev].slice(0, 8))
    }
    prevPrice.current = p
  }, [live, barsReady])

  // Build feature snapshot from live state — used for the heatmap
  const featureSnap = live ? {
    price_return:           live.price_return          ?? null,
    volume:                 live.volume                ?? null,
    spread:                 live.spread                ?? null,
    volatility:             live.volatility            ?? null,
    rsi:                    live.rsi                   ?? null,
    trade_intensity:        live.trade_intensity       ?? null,
    order_book_imbalance:   live.order_book_imbalance  ?? null,
    ema_spread:             live.ema_spread            ?? null,
    bollinger_width:        live.bollinger_width        ?? null,
    price_in_band:          live.price_in_band         ?? null,
    regime_volatile:        live.regime_volatile       ?? null,
    regime_trending:        live.regime_trending        ?? null,
    btc_return:             live.btc_return            ?? null,
    eth_return:             live.eth_return            ?? null,
    algo_health_p_normal:   live.algo_health_p_normal  ?? null,
    algo_health_p_stressed: live.algo_health_p_stressed?? null,
    algo_health_p_degraded: live.algo_health_p_degraded?? null,
    session_sin:            live.session_sin           ?? null,
    session_cos:            live.session_cos           ?? null,
  } : {}

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:16 }}>

      {/* ── Warmup progress ── */}
      <div>
        <div style={{ display:'flex', justifyContent:'space-between', marginBottom:5 }}>
          <span style={{ fontSize:'0.62rem', color:'var(--text-secondary)' }}>Warm-up progress</span>
          <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.65rem', fontWeight:700 }}>
            {barsReady ?? 0} / {total} bars ({pct}%)
          </span>
        </div>
        <div style={{ height:6, background:'var(--bg-panel-2)', borderRadius:3, overflow:'hidden' }}>
          <div style={{
            height:'100%', borderRadius:3, transition:'width 0.6s',
            background: pct >= 40 ? 'var(--ok)' : 'var(--warn)',
            width:`${pct}%`,
          }}/>
        </div>
        <div style={{ display:'flex', gap:16, marginTop:6 }}>
          {MILESTONES.map(m => (
            <div key={m.at} style={{ display:'flex', alignItems:'center', gap:4 }}>
              <span style={{
                fontFamily:'var(--font-mono)', fontSize:'0.70rem', fontWeight:700,
                color: (barsReady ?? 0) >= m.at ? 'var(--ok)' : 'var(--text-tertiary)',
              }}>
                {(barsReady ?? 0) >= m.at ? '✓' : '○'}
              </span>
              <span style={{
                fontFamily:'var(--font-mono)', fontSize:'0.58rem',
                color: (barsReady ?? 0) >= m.at ? 'var(--ok)' : 'var(--text-secondary)',
              }}>
                {m.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Adaptive PCMCI Refresh strip ── */}
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
          <KV label="τ_max"    val={live?.tau_max       != null ? `${live.tau_max} bars` : '— bars'} color="#7c3aed" />
          <KV label="α"        val={live?.pcmci_alpha   != null ? live.pcmci_alpha.toFixed(3)    : '—'} color="#7c3aed" />
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
          {live?.adaptive_fast ? 'Vol > median → refresh halved' : 'Vol ≤ median → full interval'}
        </div>
      </div>

      {/* ── Price micro-chart ── */}
      <div>
        <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.08em',color:'var(--text-secondary)',marginBottom:6}}>
          PRICE TRACE (last {priceHistory.length} samples)
        </div>
        <PriceSparkline prices={priceHistory} currentPrice={live?.price} />
      </div>

      {/* ── Bar close events ── */}
      <div>
        <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.08em',color:'var(--text-secondary)',marginBottom:7}}>
          BAR CLOSE EVENTS
        </div>
        {barEvents.length === 0 ? (
          <div style={{
            padding:'10px 12px', borderRadius:6,
            background:'var(--bg-panel-2)', border:'1px solid var(--border-light)',
            fontFamily:'var(--font-mono)', fontSize:'0.62rem', color:'var(--text-secondary)',
            textAlign:'center',
          }}>
            Waiting for bar close ({barsReady ?? 0} bars so far)…
          </div>
        ) : (
          <div style={{display:'flex',flexDirection:'column',gap:2}}>
            {barEvents.map((b, i) => {
              const retPct = b.ret != null ? (b.ret * 100) : null
              const retColor = retPct == null ? 'var(--text-secondary)' : retPct > 0 ? 'var(--ok)' : retPct < 0 ? 'var(--bad)' : 'var(--text-secondary)'
              return (
                <div key={i} style={{
                  display:'grid',
                  gridTemplateColumns:'28px 1fr 60px 52px 52px 52px',
                  gap:6, alignItems:'center',
                  padding:'4px 8px',
                  borderRadius:4,
                  background: i === 0 ? 'var(--bars-bg,#eef2ff)' : 'var(--bg-panel)',
                  border:`1px solid ${i === 0 ? 'var(--bars,#4f46e5)' : 'var(--border-light)'}`,
                  opacity: 1 - i * 0.1,
                  transition:'all 0.2s',
                }}>
                  <span style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',fontWeight:700,color:'var(--bars,#4f46e5)'}}>
                    B{b.barN}
                  </span>
                  <span style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',color:'var(--text-secondary)'}}>
                    {b.ts}
                  </span>
                  <span style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',fontWeight:600}}>
                    ${b.price ? Number(b.price).toFixed(3) : '—'}
                  </span>
                  <span style={{fontFamily:'var(--font-mono)',fontSize:'0.60rem',color:retColor,fontWeight:retPct != null ? 600 : 400}}>
                    {retPct != null ? `${retPct > 0 ? '+' : ''}${retPct.toFixed(3)}%` : '—'}
                  </span>
                  <span style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',color:'var(--text-secondary)'}}>
                    RSI {b.rsi != null ? b.rsi.toFixed(0) : '—'}
                  </span>
                  <span style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',color:'var(--text-secondary)'}}>
                    σ {b.vol != null ? b.vol.toFixed(5) : '—'}
                  </span>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* ── Feature heatmap — live 19-variable snapshot ── */}
      <div>
        <div style={{
          display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:7,
        }}>
          <div style={{fontSize:'0.60rem',fontWeight:700,letterSpacing:'0.08em',color:'var(--text-secondary)'}}>
            FEATURE VECTOR — {VARIABLE_NAMES.length} variables (live snapshot)
          </div>
          {live && (
            <span style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',color:'var(--text-secondary)'}}>
              {Object.values(featureSnap).filter(v => v != null).length} / {VARIABLE_NAMES.length} populated
            </span>
          )}
        </div>

        {!live ? (
          <div style={{
            padding:'12px', borderRadius:6,
            background:'var(--bg-panel-2)', border:'1px solid var(--border-light)',
            fontFamily:'var(--font-mono)', fontSize:'0.62rem', color:'var(--text-secondary)', textAlign:'center',
          }}>
            Waiting for first feed update…
          </div>
        ) : (
          <div style={{display:'flex',flexDirection:'column',gap:10}}>
            {VAR_GROUPS.map(g => (
              <div key={g.label}>
                <div style={{
                  fontSize:'0.55rem',fontWeight:700,letterSpacing:'0.08em',
                  color:g.color,marginBottom:4,
                }}>
                  {g.label}
                </div>
                <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(120px,1fr))',gap:4}}>
                  {g.vars.map(v => (
                    <FeatureCell
                      key={v}
                      name={VAR_LABEL[v] ?? v}
                      val={featureSnap[v]}
                      color={g.color}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

    </div>
  )
}

// ── Price sparkline chart ─────────────────────────────────────────────────────
function PriceSparkline({ prices, currentPrice }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || prices.length < 2) return
    const ctx    = canvas.getContext('2d')
    const W      = canvas.width
    const H      = canvas.height

    ctx.clearRect(0, 0, W, H)

    const mn = Math.min(...prices)
    const mx = Math.max(...prices)
    const rng = mx - mn || 1

    const xStep = W / (prices.length - 1)
    const pts   = prices.map((p, i) => ({
      x: i * xStep,
      y: H - 4 - ((p - mn) / rng) * (H - 8),
    }))

    // Gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, H)
    grad.addColorStop(0,   '#4f46e530')
    grad.addColorStop(1,   '#4f46e500')
    ctx.beginPath()
    ctx.moveTo(pts[0].x, H)
    pts.forEach(p => ctx.lineTo(p.x, p.y))
    ctx.lineTo(pts[pts.length - 1].x, H)
    ctx.closePath()
    ctx.fillStyle = grad
    ctx.fill()

    // Line
    ctx.beginPath()
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y))
    ctx.strokeStyle = '#4f46e5'
    ctx.lineWidth   = 1.5
    ctx.stroke()

    // Last point dot
    const last = pts[pts.length - 1]
    ctx.beginPath()
    ctx.arc(last.x, last.y, 3, 0, Math.PI * 2)
    ctx.fillStyle = '#4f46e5'
    ctx.fill()

    // Price labels
    ctx.font      = '600 8px JetBrains Mono, monospace'
    ctx.fillStyle = '#64748b'
    ctx.textAlign = 'left'
    ctx.fillText(`$${mn.toFixed(3)}`, 2, H - 2)
    ctx.fillText(`$${mx.toFixed(3)}`, 2, 10)

  }, [prices])

  if (prices.length < 2) {
    return (
      <div style={{
        height:60, borderRadius:6, background:'var(--bg-panel-2)',
        border:'1px solid var(--border-light)',
        display:'flex', alignItems:'center', justifyContent:'center',
        fontFamily:'var(--font-mono)', fontSize:'0.60rem', color:'var(--text-secondary)',
      }}>
        {currentPrice ? `$${Number(currentPrice).toFixed(4)} — collecting samples…` : 'No price data yet'}
      </div>
    )
  }

  return (
    <div style={{ position:'relative' }}>
      <canvas
        ref={canvasRef}
        width={600}
        height={60}
        style={{ width:'100%', height:60, borderRadius:6,
          background:'var(--bg-panel-2)', border:'1px solid var(--border-light)' }}
      />
      {currentPrice && (
        <div style={{
          position:'absolute', top:4, right:8,
          fontFamily:'var(--font-mono)', fontSize:'0.72rem', fontWeight:700,
          color:'#4f46e5',
        }}>
          ${Number(currentPrice).toFixed(4)}
        </div>
      )}
    </div>
  )
}

// ── Feature value cell with bar indicator ─────────────────────────────────────
function FeatureCell({ name, val, color }) {
  const isNull = val == null

  // Normalise value for the bar — cap display at ±3 (handles Z-scored values and bounded [0,1] values)
  const norm = isNull ? 0 : Math.min(1, Math.abs(val) / 3)
  const isNeg = !isNull && val < 0

  // Format value for display
  let display = '—'
  if (!isNull) {
    const abs = Math.abs(val)
    if (abs < 0.0001 && val !== 0) display = val.toExponential(2)
    else if (abs < 0.01)           display = val.toFixed(5)
    else if (abs < 1)              display = val.toFixed(4)
    else if (abs < 100)            display = val.toFixed(2)
    else                           display = val.toFixed(0)
  }

  return (
    <div style={{
      padding:'4px 7px', borderRadius:5,
      background: isNull ? 'var(--bg-panel)' : `${color}08`,
      border:`1px solid ${isNull ? 'var(--border-light)' : color+'22'}`,
    }}>
      {/* Name */}
      <div style={{
        fontFamily:'var(--font-mono)', fontSize:'0.55rem',
        color: isNull ? 'var(--text-tertiary)' : color,
        fontWeight: 600, letterSpacing:'0.02em', marginBottom:2,
        overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap',
      }}>
        {name}
      </div>
      {/* Value */}
      <div style={{
        fontFamily:'var(--font-mono)', fontSize:'0.68rem', fontWeight:700,
        color: isNull ? 'var(--text-tertiary)' :
               name === 'RSI' ? (val > 70 ? 'var(--bad)' : val < 30 ? 'var(--ok)' : 'var(--text-primary)') :
               isNeg ? 'var(--bad)' : 'var(--text-primary)',
      }}>
        {display}
      </div>
      {/* Mini bar */}
      {!isNull && (
        <div style={{ marginTop:3, height:2, background:'var(--border-light)', borderRadius:1, overflow:'hidden' }}>
          <div style={{
            height:'100%', borderRadius:1,
            background: isNeg ? 'var(--bad)' : color,
            width:`${Math.round(norm * 100)}%`,
            transition:'width 0.4s ease',
          }}/>
        </div>
      )}
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
