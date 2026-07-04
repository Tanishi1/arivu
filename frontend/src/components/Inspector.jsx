export default function Inspector({ state, graph, live }) {
  const vec = state?.active_do?.algo_health_vector || []
  const tp  = state?.active_do?.tuned_params || {}
  const feedOk = live?.ts && (Date.now() - new Date(live?.ts || 0).getTime()) < 30000

  return (
    <aside className="inspector">
      <div className="insp-hdr">
        INSPECTOR
        <span className={`pulse-dot ${feedOk ? '' : 'stale'}`} style={{ marginLeft:'auto' }} />
      </div>
      <div className="insp-body">

        <div className="insp-sec">
          <h4>Live Feed</h4>
          <R label="Price"       val={live?.price ? `$${Number(live.price).toFixed(4)}` : '—'} />
          <R label="Vol"         val={live?.volatility?.toFixed(6) ?? '—'} />
          <R label="Spread"      val={live?.spread?.toFixed(6) ?? '—'} />
          <R label="RSI"         val={live?.rsi?.toFixed(2) ?? '—'} />
          <R label="Hold"        val={live?.hold_reason || '—'} />
          <R label="Best score"  val={live?.best_score?.toFixed(4) ?? '—'} />
        </div>
        <div className="insp-div" />

        <div className="insp-sec">
          <h4>Graph</h4>
          <R label="Version"   val={state?.graph_version || '—'} />
          <R label="Algorithm" val={graph?.algorithm || '—'} />
          <R label="Edges"     val={graph?.edge_count ?? '—'} />
          <R label="Bars used" val={graph?.n_bars_used ?? '—'} />
          <R label="Validated" val={graph?.edges?.filter(e => e.validated).length ?? '—'} />
        </div>
        <div className="insp-div" />

        {state?.active_do && (
          <>
            <div className="insp-sec">
              <h4>Active Decision</h4>
              <R label="ID"        val={state.active_do.id?.slice(0,8)} />
              <R label="Status"    val={state.active_do.status} />
              <R label="Chain"     val={tp.chain_summary || '—'} />
              <R label="Direction" val={tp.predicted_direction?.toUpperCase() || '—'} />
              <R label="L1 score"  val={tp.layer1_score?.toFixed(4) ?? '—'} />
              <R label="L2 score"  val={tp.layer2_score?.toFixed(4) ?? '—'} />
            </div>
            <div className="insp-div" />
          </>
        )}

        <div className="insp-sec">
          <h4>ML1 Health</h4>
          {Array.isArray(vec) && vec.length === 3 ? (
            <>
              <Pb label="normal"   val={vec[0]} color="var(--ok)" />
              <Pb label="stressed" val={vec[1]} color="var(--warn)" />
              <Pb label="degraded" val={vec[2]} color="var(--bad)" />
            </>
          ) : <p style={{ fontSize:'0.68rem', color:'var(--text-secondary)' }}>No vector yet</p>}
        </div>

        {(live?.hypotheses?.length > 0) && (
          <>
            <div className="insp-div" />
            <div className="insp-sec">
              <h4>Hypotheses ({live.hypotheses.length})</h4>
              {live.hypotheses.slice(0, 4).map((h, i) => (
                <R key={i} label={`H${i+1} ${h.chain}`} val={h.composite_score?.toFixed(4)} />
              ))}
            </div>
          </>
        )}

      </div>
    </aside>
  )
}

function R({ label, val }) {
  return (
    <div className="insp-row">
      <span className="insp-label">{label}</span>
      <span className="insp-val">{String(val ?? '—')}</span>
    </div>
  )
}

function Pb({ label, val, color }) {
  const pct = Math.round((val || 0) * 100)
  return (
    <div className="prob-strip">
      <span className="prob-label">{label}</span>
      <div className="prob-track"><div className="prob-fill" style={{ width:`${pct}%`, background:color }} /></div>
      <span className="prob-pct">{pct}%</span>
    </div>
  )
}
