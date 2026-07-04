import { useState, useEffect } from 'react'
import { NavLink } from 'react-router-dom'

const NAV = [
  { to: '/',         label: 'Cockpit' },
  { to: '/compare',  label: 'Comparison' },
  { to: '/ledger',   label: 'Ledger' },
  { to: '/research', label: 'Research' },
]

const ARM_COLORS = { CausalAgent:'#2563eb', ppo_standard:'#7c3aed', ppo_causal_feature:'#9333ea' }

export default function LedgerPage() {
  const [data, setData] = useState(null)
  const [strategy, setStrategy] = useState('CausalAgent')

  useEffect(() => {
    fetch(`/api/ledger?limit=50&strategy=${strategy}`)
      .then(r => r.json()).then(setData)
  }, [strategy])

  const decisions = data?.decisions || []
  const closed    = decisions.filter(d => d.outcome)
  const wins      = closed.filter(d => (d.outcome?.actual_pnl||0) > 0)
  const totalPnl  = closed.reduce((s,d) => s+(d.outcome?.actual_pnl||0), 0)
  const winRate   = closed.length > 0 ? (wins.length/closed.length*100).toFixed(0) : null

  return (
    <div className="shell">
      <nav className="top-nav">
        <span className="nav-brand">ARIVU</span>
        <div className="nav-tabs">
          {NAV.map(({to,label}) => (
            <NavLink key={to} to={to} end={to==='/'} className={({isActive})=>`nav-tab${isActive?' active':''}`}>{label}</NavLink>
          ))}
        </div>
        <div className="nav-right">
          <div style={{ display:'flex', gap:5 }}>
            {['CausalAgent','ppo_standard','ppo_causal_feature','all'].map(s => (
              <button key={s} onClick={()=>setStrategy(s)} style={{
                padding:'3px 10px',borderRadius:4,fontSize:'0.68rem',fontWeight:600,cursor:'pointer',
                border:`1px solid ${strategy===s?(ARM_COLORS[s]||'var(--graph)'):'var(--border)'}`,
                background: strategy===s ? (ARM_COLORS[s]||'var(--graph)') : '#fff',
                color: strategy===s ? '#fff' : 'var(--text-secondary)',
                transition:'all 0.12s',
              }}>{s === 'all' ? 'All' : s.replace('ppo_','PPO-').replace('CausalAgent','Causal')}</button>
            ))}
          </div>
        </div>
      </nav>

      {/* Stats summary bar */}
      {decisions.length > 0 && (
        <div style={{display:'flex',gap:20,padding:'8px 20px',background:'var(--bg-panel)',borderBottom:'1px solid var(--border-light)',flexShrink:0}}>
          <Stat label="Showing" val={decisions.length} />
          <Stat label="Closed" val={closed.length} />
          <Stat label="Win Rate" val={winRate!=null?`${winRate}%`:'—'} color={winRate>50?'var(--ok)':winRate<40?'var(--bad)':null}/>
          <Stat label="Total PnL" val={`$${totalPnl.toFixed(2)}`} color={totalPnl>=0?'var(--ok)':'var(--bad)'}/>
        </div>
      )}

      <div style={{ flex:1, overflowY:'auto', padding:'14px 20px' }}>
        <div style={{ overflowX:'auto' }}>
          <table className="ltable" style={{ width:'100%' }}>
            <thead>
              <tr>
                <th>ID</th><th>Strategy</th><th>Time</th><th>Action</th>
                <th>Chain / Signal</th><th>Status</th><th>Phase</th>
                <th>Score</th><th>PnL</th><th>Close Reason</th>
              </tr>
            </thead>
            <tbody>
              {decisions.map(d => {
                const tp  = d.tuned_params || {}
                const out = d.outcome
                const action = d.strategy_name === 'CausalAgent'
                  ? (tp.predicted_direction === 'up' ? 'BUY' : 'SELL')
                  : d.status
                const pnl = out?.actual_pnl || 0
                return (
                  <tr key={d.id} style={{borderLeft:`3px solid ${ARM_COLORS[d.strategy_name]||'transparent'}`}}>
                    <td style={{color:'var(--text-secondary)'}}>{d.id?.slice(0,8)}</td>
                    <td>
                      <span style={{
                        background:ARM_COLORS[d.strategy_name]||'var(--graph)',color:'#fff',
                        fontFamily:'var(--font-mono)',fontSize:'0.58rem',fontWeight:700,
                        padding:'1px 6px',borderRadius:3,
                      }}>{d.strategy_name?.replace('ppo_','PPO-').replace('CausalAgent','ARIVU')}</span>
                    </td>
                    <td>{d.timestamp_committed ? new Date(d.timestamp_committed).toLocaleTimeString() : '—'}</td>
                    <td style={{ color: action==='BUY'?'var(--ok)':action==='SELL'?'var(--bad)':'var(--text-secondary)', fontWeight:700 }}>{action}</td>
                    <td style={{ maxWidth:200, overflow:'hidden', textOverflow:'ellipsis', color:'var(--text-secondary)' }}>{tp.chain_summary || '—'}</td>
                    <td><span className={`sbadge ${d.status}`}>{d.status}</span></td>
                    <td style={{color:'var(--text-secondary)'}}>{d.phase}</td>
                    <td style={{fontFamily:'var(--font-mono)'}}>{tp.composite_score?.toFixed(4) ?? '—'}</td>
                    <td style={{ color: out ? (pnl>=0?'var(--ok)':'var(--bad)') : 'var(--text-secondary)', fontWeight: out ? 700 : 400 }}>
                      {out ? `${pnl>=0?'+':''}$${pnl.toFixed(2)}` : '—'}
                    </td>
                    <td style={{ color:'var(--text-secondary)' }}>{out?.close_reason || '—'}</td>
                  </tr>
                )
              })}
              {decisions.length === 0 && (
                <tr><td colSpan={10} style={{ textAlign:'center', padding:48, color:'var(--text-secondary)' }}>
                  No decisions recorded yet
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, val, color }) {
  return (
    <div style={{display:'flex',alignItems:'center',gap:6}}>
      <span style={{fontSize:'0.62rem',color:'var(--text-secondary)'}}>{label}:</span>
      <span style={{fontFamily:'var(--font-mono)',fontSize:'0.72rem',fontWeight:700,color:color||'var(--text-primary)'}}>{val}</span>
    </div>
  )
}
