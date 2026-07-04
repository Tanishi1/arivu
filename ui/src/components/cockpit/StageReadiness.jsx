import React from 'react';

export function StageReadiness({ data }) {
  if (!data) return <div>No Readiness Data</div>;

  return (
    <div style={{
      border: '1px solid var(--border-soft)',
      borderRadius: 8,
      padding: 16,
      background: 'var(--panel-bg)',
      color: 'var(--text-primary)',
      fontFamily: 'JetBrains Mono, monospace'
    }}>
      <h3 style={{ marginTop: 0 }}>READINESS</h3>
      
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>ML1 VERDICT: {data.ml1_verdict}</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>normal</span><span>{data.probabilities?.normal?.toFixed(2)}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>stressed</span><span>{data.probabilities?.stressed?.toFixed(2)}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>degraded</span><span>{data.probabilities?.degraded?.toFixed(2)}</span>
          </div>
        </div>
      </div>

      <div style={{ paddingTop: 16, borderTop: '1px solid var(--border-soft)' }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>ML2 BREACH RISK: {data.ml2_breach_risk?.toFixed(2)}</div>
        <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
          Risk verdict: {data.ml2_breach_risk > 0.5 ? 'high' : 'acceptable'}
        </div>
      </div>
    </div>
  );
}
