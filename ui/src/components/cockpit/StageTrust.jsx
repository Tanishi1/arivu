import React from 'react';

export function StageTrust({ data }) {
  if (!data) return <div>No Trust Data</div>;

  return (
    <div style={{
      border: '1px solid var(--border-soft)',
      borderRadius: 8,
      padding: 16,
      background: 'var(--panel-bg)',
      color: 'var(--text-primary)',
      fontFamily: 'JetBrains Mono, monospace'
    }}>
      <h3 style={{ color: 'var(--trust-color)', marginTop: 0 }}>LAYER 2 TRUST</h3>
      
      <div style={{ fontSize: 12, display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div>chain: {data.chain_id}</div>
        <div>before: {data.before?.toFixed(2)}</div>
        <div>after: {data.after?.toFixed(2)}</div>
        <div style={{ color: data.delta > 0 ? 'var(--success)' : 'var(--error)' }}>
          delta: {data.delta > 0 ? '+' : ''}{data.delta?.toFixed(2)}
        </div>
        <div>reason: {data.reason}</div>
      </div>
    </div>
  );
}
