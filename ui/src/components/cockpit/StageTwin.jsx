import React from 'react';

export function StageTwin({ data }) {
  if (!data || !data.trajectories) return <div>No Twin Data</div>;

  return (
    <div style={{
      border: '1px solid var(--border-soft)',
      borderRadius: 8,
      padding: 16,
      background: 'var(--panel-bg)',
      color: 'var(--text-primary)',
      fontFamily: 'JetBrains Mono, monospace'
    }}>
      <h3 style={{ color: 'var(--twin-color)', marginTop: 0 }}>TWIN (Response Simulator)</h3>
      
      <div style={{ fontSize: 11, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {Object.entries(data.trajectories).map(([id, traj]) => (
          <div key={id} style={{
            display: 'flex',
            justifyContent: 'space-between',
            padding: 8,
            borderRadius: 4,
            border: `1px solid ${traj.selected ? 'var(--accent)' : 'var(--border-soft)'}`,
            background: traj.selected ? 'var(--accent-dim)' : 'transparent',
            opacity: traj.selected ? 1 : 0.5
          }}>
            <span>Hypothesis {id}</span>
            <span>Breached: {traj.breached ? 'yes' : 'no'}</span>
          </div>
        ))}
        
        {data.actual_path && (
          <div style={{ marginTop: 8, padding: 8, background: 'var(--border-soft)', borderRadius: 4 }}>
            <div>Actual Path Tracked</div>
          </div>
        )}
      </div>
    </div>
  );
}
