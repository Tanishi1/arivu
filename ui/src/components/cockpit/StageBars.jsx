import React, { useState } from 'react';
import { useRegimeContext } from '../../hooks/useRegimeContext';

export function StageBars({ data }) {
  const regime = useRegimeContext();
  const [expanded, setExpanded] = useState(false);

  if (!data) return <div>No Bars Data</div>;

  return (
    <div style={{
      border: '1px solid var(--border-soft)',
      borderRadius: 8,
      padding: 16,
      background: 'var(--panel-bg)',
      color: 'var(--text-primary)',
      fontFamily: 'JetBrains Mono, monospace'
    }}>
      <h3 style={{ color: 'var(--bars-color)', marginTop: 0 }}>BARS</h3>
      <div style={{ marginBottom: 12, fontSize: 12 }}>
        <div>Bars Ready: {data.bars_ready} / {data.required_bars}</div>
        <div>Current Regime: {regime.clusterId || 'N/A'}</div>
      </div>
      
      <div>
        <button 
          onClick={() => setExpanded(!expanded)}
          style={{
            background: 'transparent',
            border: '1px solid var(--border-soft)',
            color: 'var(--text-primary)',
            padding: '4px 8px',
            borderRadius: 4,
            cursor: 'pointer',
            fontSize: 11
          }}
        >
          {expanded ? 'Collapse Features' : 'Expand Features'}
        </button>

        {expanded && (
          <div style={{ marginTop: 12, fontSize: 11 }}>
            {Object.entries(data.feature_groups || {}).map(([group, features]) => (
              <div key={group} style={{ marginBottom: 8 }}>
                <strong style={{ color: 'var(--text-secondary)' }}>{group.toUpperCase()}</strong>
                <div>{features.length} features available</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
