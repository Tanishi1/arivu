import React, { useState } from 'react';
import { getEmphasisStyle } from '../../../utils/hypothesisEmphasis';

export function StageGraph({ data }) {
  const [filter, setFilter] = useState('All');
  const filters = ['All', 'Macro', 'Market', 'Microstructure', 'Target', 'Hypotheses', 'Best Only'];

  if (!data || !data.nodes) return <div>No Graph Data</div>;

  return (
    <div style={{
      border: '1px solid var(--border-soft)',
      borderRadius: 8,
      padding: 16,
      background: 'var(--panel-bg)',
      color: 'var(--text-primary)',
      fontFamily: 'JetBrains Mono, monospace'
    }}>
      <h3 style={{ color: 'var(--graph-color)', marginTop: 0 }}>GRAPH (Causal State)</h3>
      
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 16 }}>
        {filters.map(f => (
          <button 
            key={f} 
            onClick={() => setFilter(f)}
            style={{
              padding: '4px 8px',
              fontSize: 10,
              borderRadius: 12,
              background: filter === f ? 'var(--accent)' : 'var(--border-soft)',
              color: filter === f ? 'var(--text-on-accent)' : 'var(--text-secondary)',
              border: 'none',
              cursor: 'pointer'
            }}
          >
            {f}
          </button>
        ))}
      </div>

      <div style={{ 
        width: '100%', 
        height: 300, 
        background: 'var(--bg-base)', 
        border: '1px solid var(--border-soft)', 
        borderRadius: 4,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column'
      }}>
        <div style={{ color: 'var(--text-muted)' }}>[Layered Compound Causal Graph Canvas]</div>
        <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 8 }}>
          Nodes: {data.nodes.length} | Edges: {data.edges?.length || 0}
        </div>
      </div>
    </div>
  );
}
