import React from 'react';
import { EscapeValveBranch } from './EscapeValveBranch';

export function StageThesis({ data }) {
  if (!data || !data.hypotheses) return <div>No Thesis Data</div>;

  const getRowStyle = (hypothesis) => {
    if (hypothesis.selected) {
      return { background: 'var(--accent-dim)', border: '1px solid var(--accent)', opacity: 1 };
    }
    return { background: 'transparent', border: '1px solid var(--border-soft)', opacity: 0.5 };
  };

  return (
    <div style={{
      border: '1px solid var(--border-soft)',
      borderRadius: 8,
      padding: 16,
      background: 'var(--panel-bg)',
      color: 'var(--text-primary)',
      fontFamily: 'JetBrains Mono, monospace'
    }}>
      <h3 style={{ color: 'var(--thesis-color)', marginTop: 0 }}>THESIS (Causal Chain)</h3>
      
      <EscapeValveBranch bestScore={data.hypotheses[0]?.score} requiredScore={0.6} />

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {data.hypotheses.map(hyp => (
          <div key={hyp.id} style={{
            ...getRowStyle(hyp),
            padding: 8,
            borderRadius: 4,
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
            fontSize: 11
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: hyp.selected ? 700 : 400 }}>
              <span>{hyp.id} (Rank: {hyp.rank})</span>
              <span>Score: {hyp.score.toFixed(2)}</span>
            </div>
            
            <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
              {hyp.chain.map((link, idx) => (
                <React.Fragment key={idx}>
                  <span>{link.source}</span>
                  <span style={{ color: 'var(--accent)' }}>→</span>
                  {idx === hyp.chain.length - 1 && <span>{link.target}</span>}
                </React.Fragment>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
