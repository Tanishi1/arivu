import React from 'react';
import { EscapeValveMemory } from '../EscapeValveMemory';

function CandidateCard({ candidate }) {
  const isActive = candidate.status === 'active';
  const isStagnating = candidate.status === 'stagnating';
  
  return (
    <div style={{
      border: `1px solid ${isActive ? 'var(--accent)' : isStagnating ? 'var(--warning)' : 'var(--border-soft)'}`,
      background: isActive ? 'var(--accent-dim)' : 'transparent',
      opacity: isActive ? 1 : 0.6,
      padding: 12,
      borderRadius: 6,
      fontSize: 11
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 700, marginBottom: 8 }}>
        <span>Candidate {candidate.id}</span>
        <span style={{ color: isActive ? 'var(--accent)' : isStagnating ? 'var(--warning)' : 'inherit' }}>
          {candidate.status}
        </span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
        <span>k_runs: {candidate.k_runs}</span>
        <span>min_runs: {candidate.min_runs}</span>
        <span>threshold: {candidate.threshold?.toFixed(2)}</span>
        <span>score: {candidate.score?.toFixed(2)}</span>
      </div>
      
      {isStagnating && (
        <div style={{ marginTop: 8, padding: 4, background: 'var(--warning-bg)', color: 'var(--warning)', borderRadius: 4 }}>
          STAGNATION MONITOR: {candidate.stagnation_counter}
          <div style={{ fontSize: 10 }}>near replacement</div>
        </div>
      )}
    </div>
  );
}

export function StageOptimizer({ data }) {
  if (!data || !data.candidates) return <div>No Optimizer Data</div>;

  return (
    <div style={{
      border: '1px solid var(--border-soft)',
      borderRadius: 8,
      padding: 16,
      background: 'var(--panel-bg)',
      color: 'var(--text-primary)',
      fontFamily: 'JetBrains Mono, monospace'
    }}>
      <h3 style={{ color: 'var(--optimizer-color)', marginTop: 0 }}>META PARAMETER OPTIMIZER</h3>
      
      <div style={{ marginBottom: 12, fontSize: 12, fontWeight: 700 }}>
        REGIME: {data.regime}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {data.candidates.map(c => (
          <CandidateCard key={c.id} candidate={c} />
        ))}
      </div>

      <EscapeValveMemory edgeStability={0.8} />
    </div>
  );
}
