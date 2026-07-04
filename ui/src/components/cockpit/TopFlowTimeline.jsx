import React from 'react';
import { useArivuStore } from '../../store/useArivuStore';

const STAGES = [
  'FEED', 'BARS', 'GRAPH', 'THESIS', 'READINESS', 
  'TWIN', 'LEDGER', 'TRUST', 'OPTIMIZER'
];

export function TopFlowTimeline() {
  const { activeStage, stageData } = useArivuStore();
  
  const getStageStatus = (stage) => {
    const data = stageData[stage];
    if (!data) return 'WAITING';
    
    switch(stage) {
      case 'FEED': return data.stream_status || 'LIVE';
      case 'BARS': return `${data.bars_ready}/${data.required_bars}`;
      case 'GRAPH': return data.nodes ? 'READY' : 'WAITING';
      case 'THESIS': return data.hypotheses?.length > 0 ? 'READY' : 'NONE';
      case 'READINESS': return data.ml1_verdict || 'WAITING';
      case 'TWIN': return data.trajectories ? 'DONE' : 'WAITING';
      case 'LEDGER': return data.decision_object ? 'COMMITTED' : 'WAITING';
      case 'TRUST': return data.delta ? 'UPDATED' : 'WAITING';
      case 'OPTIMIZER': return data.candidates ? 'ACTIVE' : 'WAITING';
      default: return 'WAITING';
    }
  };

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 4,
      padding: '8px 0',
      width: '100%',
      overflowX: 'auto'
    }}>
      {STAGES.map((stage, idx) => {
        const isActive = activeStage === stage;
        return (
          <React.Fragment key={stage}>
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              padding: '4px 8px',
              borderRadius: 4,
              background: isActive ? 'var(--accent-dim)' : 'transparent',
              border: isActive ? '1px solid var(--border-active)' : '1px solid transparent',
              minWidth: 80
            }}>
              <span style={{ fontSize: 10, fontWeight: 700, color: isActive ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                {stage}
              </span>
              <span style={{ fontSize: 10, fontFamily: 'JetBrains Mono, monospace', color: isActive ? 'var(--accent)' : 'var(--text-faint)' }}>
                {getStageStatus(stage)}
              </span>
            </div>
            {idx < STAGES.length - 1 && (
              <div style={{
                width: 12,
                height: 1,
                background: 'var(--border-soft)'
              }} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
