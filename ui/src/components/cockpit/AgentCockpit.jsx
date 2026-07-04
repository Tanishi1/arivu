import React from 'react';
import { useArivuStore } from '../../store/useArivuStore';
import { TopFlowTimeline } from './TopFlowTimeline';
import { StatusLine } from './StatusLine';
import { ContextChipRow } from './ContextChipRow';

import { StageFeed } from './StageFeed';
import { StageBars } from './StageBars';
import { StageGraph } from './StageGraph/StageGraph';
import { StageThesis } from './StageThesis';
import { StageReadiness } from './StageReadiness';
import { StageTwin } from './StageTwin';
import { StageLedger } from './StageLedger';
import { StageTrust } from './StageTrust';
import { StageOptimizer } from './StageOptimizer/StageOptimizer';
import { InspectorPanel } from './InspectorPanel';
import { useTraceController } from '../../hooks/useTraceController';

const STAGES = [
  'FEED', 'BARS', 'GRAPH', 'THESIS', 'READINESS', 
  'TWIN', 'LEDGER', 'TRUST', 'OPTIMIZER'
];

export function AgentCockpit() {
  const { setTraceMode, activeStage, setActiveStage, stageData } = useArivuStore();
  useTraceController();

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      width: '100%',
      background: 'var(--bg-base)',
      color: 'var(--text-primary)',
      fontFamily: 'Inter, sans-serif'
    }}>
      {/* Top Bar */}
      <div style={{
        padding: '12px 24px',
        borderBottom: '1px solid var(--border-soft)',
        display: 'flex',
        flexDirection: 'column',
        gap: 12
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <ContextChipRow />
        </div>
        <TopFlowTimeline />
      </div>

      {/* Control Row */}
      <div style={{
        padding: '12px 24px',
        borderBottom: '1px solid var(--border-soft)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'var(--panel-bg)'
      }}>
        <div style={{ display: 'flex', gap: 12 }}>
          <button 
            onClick={() => setTraceMode('LIVE')}
            style={{
              padding: '6px 16px',
              background: 'var(--accent)',
              color: 'var(--text-on-accent)',
              border: 'none',
              borderRadius: 4,
              fontWeight: 600,
              cursor: 'pointer'
            }}>
            Start Live Trace
          </button>
          <button 
            onClick={() => setTraceMode('REPLAY')}
            style={{
              padding: '6px 16px',
              background: 'transparent',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-soft)',
              borderRadius: 4,
              fontWeight: 600,
              cursor: 'pointer'
            }}>
            Replay Last Experiment
          </button>
        </div>
        <StatusLine />
      </div>

      {/* Main Content Area */}
      <div style={{
        display: 'flex',
        flex: 1,
        overflow: 'hidden'
      }}>
        {/* Main Flow Area */}
        <div style={{
          flex: 1,
          padding: 24,
          overflowY: 'auto',
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          gap: 16
        }}>
          {activeStage === 'FEED' && <StageFeed data={stageData?.FEED} />}
          {activeStage === 'BARS' && <StageBars data={stageData?.BARS} />}
          {activeStage === 'GRAPH' && <StageGraph data={stageData?.GRAPH} />}
          {activeStage === 'THESIS' && <StageThesis data={stageData?.THESIS} />}
          {activeStage === 'READINESS' && <StageReadiness data={stageData?.READINESS} />}
          {activeStage === 'TWIN' && <StageTwin data={stageData?.TWIN} />}
          {activeStage === 'LEDGER' && <StageLedger data={stageData?.LEDGER} />}
          {activeStage === 'TRUST' && <StageTrust data={stageData?.TRUST} />}
          {activeStage === 'OPTIMIZER' && <StageOptimizer data={stageData?.OPTIMIZER} />}
          
          {/* Dev only: click to test layout */}
          <div style={{ marginTop: 40, borderTop: '1px solid var(--border-soft)', paddingTop: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 10, color: 'var(--text-muted)', width: '100%' }}>DEV ONLY: Click to change stage manually</span>
            {STAGES.map(s => (
              <button key={s} onClick={() => setActiveStage(s)} style={{ fontSize: 10 }}>{s}</button>
            ))}
          </div>
        </div>

        {/* Right Inspector */}
        <div style={{
          width: 320,
          borderLeft: '1px solid var(--border-soft)',
          background: 'var(--panel-bg)',
          padding: 16,
          overflowY: 'auto'
        }}>
          <InspectorPanel />
        </div>
      </div>
    </div>
  );
}
