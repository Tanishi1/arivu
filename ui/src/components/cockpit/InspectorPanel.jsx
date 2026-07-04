import React from 'react';
import { useInspectorStore } from '../../store/useInspectorStore';
import { useArivuStore } from '../../store/useArivuStore';

function InspectorField({ label, value }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-soft)', padding: '4px 0' }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{ textAlign: 'right' }}>{value}</span>
    </div>
  );
}

function ReplayEvidencePanel() {
  const { traceMode } = useArivuStore();
  if (traceMode !== 'REPLAY') return null;
  
  return (
    <div style={{ marginTop: 16, border: '1px solid var(--border-soft)', padding: 12, borderRadius: 6 }}>
      <div style={{ fontWeight: 700, marginBottom: 8, color: 'var(--text-secondary)' }}>REPLAY EVIDENCE</div>
      <InspectorField label="decision_objects" value="loaded" />
      <InspectorField label="outcome_records" value="loaded" />
      <InspectorField label="causal_graphs" value="loaded" />
      <InspectorField label="causal_agent.jsonl" value="34 records" />
    </div>
  );
}

export function InspectorPanel() {
  const { pinnedObject, autoFollow, clearPinned } = useInspectorStore();
  const { activeStage } = useArivuStore();

  const title = pinnedObject ? `Pinned: ${pinnedObject.type}` : `Auto-follow: ${activeStage}`;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      color: 'var(--text-primary)',
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: 11
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h3 style={{ margin: 0 }}>INSPECTOR</h3>
        {pinnedObject && (
          <button 
            onClick={clearPinned}
            style={{
              background: 'transparent', border: '1px solid var(--border-soft)', color: 'var(--text-secondary)',
              borderRadius: 4, cursor: 'pointer', padding: '2px 8px', fontSize: 10
            }}
          >
            Unpin
          </button>
        )}
      </div>

      <div style={{ marginBottom: 12, color: 'var(--text-muted)' }}>{title}</div>

      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4 }}>
        {!pinnedObject ? (
          <div style={{ color: 'var(--text-muted)' }}>Raw fields for {activeStage} will appear here.</div>
        ) : (
          Object.entries(pinnedObject.data || {}).map(([key, val]) => (
            <InspectorField key={key} label={key} value={String(val)} />
          ))
        )}
      </div>

      <ReplayEvidencePanel />
    </div>
  );
}
