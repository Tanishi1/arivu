import React from 'react';
import { useCrossCuttingStore } from '../../store/useCrossCuttingStore';

export function EscapeValveMemory({ edgeStability }) {
  const { escapeValve } = useCrossCuttingStore();
  
  if (!escapeValve.fired && escapeValve.holdsCount === 0) return null;

  return (
    <div style={{
      border: '1px solid var(--border-soft)',
      borderRadius: 6,
      padding: 12,
      marginTop: 16,
      background: 'var(--panel-bg)',
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: 11
    }}>
      <div style={{ fontWeight: 700, color: 'var(--text-secondary)', marginBottom: 8 }}>
        ESCAPE VALVE MEMORY
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, color: 'var(--text-primary)' }}>
        <span>holds before trade: {escapeValve.holdsCount}</span>
        <span>escape trade used: {escapeValve.fired ? 'yes' : 'no'}</span>
        <span>position fraction: {escapeValve.positionFraction}</span>
        <span>edge stability: {edgeStability?.toFixed(2) || 'N/A'}</span>
        <span>outcome added to optimizer: {escapeValve.fired ? 'yes' : 'no'}</span>
        <span>feedback enabled: {escapeValve.feedbackEnabled ? 'yes' : 'no'}</span>
      </div>
    </div>
  );
}
