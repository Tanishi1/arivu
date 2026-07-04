import React from 'react';
import { useCrossCuttingStore } from '../../store/useCrossCuttingStore';

export function EscapeValveBranch({ bestScore, requiredScore }) {
  const { escapeValve } = useCrossCuttingStore();
  
  if (!escapeValve.fired) return null;

  return (
    <div style={{
      border: '2px dashed var(--escape-valve-color)',
      borderRadius: 8,
      padding: 16,
      margin: '16px 0',
      background: 'var(--escape-valve-bg)',
      color: 'var(--escape-valve-color)',
      fontFamily: 'JetBrains Mono, monospace'
    }}>
      <h4 style={{ margin: '0 0 8px 0', fontSize: 14 }}>NO VALIDATED CHAIN</h4>
      <div style={{ fontSize: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <span>best score: {bestScore?.toFixed(2) || 'N/A'}</span>
        <span>required score: {requiredScore?.toFixed(2) || 'N/A'}</span>
        <span>consecutive holds: {escapeValve.holdsCount}/{escapeValve.holdsThreshold}</span>
        <span style={{ fontWeight: 700 }}>ESCAPE VALVE FIRED</span>
        <span>position fraction: {escapeValve.positionFraction}</span>
        <span>purpose: training data</span>
      </div>
    </div>
  );
}
