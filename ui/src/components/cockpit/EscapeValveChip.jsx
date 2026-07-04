import React from 'react';
import { useCrossCuttingStore } from '../../store/useCrossCuttingStore';

export function EscapeValveChip() {
  const { escapeValve } = useCrossCuttingStore();
  
  let statusText = 'INACTIVE';
  let holdsDisplay = `${escapeValve.holdsCount}/${escapeValve.holdsThreshold} HOLDS`;
  let color = 'var(--text-muted)';
  let bg = 'var(--border-soft)';

  if (escapeValve.fired) {
    statusText = 'FIRED';
    holdsDisplay = `${escapeValve.positionFraction * 100}% SIZE`;
    color = 'var(--text-on-accent)';
    bg = 'var(--escape-valve-color)';
  } else if (escapeValve.armed) {
    statusText = 'ARMED';
    color = 'var(--escape-valve-color)';
    bg = 'var(--escape-valve-bg)';
  }

  return (
    <div style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 6,
      padding: '4px 10px',
      borderRadius: 12,
      background: bg,
      color: color,
      fontSize: 11,
      fontFamily: 'JetBrains Mono, monospace',
      fontWeight: 700,
      border: `1px solid ${escapeValve.armed && !escapeValve.fired ? color : 'transparent'}`
    }}>
      <span>ESCAPE VALVE: {statusText}</span>
      <span style={{ opacity: 0.6 }}>·</span>
      <span>{holdsDisplay}</span>
      {escapeValve.fired && escapeValve.feedbackEnabled && (
        <>
          <span style={{ opacity: 0.6 }}>·</span>
          <span>FEEDBACK ENABLED</span>
        </>
      )}
    </div>
  );
}
