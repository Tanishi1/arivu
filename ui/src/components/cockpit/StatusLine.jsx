import React from 'react';
import { useArivuStore } from '../../store/useArivuStore';

export function StatusLine() {
  const { statusLine } = useArivuStore();
  const { state, action, reason } = statusLine || { state: 'IDLE', action: 'HOLD', reason: 'WAITING' };
  
  return (
    <div style={{
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: 12,
      fontWeight: 600,
      color: 'var(--text-secondary)',
      padding: '4px 12px',
      background: 'var(--panel-bg)',
      border: '1px solid var(--border-soft)',
      borderRadius: 4,
      display: 'inline-flex',
      alignItems: 'center',
      gap: 12
    }}>
      <span>STATE: {state}</span>
      <span style={{ color: 'var(--border-soft)' }}>·</span>
      <span>ACTION: {action}</span>
      <span style={{ color: 'var(--border-soft)' }}>·</span>
      <span>REASON: {reason}</span>
    </div>
  );
}
