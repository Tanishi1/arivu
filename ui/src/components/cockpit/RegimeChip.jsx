import React from 'react';
import { useCrossCuttingStore } from '../../store/useCrossCuttingStore';

export function RegimeChip() {
  const { regime } = useCrossCuttingStore();
  
  let displayText = '';
  
  if (regime.status === 'bootstrap') {
    displayText = 'REGIME MODEL: BOOTSTRAP · threshold rules active';
  } else if (regime.status === 'retraining') {
    displayText = `REGIME MODEL: RETRAINING · ${regime.populationSize} closed trades`;
  } else {
    displayText = `REGIME MODEL: TRAINED · K=${regime.k} · ${regime.clusterId || 'unknown'}`;
    if (regime.confidence) {
      displayText += ` · confidence ${regime.confidence.toFixed(2)}`;
    }
  }

  return (
    <div style={{
      display: 'inline-flex',
      alignItems: 'center',
      padding: '4px 10px',
      borderRadius: 12,
      background: 'var(--border-soft)',
      color: 'var(--text-secondary)',
      fontSize: 11,
      fontFamily: 'JetBrains Mono, monospace',
      fontWeight: 700
    }}>
      {displayText}
    </div>
  );
}
