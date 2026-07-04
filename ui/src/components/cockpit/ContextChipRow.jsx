import React from 'react';
import { EscapeValveChip } from './EscapeValveChip';
import { RegimeChip } from './RegimeChip';

export function ContextChipRow() {
  return (
    <div style={{ display: 'flex', gap: 8 }}>
      <EscapeValveChip />
      <RegimeChip />
    </div>
  );
}
