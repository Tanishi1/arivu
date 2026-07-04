import React from 'react';

export function StageLedger({ data }) {
  if (!data || !data.decision_object) return <div>No Ledger Data</div>;

  const { decision_object, outcome_record } = data;

  return (
    <div style={{
      border: '1px solid var(--border-soft)',
      borderRadius: 8,
      padding: 16,
      background: 'var(--panel-bg)',
      color: 'var(--text-primary)',
      fontFamily: 'JetBrains Mono, monospace'
    }}>
      <h3 style={{ color: 'var(--ledger-color)', marginTop: 0 }}>LEDGER (Audit)</h3>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 11 }}>
        <div>
          <strong style={{ color: 'var(--text-secondary)' }}>DECISION OBJECT</strong>
          <div>ID: {decision_object.id}</div>
          <div style={{ marginTop: 4 }}>
            <div>ASSUMPTIONS COMMITTED: {decision_object.assumptions?.length || 0}</div>
            {decision_object.assumptions?.map((assm, i) => (
              <div key={i}>
                {i + 1}. {assm.source} → {assm.target}
              </div>
            ))}
          </div>
        </div>

        {outcome_record && (
          <div style={{ paddingTop: 12, borderTop: '1px solid var(--border-soft)' }}>
            <strong style={{ color: 'var(--text-secondary)' }}>OUTCOME RECORD</strong>
            <div>PnL: {outcome_record.pnl}</div>
          </div>
        )}
      </div>
    </div>
  );
}
