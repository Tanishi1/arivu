import React from 'react';

export function StageFeed({ data }) {
  if (!data) return <div>No Feed Data</div>;

  return (
    <div style={{
      border: '1px solid var(--border-soft)',
      borderRadius: 8,
      padding: 16,
      background: 'var(--panel-bg)',
      color: 'var(--text-primary)',
      fontFamily: 'JetBrains Mono, monospace'
    }}>
      <h3 style={{ color: 'var(--feed-color)', marginTop: 0 }}>FEED (Live Market)</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, fontSize: 12 }}>
        <div>Price: {data.price}</div>
        <div>Spread: {data.spread}</div>
        <div>Volume: {data.volume}</div>
        <div>Trade Pressure: {data.trade_pressure}</div>
        <div>BTC Return: {data.btc_return}</div>
        <div>ETH Return: {data.eth_return}</div>
        <div>Stream: <span style={{ color: data.stream_status === 'live' ? 'var(--success)' : 'var(--error)' }}>{data.stream_status}</span></div>
      </div>
    </div>
  );
}
