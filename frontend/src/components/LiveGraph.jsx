import { useEffect, useRef, useCallback, useState } from 'react'

// ├óΓÇ¥Γé¼├óΓÇ¥Γé¼ Pipeline stage sequence (exact Arivu pipeline) ├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼

// STREAMS: 6 Binance WS feeds ΓåÆ tick merge

// BARS:    ticks accumulate 10s ΓåÆ bar close ΓåÆ 19 variables computed ΓåÆ adaptive PCMCI refresh

// PCMCI:   graph discovery (PCMCI, ╧ä_max/╬▒ driven by MetaOptimizer hill-climb) ΓåÆ GraphSnapshot

// HYPOTHESIS: walk chains to price_return ΓåÆ rank top-5 ΓåÆ composite score (geometric mean for multi-hop)

// OPTIMIZER:  MetaParameterOptimizer hill-climbs 5 dims: k_runs/min_runs/threshold/╧ä_max/╬▒ per regime

// TWIN:    TwinSimulator: expected causal path vs actual market path

// DECISION: trade (if score>0.02) or hold (log reason)

const VSTAGES = ['STREAMS','BARS','PCMCI','HYPOTHESIS','OPTIMIZER','TWIN','DECISION','LEARNING']

const VDUR = {

  STREAMS:   8000,

  BARS:      14000,

  PCMCI:     20000,

  HYPOTHESIS: 4000,

  OPTIMIZER: 10000,

  TWIN:      10000,

  DECISION:   6000,

  LEARNING:  10000,

}

// ├óΓÇ¥Γé¼├óΓÇ¥Γé¼ Layer config ├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼

const LO = ['macro','market','microstructure','health','target']

const LI = { macro:0, market:1, microstructure:2, health:3, target:4 }

const LX = { macro:.10, market:.30, microstructure:.52, health:.74, target:.92 }

const LC = { macro:'#7c3aed', market:'#4f46e5', microstructure:'#2563eb', health:'#475569', target:'#16a34a' }

// Patterns match actual VARIABLE_NAMES from feature_bar.py

const LP = {

  macro:          [/^btc_return/, /^eth_return/, /^session_/],

  market:         [/^price_in_band/, /^regime_trending/, /^volatil/, /^spread$/, /^volume$/, /^rsi$/, /^trade_int/, /^ema_sp/, /^bollinger/],

  microstructure: [/^order_book_imb/],

  health:         [/^algo_health/, /^regime_vol/],

  target:         [/^price_return/],

}

function getLayer(n){ for(const l of LO){ if(LP[l]?.some(p=>p.test(n))) return l } return 'market' }

// ├óΓÇ¥Γé¼├óΓÇ¥Γé¼ 19 VARIABLE_NAMES (exact order from feature_bar.py) ├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼

const VARIABLE_NAMES = [

  'price_return','volume','spread','volatility','rsi',

  'trade_intensity','order_book_imbalance','ema_spread',

  'bollinger_width','price_in_band','regime_volatile',

  'regime_trending','btc_return','eth_return',

  'algo_health_p_normal','algo_health_p_stressed','algo_health_p_degraded',

  'session_sin','session_cos',

]

// ├óΓÇ¥Γé¼├óΓÇ¥Γé¼ 6 Binance WS streams and what they produce ├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼

const WS_STREAMS = [

  { name:'kline_1m',      col:'#7c3aed', fields:'price ┬╖ volume (OHLCV)',         vars:['price_return','volume'] },

  { name:'bookTicker',    col:'#4f46e5', fields:'bid ┬╖ ask  ΓåÆ  spread',            vars:['spread','ema_spread'] },

  { name:'aggTrade',      col:'#2563eb', fields:'trade side aggression',            vars:['trade_intensity'] },

  { name:'depth5@100ms',  col:'#0891b2', fields:'L2 order book  ΓåÆ  imbalance',     vars:['order_book_imbalance'] },

  { name:'btcusdt',       col:'#f59e0b', fields:'BTC macro price  ΓåÆ  btc_return',  vars:['btc_return'] },

  { name:'ethusdt',       col:'#ef4444', fields:'ETH macro price  ΓåÆ  eth_return',  vars:['eth_return'] },

]

// ├óΓÇ¥Γé¼├óΓÇ¥Γé¼ 5 causal layers (groups of VARIABLE_NAMES) ├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼

const GROUPS = [

  { l:'macro',         label:'MACRO',  col:'#7c3aed',

    vars:['btc_return','eth_return','session_sin','session_cos'] },

  { l:'market',        label:'MARKET', col:'#4f46e5',

    vars:['volatility','spread','volume','rsi','trade_intensity','ema_spread','bollinger_width','price_in_band','regime_trending'] },

  { l:'microstructure',label:'MICRO',  col:'#2563eb',

    vars:['order_book_imbalance'] },

  { l:'health',        label:'HEALTH', col:'#475569',

    vars:['algo_health_p_normal','algo_health_p_stressed','algo_health_p_degraded','regime_volatile'] },

  { l:'target',        label:'TARGET', col:'#16a34a',

    vars:['price_return'] },

]

// ├óΓÇ¥Γé¼├óΓÇ¥Γé¼ Mock fallback (real WS field names: ts, price, spread, volatility, rsi) ├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼

const _ts = s => new Date(Date.now()-s*1000).toISOString()

const MOCK_TICKS = [

  { ts:_ts(0),  price:82.23, spread:0.01, volatility:0.000043, rsi:61.2 },

  { ts:_ts(10), price:82.19, spread:0.01, volatility:0.000051, rsi:58.7 },

  { ts:_ts(20), price:82.31, spread:0.01, volatility:0.000038, rsi:66.3 },

  { ts:_ts(30), price:82.28, spread:0.01, volatility:0.000047, rsi:63.1 },

  { ts:_ts(40), price:82.15, spread:0.02, volatility:0.000062, rsi:51.4 },

  { ts:_ts(50), price:82.09, spread:0.01, volatility:0.000071, rsi:46.9 },

]

const MOCK_GRAPH = {

  variables: VARIABLE_NAMES,

  edges: [

    { source:'btc_return',           target:'price_return',         lag:1, coeff: 0.61, p_value:0.002,  stability:1.00, validated:true  },

    { source:'rsi',                  target:'price_return',         lag:2, coeff: 0.38, p_value:0.018,  stability:0.80, validated:true  },

    { source:'volatility',           target:'price_return',         lag:1, coeff:-0.29, p_value:0.031,  stability:0.75, validated:true  },

    { source:'order_book_imbalance', target:'price_return',         lag:1, coeff: 0.44, p_value:0.009,  stability:0.90, validated:true  },

    { source:'session_cos',          target:'price_return',         lag:3, coeff: 0.22, p_value:0.041,  stability:1.00, validated:true  },

    { source:'regime_volatile',      target:'price_return',         lag:4, coeff:-0.17, p_value:0.038,  stability:1.00, validated:true  },

    { source:'eth_return',           target:'btc_return',           lag:1, coeff: 0.58, p_value:0.001,  stability:1.00, validated:true  },

    { source:'btc_return',           target:'volatility',           lag:2, coeff: 0.21, p_value:0.034,  stability:0.67, validated:true  },

    { source:'bollinger_width',      target:'volatility',           lag:1, coeff: 0.28, p_value:0.027,  stability:0.75, validated:true  },

    { source:'trade_intensity',      target:'order_book_imbalance', lag:1, coeff: 0.33, p_value:0.022,  stability:0.80, validated:true  },

    { source:'ema_spread',           target:'spread',               lag:1, coeff: 0.72, p_value:0.0001, stability:1.00, validated:true  },

    { source:'spread',               target:'order_book_imbalance', lag:2, coeff:-0.19, p_value:0.045,  stability:0.67, validated:false },

  ],

}

const MOCK_HYPOTHESIS = {

  id:'h1',

  chain:[{ source:'btc_return', target:'price_return', lag:1, coeff:0.61 }],

  chain_summary:()=>'btc_returnΓåÆ(lag=1)ΓåÆprice_return',

  predicted_direction:'up', predicted_magnitude:0.0061,

  composite_score:0.3742, layer1_score:0.68, layer2_score:0.55, is_escape_valve:false,

}

// ├óΓÇ¥Γé¼├óΓÇ¥Γé¼ Helpers ├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼

const ease = t => t<0.5 ? 2*t*t : -1+(4-2*t)*t

const abbr = n => n.length>12 ? n.slice(0,5)+'ΓÇª'+n.slice(-3) : n

// Robustly extract "source|target" edge pairs from any hypothesis format:

//   Real JSONL ΓåÆ chain_edges:[{source,target}]  OR  chain:"AΓåÆ(lag=1)ΓåÆB" string

//   Mock       ΓåÆ chain:[{source,target}] array

function hypPairs(hyp){

  if(Array.isArray(hyp?.chain_edges)&&hyp.chain_edges.length>0)

    return new Set(hyp.chain_edges.map(e=>`${e.source}|${e.target}`))

  if(Array.isArray(hyp?.chain)&&hyp.chain.length>0)

    return new Set(hyp.chain.map(e=>`${e.source}|${e.target}`))

  if(typeof hyp?.chain==='string'&&hyp.chain)

    return chainSet(hyp.chain)

  return new Set()

}

function bzPt(sx,sy,tx,ty,t){

  const mx=(sx+tx)/2, mt=1-t

  return {

    x: mt*mt*mt*sx + 3*mt*mt*t*mx + 3*mt*t*t*mx + t*t*t*tx,

    y: mt*mt*mt*sy + 3*mt*mt*t*sy + 3*mt*t*t*ty  + t*t*t*ty,

  }

}

function chainSet(c){
  if(!c) return new Set()
  const s = new Set()

  // Handle format: "src->(lag=N)->tgt | src2->(lag=M)->tgt2 | ..."
  // Split on ' | ' pipe separator first (multi-edge chains)
  const segments = c.split(/\s*\|\s*/)

  segments.forEach(seg => {
    // Each segment: "source->(lag=N)->target" or "sourceΓåÆ(lag=N)ΓåÆtarget"
    // Split on -> or ΓåÆ (with optional whitespace), then strip (lag=N) and dashes
    const parts = seg
      .split(/\s*-?[ΓåÆ>]-?\s*/)               // split on -> or ΓåÆ (with optional -)
      .map(p => p.replace(/\(.*?\)/g,'')     // strip (lag=N) etc
                  .replace(/^-+|-+$/g,'')    // strip leading/trailing dashes
                  .trim())
      .filter(Boolean)

    // Build source|target pairs from adjacent nodes
    parts.forEach((p, i) => { if(parts[i+1]) s.add(`${p}|${parts[i+1]}`) })
  })

  return s
}


// ├óΓÇ¥Γé¼├óΓÇ¥Γé¼ Draw primitives ├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼├óΓÇ¥Γé¼

function drawNode(ctx,n,alpha=0.85){

  const col=LC[n.l]||'#94a3b8', r=n.name==='price_return'?20:14

  ctx.save();ctx.globalAlpha=alpha

  ctx.beginPath();ctx.arc(n.x,n.y,r,0,Math.PI*2)

  ctx.fillStyle=col+'22';ctx.fill()

  ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.stroke()

  ctx.font='600 9px JetBrains Mono,monospace';ctx.fillStyle=col;ctx.textAlign='center'

  ctx.fillText(abbr(n.name),n.x,n.y+r+11)

  ctx.restore();ctx.globalAlpha=1

}

function drawAtom(ctx,x,y,col,r=5){

  ctx.save()

  ctx.beginPath();ctx.arc(x,y,r+4,0,Math.PI*2);ctx.fillStyle=col+'28';ctx.fill()

  ctx.shadowColor=col;ctx.shadowBlur=14

  ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=col;ctx.fill()

  ctx.shadowBlur=0;ctx.restore()

}

function drawGraph(ctx,nodes,edges,cs,alpha=1,chainAlpha=1){

  edges.forEach(e=>{

    const sI=LI[getLayer(e.source)]??2,tI=LI[getLayer(e.target)]??2;if(sI>=tI)return

    const onC=cs.has(`${e.source}|${e.target}`)

    const col=LC[getLayer(e.source)]||'#94a3b8',mx=(e.sx+e.tx)/2

    const nr=e.target==='price_return'?16:10

    const dx=e.tx-e.sx,dy=e.ty-e.sy,len=Math.sqrt(dx*dx+dy*dy)||1

    const tx2=e.tx-(dx/len)*nr,ty2=e.ty-(dy/len)*nr

    ctx.save();ctx.strokeStyle=col

    ctx.lineWidth=onC?2.5:(e.validated?1.5:0.5)

    ctx.globalAlpha=onC?chainAlpha:(e.validated?alpha*0.5:alpha*0.12)

    if(onC){ctx.shadowColor=col;ctx.shadowBlur=8}

    ctx.beginPath();ctx.moveTo(e.sx,e.sy);ctx.bezierCurveTo(mx,e.sy,mx,e.ty,tx2,ty2);ctx.stroke()

    ctx.shadowBlur=0

    if((onC||e.validated)&&alpha>0.25){

      const ang=Math.atan2(ty2-e.ty,tx2-mx)

      ctx.translate(tx2,ty2);ctx.rotate(ang)

      ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(-7,-3.5);ctx.lineTo(-7,3.5);ctx.closePath()

      ctx.fillStyle=col;ctx.fill()

    }

    ctx.restore();ctx.globalAlpha=1

  })

  Object.values(nodes).forEach(n=>drawNode(ctx,n,alpha*0.85))

}


function stageLabel(ctx,W,H,text,sT){

  ctx.font='600 12px JetBrains Mono,monospace';ctx.fillStyle='#94a3b8';ctx.textAlign='center'

  ctx.fillText(text,W/2,H-8)

  const pad=(W-500)/2

  ctx.fillStyle='#e2e8f0';ctx.fillRect(pad,H-4,500,2)

  ctx.fillStyle='#7c3aed';ctx.fillRect(pad,H-4,Math.min(500,500*sT),2)

}

function drawLivePanel(ctx,W,H,ticks,ld){

  if(!ld)return

  const px=W-236,py=12,pw=222,ph=120

  ctx.save();ctx.shadowColor='rgba(0,0,0,0.1)';ctx.shadowBlur=18

  ctx.fillStyle='rgba(255,255,255,0.97)';ctx.beginPath();ctx.roundRect(px,py,pw,ph,10);ctx.fill()

  ctx.shadowBlur=0;ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1;ctx.stroke()

  ctx.fillStyle='#16a34a';ctx.fillRect(px,py,pw,18)

  ctx.font='700 9px JetBrains Mono,monospace';ctx.fillStyle='#fff';ctx.textAlign='left'

  ctx.fillText('LIVE',px+10,py+13)

  ctx.font='700 20px JetBrains Mono,monospace';ctx.fillStyle='#1e1b4b'

  ctx.fillText(`$${Number(ld.price||0).toFixed(3)}`,px+10,py+46)

  ctx.font='600 8px JetBrains Mono,monospace';ctx.fillStyle='#64748b'

  ctx.fillText(`RSI ${ld.rsi?.toFixed(1)||'ΓÇö'}  ╧â ${ld.volatility?.toFixed(5)||'ΓÇö'}`,px+10,py+60)

  // Mini sparkline

  const pts=ticks.slice(-18);if(pts.length>1){

    const mn=Math.min(...pts.map(t=>t.price||0)),mx2=Math.max(...pts.map(t=>t.price||0)),rng=mx2-mn||1

    ctx.strokeStyle='#4f46e5';ctx.lineWidth=1.2;ctx.beginPath()

    pts.forEach((t,i)=>{

      const x=px+10+i*(pw-20)/(pts.length-1),y=py+78-((t.price-mn)/rng)*14

      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)

    });ctx.stroke()

  }

  ctx.font='600 7.5px JetBrains Mono,monospace';ctx.fillStyle='#2563eb'

  ctx.fillText('HYP: '+(ld.selected_chain||'ΓÇö').slice(0,28),px+10,py+102)

  ctx.fillStyle=ld.hold_reason==='traded'?'#16a34a':'#f59e0b'

  ctx.fillText(ld.hold_reason==='traded'?'Γû▓ TRADE':('ΓùÅ '+(ld.hold_reason||'observing')).slice(0,26),px+10,py+115)

  ctx.restore()

}

// ΓöÇΓöÇ Layer column backgrounds (for graph stages) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

function drawLayerBg(ctx,W,H,nodes){

  LO.forEach(l=>{

    const x=W*LX[l],col=LC[l]

    ctx.globalAlpha=0.06;ctx.fillStyle=col;ctx.fillRect(x-W*.11,0,W*.22,H)

    ctx.globalAlpha=0.10;ctx.strokeStyle=col;ctx.lineWidth=1

    ctx.beginPath();ctx.moveTo(x-W*.11,0);ctx.lineTo(x-W*.11,H);ctx.stroke()

    ctx.beginPath();ctx.moveTo(x+W*.11,0);ctx.lineTo(x+W*.11,H);ctx.stroke()

    ctx.globalAlpha=1

  })

  ctx.font='700 10px JetBrains Mono,monospace';ctx.textAlign='center'

  LO.forEach(l=>{

    const ns=Object.values(nodes).filter(n=>n.l===l)

    if(!ns.length)return

    ctx.fillStyle=LC[l]+'99';ctx.globalAlpha=0.8

    ctx.fillText(l.toUpperCase(),W*LX[l],18)

    ctx.globalAlpha=1

  })

}

// ΓöÇΓöÇ Main Component ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

export default function LiveGraph({

  graph, ticks=[], selectedChain, hypotheses=[], activeStage='STREAMS',

  playing=true, liveData, agentState, optimizer

}){

  const cvs          = useRef(null)

  const nodesRef     = useRef({})

  const edgesRef     = useRef([])

  const vsIdxRef     = useRef(0)

  const vsStartRef   = useRef(null)

  const propsRef     = useRef({})

  const clickedLayer = useRef(null)

  const [vsLabel, setVsLabel] = useState('STREAMS')

  propsRef.current = { graph, ticks, selectedChain, hypotheses, activeStage, playing, liveData, agentState, optimizer }

  // Stage pill ΓåÆ pipeline index mapping

  useEffect(()=>{

    const map = {

      STREAMS:0, FEED:0,

      BARS:1,

      PCMCI:2, GRAPH:2, FLOW:2,

      HYPOTHESIS:3, THESIS:3,

      OPTIMIZER:4,

      TWIN:5,

      DECISION:6, READINESS:6, LEDGER:6, TRUST:6,

      LEARNING:7,

    }

    const idx = map[activeStage] ?? 0

    vsIdxRef.current = idx

    vsStartRef.current = null

    setVsLabel(VSTAGES[idx])

  },[activeStage])

  // Build node/edge layout from graph snapshot

  const layout = useCallback((g,W,H)=>{

    if(!g||!W||!H) return

    const ns = new Set(VARIABLE_NAMES)

    g.edges?.forEach(e=>{ ns.add(e.source); ns.add(e.target) })

    g.variables?.forEach(v=>ns.add(v))

    const bk = {}; LO.forEach(l=>bk[l]=[])

    ns.forEach(n=>bk[getLayer(n)].push(n))

    const nodes={}, pH=44, uH=H-pH*2

    LO.forEach(l=>{

      const arr=bk[l]; if(!arr.length)return

      const x=W*LX[l], st=uH/(arr.length+1)

      arr.forEach((n,i)=>nodes[n]={x, y:pH+st*(i+1), l, name:n})

    })

    nodesRef.current = nodes

    edgesRef.current = (g.edges||[]).map(e=>{

      const s=nodes[e.source], t=nodes[e.target]

      if(!s||!t) return null

      return { ...e, sx:s.x, sy:s.y, tx:t.x, ty:t.y }

    }).filter(Boolean)

  },[])

  // RAF render loop

  const drawFn = useRef(null)

  drawFn.current = useCallback((ts)=>{

    const canvas = cvs.current; if(!canvas) return

    const par = canvas.parentElement; if(!par) return

    const W=par.clientWidth, H=par.clientHeight

    if(canvas.width!==W||canvas.height!==H){

      canvas.width=W; canvas.height=H

      layout(propsRef.current.graph||MOCK_GRAPH,W,H)

    }

    if(!W||!H) return

    const { ticks:tk, selectedChain:sc, playing:pl, liveData:ld, optimizer:opt } = propsRef.current

    const nodes   = nodesRef.current

    const edges   = edgesRef.current

    const cs      = chainSet(sc)

    const activeTicks = (tk&&tk.length) ? tk : MOCK_TICKS

    const activeHyps  = (propsRef.current.hypotheses?.length) ? propsRef.current.hypotheses : [MOCK_HYPOTHESIS]

    // Timing

    if(vsStartRef.current===null) vsStartRef.current=ts

    const elapsed = ts - vsStartRef.current

    const vsName  = VSTAGES[vsIdxRef.current]

    const dur     = VDUR[vsName]||8000

    let sT = Math.min(1, elapsed/dur)

    if(pl && sT>=1){

      vsIdxRef.current = (vsIdxRef.current+1) % VSTAGES.length

      vsStartRef.current = ts; sT=0

      setVsLabel(VSTAGES[vsIdxRef.current])

    }

    // Clear

    const ctx = canvas.getContext('2d')

    ctx.clearRect(0,0,W,H)

    ctx.fillStyle='#fafafa'; ctx.fillRect(0,0,W,H)

    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    // STAGE 1: STREAMS ΓÇö 6 Binance WS feeds ΓåÆ tick merge

    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    if(vsName==='STREAMS'){

      const stageAlpha=sT<0.85?1:ease(Math.max(0,(1-sT)/0.15))

      ctx.save(); ctx.globalAlpha=stageAlpha

      const startX=12, streamW=W*0.38, streamH=52

      const totalH=WS_STREAMS.length*(streamH+6)

      const mergeY=H/2, mergeX=startX+streamW+50

      WS_STREAMS.forEach((s,i)=>{

        const sy=H/2-(totalH/2)+i*(streamH+6)

        const alpha=ease(Math.min(1,Math.max(0,(sT-i*0.06)/0.18)))

        if(alpha<=0) return

        ctx.save()

        ctx.globalAlpha=Math.min(1,alpha)*stageAlpha

        ctx.shadowColor=s.col+'33'; ctx.shadowBlur=10

        ctx.fillStyle='rgba(255,255,255,0.97)'

        ctx.beginPath(); ctx.roundRect(startX,sy,streamW,streamH,8); ctx.fill()

        ctx.shadowBlur=0; ctx.strokeStyle=s.col+'77'; ctx.lineWidth=1.5; ctx.stroke()

        // Left accent bar

        ctx.fillStyle=s.col

        ctx.beginPath(); ctx.roundRect(startX,sy,4,streamH,{upperLeft:8,lowerLeft:8,upperRight:0,lowerRight:0}); ctx.fill()

        // Stream name (left) + fields descriptor (right, muted)

        ctx.font='700 9px JetBrains Mono,monospace'; ctx.fillStyle=s.col; ctx.textAlign='left'

        ctx.fillText(s.name,startX+14,sy+20)

        ctx.font='500 7.5px JetBrains Mono,monospace'; ctx.fillStyle='#94a3b8'; ctx.textAlign='right'

        ctx.fillText(s.fields,startX+streamW-8,sy+20)

        // Variables row

        ctx.font='600 7.5px JetBrains Mono,monospace'; ctx.fillStyle='#475569'; ctx.textAlign='left'

        ctx.fillText(s.vars.join(' ┬╖ '),startX+14,sy+40)

        ctx.restore()

        // Arrow to merge node

        ctx.save(); ctx.strokeStyle=s.col+'44'; ctx.lineWidth=1

        ctx.globalAlpha=Math.min(1,alpha)*stageAlpha*0.55

        ctx.beginPath(); ctx.moveTo(startX+streamW,sy+streamH/2); ctx.lineTo(mergeX-28,mergeY); ctx.stroke()

        ctx.restore()

        // Data particles

        if(alpha>0.4){

          for(let pi=0;pi<3;pi++){

            const t=((sT*1.3+i*0.14+pi/3)%1)

            const px2=startX+streamW+(mergeX-28-(startX+streamW))*t

            const py2=(sy+streamH/2)+(mergeY-(sy+streamH/2))*t

            ctx.save(); ctx.globalAlpha=stageAlpha*Math.sin(t*Math.PI)*0.85

            drawAtom(ctx,px2,py2,s.col,3)

            ctx.restore()

          }

        }

      })

      // TICK MERGE node

      const mA=ease(Math.min(1,sT*4-0.8))

      if(mA>0){

        ctx.save(); ctx.globalAlpha=Math.min(1,mA)*stageAlpha

        ctx.shadowColor='#4f46e544'; ctx.shadowBlur=22

        ctx.fillStyle='rgba(255,255,255,0.97)'

        ctx.beginPath(); ctx.arc(mergeX,mergeY,38,0,Math.PI*2); ctx.fill()

        ctx.strokeStyle='#4f46e5'; ctx.lineWidth=2.5; ctx.stroke(); ctx.shadowBlur=0

        ctx.font='700 9px JetBrains Mono,monospace'; ctx.fillStyle='#4f46e5'; ctx.textAlign='center'

        ctx.fillText('TICK',mergeX,mergeY-6); ctx.fillText('MERGE',mergeX,mergeY+12)

        ctx.restore()

        // Live tick table right of merge

        const tx=mergeX+50, ty=H/2-80

        const TCOLS=['TIME','PRICE','SPREAD','VOL┬╖╧â','RSI']

        const TCW=[114,100,82,96,64]

        ctx.save(); ctx.globalAlpha=Math.min(1,mA)*stageAlpha

        ctx.font='700 10px JetBrains Mono,monospace'; ctx.fillStyle='#1e1b4b'; ctx.textAlign='left'

        let hx=tx; TCOLS.forEach((c,ci)=>{ctx.fillText(c,hx,ty);hx+=TCW[ci]})

        ctx.strokeStyle='#cbd5e1'; ctx.lineWidth=1

        ctx.beginPath(); ctx.moveTo(tx,ty+5); ctx.lineTo(tx+TCW.reduce((a,b)=>a+b,0),ty+5); ctx.stroke()

        const nShow=Math.min(6,Math.ceil(mA*6))

        for(let r=0;r<nShow;r++){

          const row=activeTicks[r]||{}, ry=ty+22+r*22

          const rd=[

            row.ts?new Date(row.ts).toISOString().slice(11,22):'--:--:--',

            row.price?`$${Number(row.price).toFixed(3)}`:'ΓÇöΓÇö',

            row.spread?.toFixed(4)||'ΓÇö',

            row.volatility?.toFixed(5)||'ΓÇö',

            row.rsi?.toFixed(1)||'ΓÇö',

          ]

          const rc=['#334155','#16a34a',

            (row.spread>0.015)?'#ef4444':'#334155',

            (row.volatility>0.00006)?'#f59e0b':'#334155',

            (row.rsi>70)?'#ef4444':(row.rsi<30)?'#7c3aed':'#334155',

          ]

          ctx.font='600 10px JetBrains Mono,monospace'; ctx.textAlign='left'

          let cx=tx; rd.forEach((d,ii)=>{ ctx.fillStyle=rc[ii]; ctx.fillText(d,cx,ry); cx+=TCW[ii] })

        }

        ctx.restore()

      }

      ctx.restore(); ctx.globalAlpha=1

      stageLabel(ctx,W,H,'STREAMS ΓÇö Binance WebSocket Feeds',sT)

    }

    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    // STAGE 2: BARS ΓÇö tick accumulation ΓåÆ 10s bar close ΓåÆ 19 features

    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    else if(vsName==='BARS'){

      const barW=Math.min(W-200,480), barX=(W-barW)/2, barY=H*0.30

      if(sT<0.32){

        // ACT A: tick accumulation progress bar

        const p=sT/0.32

        ctx.fillStyle='#f1f5f9'; ctx.beginPath(); ctx.roundRect(barX,barY,barW,22,11); ctx.fill()

        ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=1; ctx.stroke()

        const grad=ctx.createLinearGradient(barX,0,barX+barW,0)
        grad.addColorStop(0,'#7c3aed'); grad.addColorStop(1,'#4f46e5')


        ctx.fillStyle=grad; ctx.beginPath(); ctx.roundRect(barX,barY,barW*p,22,11); ctx.fill()

        ctx.font='700 10px JetBrains Mono,monospace'; ctx.fillStyle='#1e1b4b'; ctx.textAlign='center'

        ctx.fillText(`${(p*10).toFixed(1)}s / 10s  ┬╖  ACCUMULATING TICKS`,W/2,barY-14)

        const nDots=Math.floor(p*36)+2

        for(let di=0;di<Math.min(nDots,36);di++){

          const dotX=barX+10+(di/36)*(barW-20)

          const dotY=barY+44+Math.sin(di*1.4)*10

          drawAtom(ctx,dotX,dotY,WS_STREAMS[di%6].col,2.5)

        }

        ctx.font='600 9px JetBrains Mono,monospace'; ctx.fillStyle='#64748b'; ctx.textAlign='center'

        ctx.fillText(`${nDots} ticks from 6 streams ΓåÆ OHLCV + L2 + macro`,W/2,barY+68)

        stageLabel(ctx,W,H,'BARS ΓÇö Tick Accumulation',sT)

      } else if(sT<0.72){

        // ACT B: bar closed ΓåÆ 19 variable names cascade in groups

        const p=ease((sT-0.28)/0.44)

        const pulse=Math.sin(sT*Math.PI*8)*0.08+0.93

        ctx.save(); ctx.globalAlpha=pulse

        ctx.font='800 20px JetBrains Mono,monospace'; ctx.fillStyle='#16a34a'; ctx.textAlign='center'

        ctx.fillText('Γ£ô  BAR CLOSED  ΓÇö  19 VARIABLES COMPUTED',W/2,H*0.09)

        ctx.restore(); ctx.globalAlpha=1

        const colW=(W-30)/5, colH=H*0.68, colY=H*0.16

        GROUPS.forEach((g,gi)=>{

          const gx=40+gi*colW+colW/2

          const colAlpha=ease(Math.min(1,Math.max(0,(p-gi*0.18)/0.22)))

          if(colAlpha<=0) return

          ctx.globalAlpha=Math.min(1,colAlpha)

          ctx.save()

          ctx.fillStyle=g.col+'14'; ctx.strokeStyle=g.col+'66'; ctx.lineWidth=1.5

          ctx.beginPath(); ctx.roundRect(gx-colW/2+4,colY,colW-8,colH,8); ctx.fill(); ctx.stroke()

          ctx.fillStyle=g.col; ctx.fillRect(gx-colW/2+4,colY,colW-8,16)

          ctx.font='700 9px JetBrains Mono,monospace'; ctx.fillStyle='#fff'; ctx.textAlign='center'

          ctx.fillText(g.label,gx,colY+15)

          ctx.restore(); ctx.globalAlpha=1

          const varStart=gi*0.18+0.18

          const varP=Math.max(0,(p-varStart)/0.70)

          const nShow=Math.ceil(g.vars.length*Math.min(1,varP*1.4))|0

          if(nShow>0){

            ctx.globalAlpha=Math.min(1,colAlpha)

            ctx.font='600 11px JetBrains Mono,monospace'; ctx.fillStyle=g.col; ctx.textAlign='center'

            g.vars.slice(0,nShow).forEach((v,vi)=>{

              const vAlpha=ease(Math.min(1,Math.max(0,(varP*g.vars.length-vi)/1.0)))

              ctx.globalAlpha=Math.min(1,colAlpha)*vAlpha

              ctx.fillText(v,gx,colY+34+vi*18)

            })

            ctx.globalAlpha=1

          }

        })

        const totalShown=GROUPS.reduce((acc,g,gi)=>{

          const varStart=gi*0.18+0.18

          const varP=Math.max(0,(p-varStart)/0.70)

          return acc + (Math.ceil(g.vars.length*Math.min(1,varP*1.4))|0)

        },0)

        ctx.font='600 9px JetBrains Mono,monospace'; ctx.fillStyle='#64748b'; ctx.textAlign='center'

        ctx.fillText(`${Math.min(totalShown,19)} / 19 variables computed`,W/2,H*0.90)

        stageLabel(ctx,W,H,'BARS ΓÇö Features Computed',sT)

      } else {

        // ACT C: variable pills fly to graph node positions

        const p=ease((sT-0.72)/0.28)

        Object.values(nodes).forEach(n=>drawNode(ctx,n,p*0.8))

        const colW=(W-30)/5, colH=H*0.74, colY=H*0.13

        VARIABLE_NAMES.forEach((vname,vi)=>{

          const g=GROUPS.find(gr=>gr.vars.includes(vname))||GROUPS[1]

          const gi=GROUPS.indexOf(g)

          const gx=40+gi*colW+colW/2

          const gvy=colY+30+g.vars.indexOf(vname)*15

          const tn=nodes[vname]; if(!tn) return

          const vp=ease(Math.min(1,Math.max(0,(p*VARIABLE_NAMES.length-vi*0.55)/2.8)))

          if(vp>=0.97) return

          const cx2=gx+(tn.x-gx)*vp

          const cy2=gvy+(tn.y-gvy)*vp

          ctx.save(); ctx.globalAlpha=1-vp*0.4

          ctx.font='600 9px JetBrains Mono,monospace'

          const tw=ctx.measureText(abbr(vname)).width+12

          ctx.fillStyle='rgba(255,255,255,0.95)'

          ctx.beginPath(); ctx.roundRect(cx2-tw/2,cy2-8,tw,16,8); ctx.fill()

          ctx.strokeStyle=g.col; ctx.lineWidth=1.5; ctx.stroke()

          ctx.fillStyle=g.col; ctx.textAlign='center'; ctx.fillText(abbr(vname),cx2,cy2+4)

          ctx.restore(); ctx.globalAlpha=1

        })

        stageLabel(ctx,W,H,'BARS ΓÇö Variables ΓåÆ Graph Nodes',sT)

      }

    }

    else if(vsName==='PCMCI'){

      // Phase timeline: 0-0.08 nodes, then 5 layers each get 0.18 of sT

      // MACRO:0.08-0.26  MARKET:0.26-0.44  MICRO:0.44-0.62  HEALTH:0.62-0.78  TARGET:0.78-1.0

      const LAYER_PHASES=[

        {l:'macro',        t0:0.08, t1:0.26},

        {l:'market',       t0:0.26, t1:0.44},

        {l:'microstructure',t0:0.44,t1:0.62},

        {l:'health',       t0:0.62, t1:0.78},

        {l:'target',       t0:0.78, t1:1.00},

      ]

      const activeLayerPhase=LAYER_PHASES.find(p=>sT>=p.t0&&sT<p.t1)

      const activeLayerName=activeLayerPhase?.l||null

      // Draw background ΓÇö highlight the active layer column with a bright pulse

      LO.forEach(l=>{

        const x=W*LX[l], col=LC[l]

        const isActive=l===activeLayerName

        const alreadyDone=LAYER_PHASES.findIndex(p=>p.l===l)<LAYER_PHASES.findIndex(p=>p.l===activeLayerName)

        const baseA=isActive?0.18:alreadyDone?0.10:0.04

        ctx.globalAlpha=baseA; ctx.fillStyle=col; ctx.fillRect(x-W*.11,0,W*.22,H)

        if(isActive){

          // Pulsing column highlight

          const phP=LAYER_PHASES.find(p=>p.l===l)

          const layerT=(sT-phP.t0)/(phP.t1-phP.t0)

          const pulse2=Math.sin(layerT*Math.PI*4)*0.12+0.20

          ctx.globalAlpha=pulse2; ctx.fillStyle=col; ctx.fillRect(x-W*.11,0,W*.22,H)

        }

        ctx.globalAlpha=0.10; ctx.strokeStyle=col; ctx.lineWidth=1

        ctx.beginPath(); ctx.moveTo(x-W*.11,0); ctx.lineTo(x-W*.11,H); ctx.stroke()

        ctx.beginPath(); ctx.moveTo(x+W*.11,0); ctx.lineTo(x+W*.11,H); ctx.stroke()

        ctx.globalAlpha=1

      })

      // Layer labels

      ctx.font='700 10px JetBrains Mono,monospace'; ctx.textAlign='center'

      LO.forEach(l=>{

        const ns=Object.values(nodes).filter(n=>n.l===l); if(!ns.length)return

        const isActive=l===activeLayerName

        ctx.fillStyle=LC[l]+(isActive?'ff':'88'); ctx.globalAlpha=isActive?1:0.6

        ctx.fillText(l.toUpperCase(),W*LX[l],18)

        ctx.globalAlpha=1

      })

      // PCMCI header

      const hA=ease(Math.min(1,sT*6))

      ctx.globalAlpha=hA

      ctx.font='700 13px JetBrains Mono,monospace'; ctx.fillStyle='#4f46e5'; ctx.textAlign='center'

      ctx.fillText('PCMCI  ┬╖  TAU_MAX = 4  ┬╖  ALPHA = 0.05',W/2,36)

      ctx.font='600 9px JetBrains Mono,monospace'; ctx.fillStyle='#94a3b8'

      ctx.fillText('Conditional independence testing ┬╖ Granger fallback below 80 bars',W/2,51)

      ctx.globalAlpha=1

      // Nodes appear first

      const nodeP=ease(Math.min(1,sT*4))

      Object.values(nodes).forEach(n=>drawNode(ctx,n,nodeP*0.9))

      // Edges grouped by source layer, revealed layer-by-layer

      const edgesByLayer={}

      LO.forEach(l=>edgesByLayer[l]=[])

      edges.forEach(e=>{

        const sI=LI[getLayer(e.source)]??2, tI=LI[getLayer(e.target)]??2

        if(sI<tI) edgesByLayer[getLayer(e.source)]?.push(e)

      })

      let totalDrawn=0

      LAYER_PHASES.forEach(({l,t0,t1})=>{

        if(sT<t0) return  // layer not started yet

        const layerT=Math.min(1,(sT-t0)/(t1-t0))

        const layerEdges=(edgesByLayer[l]||[]).sort((a,b)=>Math.abs(b.coeff)-Math.abs(a.coeff))

        // Use gentle power curve: edges emerge slowly and visibly spaced

        const nShow=Math.floor(Math.pow(layerT,0.65)*layerEdges.length)

        const isActiveLayer=l===activeLayerName

        const col=LC[l]

        layerEdges.slice(0,nShow).forEach((e,ei)=>{

          const mx=(e.sx+e.tx)/2

          const nr=e.target==='price_return'?16:10

          const dx=e.tx-e.sx, dy=e.ty-e.sy, len=Math.sqrt(dx*dx+dy*dy)||1

          const tx2=e.tx-(dx/len)*nr, ty2=e.ty-(dy/len)*nr

          const isNew=isActiveLayer&&ei===nShow-1&&layerT<0.98

          ctx.save()

          ctx.strokeStyle=col

          ctx.lineWidth=(e.validated?Math.max(0.8,Math.abs(e.coeff)*4):0.5)*(isNew?2.5:1)

          ctx.globalAlpha=(e.validated?0.72:0.14)*(isNew?1:0.82)

          if(isNew){ctx.shadowColor=col; ctx.shadowBlur=18}

          ctx.beginPath(); ctx.moveTo(e.sx,e.sy); ctx.bezierCurveTo(mx,e.sy,mx,e.ty,tx2,ty2); ctx.stroke()

          ctx.shadowBlur=0

          if(e.validated){

            const ang=Math.atan2(ty2-e.ty,tx2-mx)

            ctx.translate(tx2,ty2); ctx.rotate(ang)

            ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(-7,-3.5); ctx.lineTo(-7,3.5); ctx.closePath()

            ctx.fillStyle=col; ctx.fill()

          }

          if(e.validated&&(isNew||Math.abs(e.coeff)>0.4)){

            ctx.restore(); ctx.save()

            ctx.globalAlpha=isNew?0.9:0.50

            ctx.font='600 8px JetBrains Mono,monospace'; ctx.fillStyle=col; ctx.textAlign='center'

            const lx=(e.sx+tx2)/2+(e.sy<e.ty?-20:20), ly=(e.sy+ty2)/2

            ctx.fillText(`lag=${e.lag}  ╬▓=${e.coeff?.toFixed(2)}`,lx,ly)

          }

          ctx.restore(); ctx.globalAlpha=1

          totalDrawn++

        })

        // Layer reveal label (while active)

        if(isActiveLayer&&layerT<0.95){

          const lx3=W*LX[l], ly3=H-70

          const labelA=Math.sin(layerT*Math.PI)*0.9+0.1

          ctx.save(); ctx.globalAlpha=labelA

          ctx.font='700 11px JetBrains Mono,monospace'; ctx.fillStyle=col; ctx.textAlign='center'

          ctx.fillText(`Γåô ${l.toUpperCase()} LAYER`,lx3,ly3)

          ctx.font='600 8px JetBrains Mono,monospace'; ctx.fillStyle=col+'88'

          ctx.fillText(`${nShow} edges`,lx3,ly3+14)

          ctx.restore(); ctx.globalAlpha=1

        }

      })

      // Progress bar (which layer is active)

      const barY=H-56; const barX=W*0.12; const barW2=W*0.76

      ctx.font='600 8px JetBrains Mono,monospace'; ctx.fillStyle='#94a3b8'; ctx.textAlign='center'

      ctx.fillText(`${totalDrawn} causal edges  ┬╖  ${activeLayerName?.toUpperCase()||'COMPLETE'}`,W/2,barY-4)

      LAYER_PHASES.forEach(({l,t0,t1},li)=>{

        const bx=barX+li*(barW2/5)+2, bw=(barW2/5)-4

        const done=sT>=t1; const active=sT>=t0&&sT<t1

        ctx.fillStyle=done?LC[l]+(active?'':' 88'):LC[l]+'22'

        ctx.fillStyle=done?LC[l]+'cc':active?LC[l]+'88':'#e2e8f0'

        ctx.beginPath(); ctx.roundRect(bx,barY,bw,5,3); ctx.fill()

        if(active){

          const fillW=bw*(sT-t0)/(t1-t0)

          ctx.fillStyle=LC[l]; ctx.beginPath(); ctx.roundRect(bx,barY,fillW,5,3); ctx.fill()

        }

        ctx.font='600 7px JetBrains Mono,monospace'; ctx.fillStyle=LC[l]; ctx.textAlign='center'

        ctx.fillText(l==='microstructure'?'MICRO':l.toUpperCase(),bx+bw/2,barY+16)

      })

      stageLabel(ctx,W,H,'PCMCI ΓÇö Layer-by-Layer Discovery',sT)

    }

    // ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó 

    // STAGE 4: HYPOTHESIS ΓÇö ALL ranked chains visible simultaneously

    // ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó ├óΓÇó 

    else if(vsName==='HYPOTHESIS'){

      drawLayerBg(ctx,W,H,nodes)

      const pulse=Math.sin(sT*Math.PI*5)*0.14+0.88

      // Colors + glow by rank (0=best) ΓÇö 12 entries for up to 12 hypotheses
      const HYP_COLS=[
        '#7c3aed','#2563eb','#0891b2','#0d9488',
        '#059669','#65a30d','#ca8a04','#ea580c',
        '#dc2626','#9f1239','#6b7280','#94a3b8',
      ]
      const HYP_GLOW=[1.0, 0.65, 0.42, 0.28, 0.20, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05]
      const HYP_LW  =[2.5,  2.0,  1.6,  1.3,  1.1,  0.9,  0.8,  0.7,  0.6,  0.55, 0.5, 0.45]

      // Build per-hypothesis edge sets (robustly parsed) ΓÇö all hypotheses, no cap
      const hypList=activeHyps.map((h,hi)=>({

        h, hi,

        col: h.is_escape_valve?'#f59e0b':(HYP_COLS[hi]||'#94a3b8'),

        glow: HYP_GLOW[hi]||0.04,

        lw:   HYP_LW[hi]||0.4,

        // Priority: cs (selected chain) for rank-0 only, then hypPairs

        pairs: (cs.size>0&&hi===0) ? cs : hypPairs(h),

      }))

      // All nodes/edges that are in ANY chain

      const allPairs=new Set(hypList.flatMap(d=>[...d.pairs]))

      const allNodes=new Set([...allPairs].flatMap(p=>p.split('|')))

      // 1. Dim everything not in any chain

      edges.forEach(e=>{

        if(allPairs.has(`${e.source}|${e.target}`)) return

        const sI=LI[getLayer(e.source)]??2, tI=LI[getLayer(e.target)]??2; if(sI>=tI) return

        const col=LC[getLayer(e.source)]||'#94a3b8', mx=(e.sx+e.tx)/2

        ctx.save(); ctx.strokeStyle=col; ctx.lineWidth=0.5; ctx.globalAlpha=0.05

        ctx.beginPath(); ctx.moveTo(e.sx,e.sy); ctx.bezierCurveTo(mx,e.sy,mx,e.ty,e.tx,e.ty); ctx.stroke()

        ctx.restore(); ctx.globalAlpha=1

      })

      Object.values(nodes).forEach(n=>{ if(!allNodes.has(n.name)) drawNode(ctx,n,0.08) })

      // 2. Draw chains lowest-rank first so best sits on top
      ;[...hypList].reverse().forEach(({h,pairs,col,glow,lw})=>{

        if(!pairs.size) return

        // Real edges + virtual edges for pairs not in graph
        const hlEdges = []
        pairs.forEach(pair => {
          const [src, tgt] = pair.split('|')
          const real = edges.find(e => e.source===src && e.target===tgt)
          if(real){
            hlEdges.push(real)
          } else {
            const sn = nodes[src], tn2 = nodes[tgt]
            if(sn && tn2) hlEdges.push({
              source:src, target:tgt,
              sx:sn.x, sy:sn.y, tx:tn2.x, ty:tn2.y,
              lag:'?', coeff:null, virtual:true
            })
          }
        })

        if(!hlEdges.length) return

        const hlNodes=new Set(hlEdges.flatMap(e=>[e.source,e.target]))

        // Edges

        hlEdges.forEach(e=>{

          const mx=(e.sx+e.tx)/2

          const nr=e.target==='price_return'?20:14

          const dx=e.tx-e.sx, dy=e.ty-e.sy, len=Math.sqrt(dx*dx+dy*dy)||1

          const tx2=e.tx-(dx/len)*nr, ty2=e.ty-(dy/len)*nr

          ctx.save()

          // Outer glow

          ctx.strokeStyle=col; ctx.lineWidth=9; ctx.globalAlpha=glow*pulse*0.13

          ctx.shadowColor=col; ctx.shadowBlur=24

          ctx.beginPath(); ctx.moveTo(e.sx,e.sy); ctx.bezierCurveTo(mx,e.sy,mx,e.ty,tx2,ty2); ctx.stroke()

          ctx.shadowBlur=0

          // Inner line

          ctx.lineWidth=lw; ctx.globalAlpha=glow*pulse*0.92

          ctx.shadowColor=col; ctx.shadowBlur=glow>0.5?14:5

          ctx.beginPath(); ctx.moveTo(e.sx,e.sy); ctx.bezierCurveTo(mx,e.sy,mx,e.ty,tx2,ty2); ctx.stroke()

          ctx.shadowBlur=0

          const ang=Math.atan2(ty2-e.ty,tx2-mx)

          ctx.translate(tx2,ty2); ctx.rotate(ang)

          ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(glow>0.5?-10:-7,glow>0.5?-5:-3.5); ctx.lineTo(glow>0.5?-10:-7,glow>0.5?5:3.5); ctx.closePath()

          ctx.fillStyle=col; ctx.globalAlpha=glow*pulse; ctx.fill()

          ctx.restore(); ctx.globalAlpha=1

          if(glow>=0.5){

            ctx.font='700 9px JetBrains Mono,monospace'; ctx.fillStyle=col; ctx.textAlign='center'

            ctx.fillText(`lag=${e.lag}  ╬▓=${e.coeff?.toFixed(3)||'?'}`,(e.sx+e.tx)/2+(e.sy<e.ty?-22:22),(e.sy+e.ty)/2)

          }

        })

        hlNodes.forEach(name=>{

          const n=nodes[name]; if(!n) return

          const r=n.name==='price_return'?20:14

          ctx.save()

          ctx.beginPath(); ctx.arc(n.x,n.y,r+7,0,Math.PI*2); ctx.fillStyle=col+'14'; ctx.fill()

          ctx.shadowColor=col; ctx.shadowBlur=glow>0.5?22:8

          ctx.beginPath(); ctx.arc(n.x,n.y,r,0,Math.PI*2); ctx.fillStyle=col+'20'; ctx.fill()

          ctx.strokeStyle=col; ctx.lineWidth=glow>0.5?2.2:1; ctx.globalAlpha=glow*pulse; ctx.stroke()

          ctx.shadowBlur=0

          if(glow>=0.5){

            ctx.font='700 9px JetBrains Mono,monospace'; ctx.fillStyle=col; ctx.textAlign='center'

            ctx.fillText(abbr(name),n.x,n.y+(n.name==='price_return'?5:r+13))

          }

          ctx.restore(); ctx.globalAlpha=1

        })

      })


      const best0=hypList[0]

      if(best0?.pairs.size){

        const bEdges=[];best0.pairs.forEach(pair=>{const [s2,t2]=pair.split('|');const re2=edges.find(e=>e.source===s2&&e.target===t2);if(re2){bEdges.push(re2)}else{const sn2=nodes[s2],tn3=nodes[t2];if(sn2&&tn3)bEdges.push({source:s2,target:t2,sx:sn2.x,sy:sn2.y,tx:tn3.x,ty:tn3.y})}})

        if(bEdges.length){

          const t=(sT*1.4)%1

          const ei=Math.floor(t*bEdges.length)

          const et=(t*bEdges.length)%1

          const e=bEdges[Math.min(ei,bEdges.length-1)]

          const pos=bzPt(e.sx,e.sy,e.tx,e.ty,et)

          drawAtom(ctx,pos.x,pos.y,best0.col,8)

        }

      }

      const legW=210, legX=W-legW-8, legY=10

      const legH=18+hypList.length*22+6

      ctx.save()

      ctx.fillStyle='rgba(255,255,255,0.96)'

      ctx.shadowColor='#00000010'; ctx.shadowBlur=12

      ctx.beginPath(); ctx.roundRect(legX,legY,legW,legH,8); ctx.fill()

      ctx.shadowBlur=0; ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=1; ctx.stroke()

      ctx.restore()

      ctx.font='700 8px JetBrains Mono,monospace'; ctx.fillStyle='#475569'; ctx.textAlign='left'

      ctx.fillText(`HYPOTHESES (${activeHyps.length})`,legX+10,legY+13)

      hypList.forEach(({h,hi,col,pairs})=>{

        const ry=legY+20+hi*22

        ctx.save()

        ctx.fillStyle=col

        ctx.beginPath(); ctx.roundRect(legX+8,ry+1,8,8,2); ctx.fill()

        ctx.font='600 8px JetBrains Mono,monospace'; ctx.fillStyle='#334155'; ctx.textAlign='left'

        const rawChain=typeof h.chain==='string'?h.chain

          :Array.isArray(h.chain_edges)?h.chain_edges.map(e=>`${abbr(e.source)}ΓåÆ${abbr(e.target)}`).join(' ')

          :Array.isArray(h.chain)?h.chain.map(e=>`${abbr(e.source)}ΓåÆ${abbr(e.target)}`).join(' '):''

        const dispChain=rawChain.replace(/\(lag=\d+\)ΓåÆ/g,'ΓåÆ').slice(0,24)

        ctx.fillText(`H${hi+1} ${dispChain}`,legX+20,ry+9)

        if(!pairs.size) { ctx.fillStyle='#94a3b8'; ctx.fillText('ΓÇö no match',legX+20,ry+9) }

        if(h.composite_score!=null){

          ctx.font='700 8px JetBrains Mono,monospace'; ctx.fillStyle=col; ctx.textAlign='right'

          ctx.fillText(h.composite_score.toFixed(3),legX+legW-8,ry+9)

        }

        ctx.restore()

      })

      stageLabel(ctx,W,H,'HYPOTHESIS ΓÇö Chain Selection',sT)

    }

    else if(vsName==='OPTIMIZER'){

      ctx.fillStyle='#f8fafc'; ctx.fillRect(0,0,W,H)

      const regimes = opt?.regimes||[]
      const activeR = (ld?.hold_reason||'').includes('volatile')?'volatile'
        :(ld?.hold_reason||'').includes('trending')?'trending':'calm'
      const rd2     = regimes.find(r=>r.regime===activeR)||regimes[regimes.length-1]
      const pop     = rd2?.population||[]
      const recentScores = rd2?.history_scores||[]
      const CCOLS   = ['#7c3aed','#2563eb','#16a34a']
      const USE_MOCK = !pop.length
      const candList = USE_MOCK
        ? [{k_runs:22,min_runs:5, threshold:0.025,tau_max:3,pcmci_alpha:0.08,recent_score:null},
           {k_runs:40,min_runs:10,threshold:0.050,tau_max:4,pcmci_alpha:0.05,recent_score:null},
           {k_runs:58,min_runs:16,threshold:0.080,tau_max:6,pcmci_alpha:0.02,recent_score:null}]
        : pop.slice(0,3)

      // ── Layout ──────────────────────────────────────────────────
      const SPLIT  = Math.floor(W * 0.58)   // left meta-history panel
      const RGAP   = 10
      const VPAD   = 10
      const BOT    = 28

      // ══════════════════════════════════════════════════════════
      // LEFT — MetaOptimizer score history + 3 candidate particles
      // ══════════════════════════════════════════════════════════
      const lW = SPLIT - 6, lH = H - VPAD - BOT, lX = 0, lY = VPAD

      ctx.save()
      ctx.fillStyle='rgba(255,255,255,0.97)'
      ctx.shadowColor='rgba(124,58,237,0.12)'; ctx.shadowBlur=18
      ctx.beginPath(); ctx.roundRect(lX+4,lY,lW-4,lH,8); ctx.fill()
      ctx.shadowBlur=0; ctx.strokeStyle='#7c3aed33'; ctx.lineWidth=1.2; ctx.stroke()
      ctx.restore()

      // Purple top bar
      ctx.fillStyle='#7c3aed'
      ctx.beginPath()
      ctx.roundRect(lX+4,lY,lW-4,4,{upperLeft:8,upperRight:8,lowerLeft:0,lowerRight:0})
      ctx.fill()

      // Regime badge
      const regCol=activeR==='volatile'?'#ea580c':activeR==='trending'?'#2563eb':'#16a34a'
      ctx.save()
      ctx.fillStyle=regCol+'18'; ctx.strokeStyle=regCol+'66'; ctx.lineWidth=1
      ctx.beginPath(); ctx.roundRect(lX+10,lY+10,70,17,4); ctx.fill(); ctx.stroke()
      ctx.font='700 8px JetBrains Mono,monospace'; ctx.fillStyle=regCol; ctx.textAlign='center'
      ctx.fillText(activeR.toUpperCase(),lX+45,lY+21)
      ctx.restore()

      // Title
      ctx.font='700 11px JetBrains Mono,monospace'; ctx.fillStyle='#7c3aed'; ctx.textAlign='center'
      ctx.fillText('META-OPTIMIZER SCORE HISTORY',lX+lW/2,lY+21)
      ctx.font='500 7px JetBrains Mono,monospace'; ctx.fillStyle='#94a3b8'
      ctx.fillText('each point = one hill-climb step · particles = candidate scores now',lX+lW/2,lY+31)

      // Sparkline graph area
      const spPad=14, spTop=lY+40, spH=lH-54, spX=lX+spPad+14, spW=lW-spPad*2-14

      // Axes
      ctx.save(); ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=0.8
      ctx.beginPath(); ctx.moveTo(spX,spTop); ctx.lineTo(spX,spTop+spH); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(spX,spTop+spH); ctx.lineTo(spX+spW,spTop+spH); ctx.stroke()
      ctx.restore()

      // Compute scale — use real scores if available, else mock sine wave
      const hasSc = recentScores.length > 1
      const mockN = 40
      const mockScores = Array.from({length:mockN},(_,i)=>{
        const x=i/mockN
        return 0.12+0.55/(1+Math.exp(-8*(x-0.38)))+0.04*Math.sin(x*Math.PI*6)
      })
      const displayScores = hasSc ? recentScores : mockScores
      const mn=Math.min(...displayScores), mx=Math.max(...displayScores), rng=mx-mn||0.01
      const toY=(s)=>spTop+spH-((s-mn)/rng)*spH
      const toX=(i)=>spX+i*(spW/(displayScores.length-1))

      // Gradient fill
      ctx.save()
      const grd2=ctx.createLinearGradient(0,spTop,0,spTop+spH)
      grd2.addColorStop(0,'#7c3aed28'); grd2.addColorStop(1,'#7c3aed04')
      ctx.beginPath()
      displayScores.forEach((s,i)=>{ i===0?ctx.moveTo(toX(i),toY(s)):ctx.lineTo(toX(i),toY(s)) })
      ctx.lineTo(toX(displayScores.length-1),spTop+spH); ctx.lineTo(spX,spTop+spH); ctx.closePath()
      ctx.fillStyle=grd2; ctx.fill()
      ctx.restore()

      // Line
      ctx.save(); ctx.strokeStyle='#7c3aed'; ctx.lineWidth=2.2; ctx.lineJoin='round'
      ctx.beginPath()
      displayScores.forEach((s,i)=>{ i===0?ctx.moveTo(toX(i),toY(s)):ctx.lineTo(toX(i),toY(s)) })
      ctx.stroke(); ctx.restore()

      // Step dots (small, grey)
      ctx.save()
      displayScores.forEach((s,i)=>{
        ctx.globalAlpha=0.5
        ctx.beginPath(); ctx.arc(toX(i),toY(s),2,0,Math.PI*2)
        ctx.fillStyle='#7c3aed66'; ctx.fill()
      })
      ctx.restore(); ctx.globalAlpha=1

      // Y axis ticks
      ctx.font='500 6px JetBrains Mono,monospace'; ctx.fillStyle='#94a3b8'; ctx.textAlign='right'
      ctx.fillText(mx.toFixed(3),spX-3,spTop+5)
      ctx.fillText(mn.toFixed(3),spX-3,spTop+spH+4)
      ctx.font='500 6px JetBrains Mono,monospace'; ctx.fillStyle='#94a3b8'; ctx.textAlign='center'
      ctx.fillText('hill-climb steps →',lX+lW/2,lY+lH-6)

      // ── SINGLE PARTICLE riding the score history, color = closest candidate ──
      // Moves smoothly left→right through every point in the curve.
      // At each step the particle takes the color of whichever candidate's
      // recent_score is closest to that history value — revealing which
      // candidate "owns" each region. Trail preserves the color history.
      const TRAIL_LEN = 40

      // Helper: color for a given score value (nearest candidate)
      const colorForScore = (scoreVal)=>{
        const scores = candList.map((c,ci)=>({
          ci,
          dist: Math.abs((c.recent_score??
            // fallback: use simulated peak for mock
            (0.20+Math.min(1,Math.max(0,(c.threshold||0.02)/0.10))*0.60)) - scoreVal)
        }))
        scores.sort((a,b)=>a.dist-b.dist)
        return CCOLS[scores[0].ci]
      }

      // Current particle position — slow, smooth, non-looping feel (wraps at end)
      const totalPts = displayScores.length
      const pFrac    = (sT * 0.18) % 1           // slow constant speed, loops cleanly
      const pIdx     = Math.min(totalPts-1, Math.floor(pFrac * totalPts))
      const pScore   = displayScores[pIdx]
      const pCol     = colorForScore(pScore)
      const ppx      = toX(pIdx)
      const ppy      = toY(pScore)

      // Comet trail — draw backwards from current position
      ctx.save()
      for(let ti=TRAIL_LEN;ti>0;ti--){
        const trIdx = Math.max(0, pIdx - ti)
        const trS   = displayScores[trIdx]
        const trCol = colorForScore(trS)
        const alpha = (1 - ti/TRAIL_LEN) * 0.65
        ctx.globalAlpha = alpha
        ctx.beginPath()
        ctx.arc(toX(trIdx), toY(trS), 1.8 + (1-ti/TRAIL_LEN)*1.2, 0, Math.PI*2)
        ctx.fillStyle = trCol
        ctx.fill()
      }
      ctx.restore(); ctx.globalAlpha=1

      // Outer glow
      const pulse = Math.sin(sT*Math.PI*6)*0.15+0.85
      ctx.save()
      ctx.shadowColor=pCol; ctx.shadowBlur=28
      ctx.globalAlpha=pulse*0.22
      ctx.beginPath(); ctx.arc(ppx,ppy,18,0,Math.PI*2); ctx.fillStyle=pCol; ctx.fill()
      // Core
      ctx.shadowBlur=14; ctx.globalAlpha=pulse
      ctx.beginPath(); ctx.arc(ppx,ppy,6,0,Math.PI*2); ctx.fillStyle=pCol; ctx.fill()
      ctx.shadowBlur=0; ctx.restore(); ctx.globalAlpha=1

      // Floating chip: which candidate colour + current score
      const chipW=76, chipH=16
      const chipX=Math.min(toX(totalPts-1)-chipW, Math.max(spX, ppx-chipW/2))
      const chipY=ppy-26
      const ownerCi=CCOLS.indexOf(pCol)
      ctx.save()
      ctx.fillStyle='rgba(255,255,255,0.95)'; ctx.strokeStyle=pCol; ctx.lineWidth=1.2
      ctx.beginPath(); ctx.roundRect(chipX,chipY,chipW,chipH,4); ctx.fill(); ctx.stroke()
      ctx.font='700 7.5px JetBrains Mono,monospace'; ctx.fillStyle=pCol; ctx.textAlign='center'
      ctx.fillText(`C${ownerCi>=0?ownerCi:'?'}  ${pScore.toFixed(4)}`, chipX+chipW/2, chipY+11)
      ctx.restore()

      // Candidate colour legend (small dots + labels, bottom of left panel)
      candList.forEach((cand,ci)=>{
        const col=CCOLS[ci]
        const lx=lX+14+ci*78, ly=lY+lH-14
        ctx.beginPath(); ctx.arc(lx+4,ly-3,4,0,Math.PI*2); ctx.fillStyle=col; ctx.fill()
        ctx.font='600 7px JetBrains Mono,monospace'; ctx.fillStyle=col; ctx.textAlign='left'
        ctx.fillText(`C${ci} k=${cand.k_runs??'—'}`,lx+12,ly)
      })

      // Latest score badge (top-right of panel)
      if(hasSc){
        const lastS=recentScores[recentScores.length-1]
        const prevS=recentScores[recentScores.length-2]
        const up=lastS>prevS
        ctx.font='800 12px JetBrains Mono,monospace'; ctx.fillStyle='#7c3aed'; ctx.textAlign='right'
        ctx.fillText(lastS.toFixed(4),lX+lW-8,lY+21)
        ctx.font='700 9px JetBrains Mono,monospace'; ctx.fillStyle=up?'#16a34a':'#94a3b8'
        ctx.fillText(up?'↑':'→',lX+lW-8,lY+32)
      } else {
        ctx.font='500 8px JetBrains Mono,monospace'; ctx.fillStyle='#cbd5e1'; ctx.textAlign='center'
        ctx.fillText('awaiting data…',lX+lW/2,spTop+spH/2)
      }

      // ══════════════════════════════════════════════════════════
      // RIGHT — 3 candidate hill-climbing score landscape panels
      // ══════════════════════════════════════════════════════════
      const rStartX = SPLIT + RGAP
      const PAD_L   = 44
      const PAD_R   = 10
      const rW      = W - rStartX - PAD_R
      const rTOP    = VPAD + 20
      const totalPH = H - rTOP - BOT
      const GAP     = 6
      const pH      = Math.floor((totalPH - GAP*2)/3)

      ctx.font='700 10px JetBrains Mono,monospace'; ctx.fillStyle='#334155'; ctx.textAlign='center'
      ctx.fillText('HILL-CLIMBING  ·  CANDIDATE LANDSCAPES', rStartX+rW/2, VPAD+13)

      const makeCurve=(cand,ci)=>{
        const thN=Math.min(1,Math.max(0,(cand.threshold  ||0.02)/0.10))
        const mnN=Math.min(1,Math.max(0,(cand.min_runs   ||5 )/20  ))
        const alN=Math.min(1,Math.max(0,(cand.pcmci_alpha||0.05)/0.10))
        const peakH=0.20+thN*0.60, rise=3+mnN*11, noise=0.01+(1-alN)*0.14, bF=4+ci*1.9+(1-mnN)*3
        return (x)=>{
          const base=peakH/(1+Math.exp(-rise*(x-0.40)))
          const decay=Math.exp(-x*3.0)
          const bumps=noise*decay*(
            Math.sin(bF*x*Math.PI*2+ci*1.7)*0.55+
            Math.sin(bF*2.1*x*Math.PI*2+ci*0.9)*0.30+
            Math.sin(bF*3.3*x*Math.PI*2+ci*2.5)*0.15)
          const jitter=Math.max(0,(x-0.68)/0.32)*alN*0.022*Math.sin(x*Math.PI*20+ci*2.9)
          return Math.min(0.94,Math.max(0.03,base+bumps+jitter))
        }
      }

      candList.forEach((cand,ci)=>{
        const col=CCOLS[ci], panelY=rTOP+ci*(pH+GAP)
        const fn=makeCurve(cand,ci)
        const kN=Math.min(1,Math.max(0,(cand.k_runs||20)/60))
        const thN=Math.min(1,Math.max(0,(cand.threshold||0.02)/0.10))
        const peakH=0.20+thN*0.60
        const gX=rStartX+PAD_L, gY=panelY+20, gW=rW-PAD_L-PAD_R, gH=pH-26

        // Panel bg
        ctx.save()
        ctx.fillStyle='rgba(255,255,255,0.97)'
        ctx.shadowColor=col+'18'; ctx.shadowBlur=8
        ctx.beginPath(); ctx.roundRect(rStartX+PAD_L-2,panelY,rW-PAD_L-PAD_R+4,pH,6); ctx.fill()
        ctx.shadowBlur=0; ctx.strokeStyle=col+'55'; ctx.lineWidth=1.2; ctx.stroke()
        ctx.restore()
        ctx.fillStyle=col
        ctx.beginPath()
        ctx.roundRect(rStartX+PAD_L-2,panelY,3,pH,{upperLeft:6,lowerLeft:6,upperRight:0,lowerRight:0})
        ctx.fill()

        // Title strip
        ctx.fillStyle=col+'18'; ctx.fillRect(rStartX+PAD_L+1,panelY,rW-PAD_L-PAD_R,18)
        ctx.font='700 8px JetBrains Mono,monospace'; ctx.fillStyle=col; ctx.textAlign='left'
        ctx.fillText(`C${ci}`,rStartX+PAD_L+8,panelY+12)
        ctx.font='500 7px JetBrains Mono,monospace'; ctx.fillStyle='#334155'
        const ps=`k=${cand.k_runs??'—'}  th=${cand.threshold?.toFixed(3)??'—'}  min=${cand.min_runs??'—'}  τ=${cand.tau_max??'—'}  α=${cand.pcmci_alpha?.toFixed(3)??'—'}`
        ctx.fillText(ps,rStartX+PAD_L+28,panelY+12)
        if(cand.recent_score!=null){
          ctx.font='700 8px JetBrains Mono,monospace'; ctx.fillStyle=col; ctx.textAlign='right'
          ctx.fillText(`score ${cand.recent_score.toFixed(4)}`,rStartX+PAD_L+gW-4,panelY+12)
        }

        // Curve
        const NPTS=120, cpts=[]
        for(let i=0;i<=NPTS;i++){ const x=i/NPTS; cpts.push({px:gX+x*gW,py:gY+gH-fn(x)*gH}) }
        ctx.save()
        ctx.beginPath(); cpts.forEach(({px,py},i)=>i===0?ctx.moveTo(px,py):ctx.lineTo(px,py))
        ctx.lineTo(gX+gW,gY+gH); ctx.lineTo(gX,gY+gH); ctx.closePath()
        const grd=ctx.createLinearGradient(0,gY,0,gY+gH)
        grd.addColorStop(0,col+'24'); grd.addColorStop(1,col+'04')
        ctx.fillStyle=grd; ctx.fill(); ctx.restore()
        ctx.save(); ctx.beginPath(); cpts.forEach(({px,py},i)=>i===0?ctx.moveTo(px,py):ctx.lineTo(px,py))
        ctx.strokeStyle=col+'bb'; ctx.lineWidth=1.8; ctx.stroke(); ctx.restore()

        // Peak guide
        const peakPY=gY+gH-peakH*gH
        ctx.save(); ctx.setLineDash([3,4]); ctx.strokeStyle=col+'33'; ctx.lineWidth=0.8
        ctx.beginPath(); ctx.moveTo(gX,peakPY); ctx.lineTo(gX+gW,peakPY); ctx.stroke()
        ctx.setLineDash([]); ctx.restore()

        // Axes
        ctx.save(); ctx.strokeStyle='#d1d5db'; ctx.lineWidth=0.8
        ctx.beginPath(); ctx.moveTo(gX,gY); ctx.lineTo(gX,gY+gH); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(gX,gY+gH); ctx.lineTo(gX+gW,gY+gH); ctx.stroke()
        ctx.restore()
        ctx.font='600 6px JetBrains Mono,monospace'; ctx.fillStyle='#94a3b8'; ctx.textAlign='right'
        ctx.fillText(peakH.toFixed(2),rStartX+PAD_L-3,peakPY+3)
        ctx.fillText('0.00',rStartX+PAD_L-3,gY+gH+3)

        // Animated particle + trail
        const speed=0.18+kN*0.82
        const pX=((sT*speed+ci*0.30)%1), pY=fn(pX), pdx=gX+pX*gW, pdy=gY+gH-pY*gH
        ctx.save()
        for(let ti=36;ti>0;ti--){
          const trX=Math.max(0,pX-ti/36*0.22)
          ctx.globalAlpha=(1-ti/36)*0.48
          ctx.beginPath(); ctx.arc(gX+trX*gW,gY+gH-fn(trX)*gH,2,0,Math.PI*2)
          ctx.fillStyle=col; ctx.fill()
        }
        ctx.restore(); ctx.globalAlpha=1
        const pulse=Math.sin(sT*Math.PI*8+ci*1.5)*0.12+0.90
        ctx.save(); ctx.shadowColor=col; ctx.shadowBlur=20
        ctx.beginPath(); ctx.arc(pdx,pdy,9,0,Math.PI*2)
        ctx.fillStyle=col+'20'; ctx.globalAlpha=pulse; ctx.fill()
        ctx.shadowBlur=12
        ctx.beginPath(); ctx.arc(pdx,pdy,4.5,0,Math.PI*2)
        ctx.fillStyle=col; ctx.globalAlpha=pulse; ctx.fill()
        ctx.shadowBlur=0; ctx.restore(); ctx.globalAlpha=1
        ctx.font='700 7px JetBrains Mono,monospace'; ctx.fillStyle=col
        ctx.textAlign=pX>0.82?'right':'left'
        ctx.fillText(pY.toFixed(3),pdx+(pX>0.82?-10:10),pdy-6)
        ctx.font='500 6px JetBrains Mono,monospace'; ctx.fillStyle=col+'88'; ctx.textAlign='right'
        ctx.fillText(`${Math.round(speed*100)}% spd`,rStartX+PAD_L+gW-2,panelY+pH-4)
      })

      ctx.font='600 7px JetBrains Mono,monospace'; ctx.fillStyle='#94a3b8'; ctx.textAlign='center'
      ctx.fillText('hill-climbing iterations →',rStartX+rW/2,H-10)
      if(USE_MOCK){
        ctx.font='500 6px JetBrains Mono,monospace'; ctx.fillStyle='#f59e0b88'; ctx.textAlign='right'
        ctx.fillText('mock params — awaiting optimizer',W-PAD_R,H-10)
      }

      stageLabel(ctx,W,H,'OPTIMIZER — Score History · Hill-Climb Landscapes',sT)

    }

    else if(vsName==='TWIN'){

      drawLayerBg(ctx,W,H,nodes)

      drawGraph(ctx,nodes,edges,cs,0.16,0.45)

      const best=activeHyps[0]

      // Robustly get chain edges: chain_edges array > chain array > parse chain string > fallback

      const rawChainEdges=

        Array.isArray(best?.chain_edges)&&best.chain_edges.length>0 ? best.chain_edges

        : Array.isArray(best?.chain)&&best.chain.length>0&&typeof best.chain[0]==='object' ? best.chain

        : []

      let chainEdges=rawChainEdges

        .map(e=>({ ...e, sn:nodes[e.source], tn:nodes[e.target] }))

        .filter(e=>e.sn&&e.tn)

      // String fallback: parse chain summary, find matching graph edges

      if(!chainEdges.length && typeof best?.chain==='string'){

        const pairs=chainSet(best.chain)

        chainEdges=edges.filter(e=>pairs.has(`${e.source}|${e.target}`))

          .map(e=>({ ...e, sn:nodes[e.source], tn:nodes[e.target] })).filter(e=>e.sn&&e.tn)

      }

      // Final fallback: all validated edges pointing to price_return

      const displayEdges=chainEdges.length>0 ? chainEdges

        : edges.filter(e=>e.target==='price_return'&&e.validated)

               .map(e=>({ ...e, sn:nodes[e.source], tn:nodes[e.target] }))

               .filter(e=>e.sn&&e.tn)

      if(displayEdges.length){

        // Highlight chain edges

        displayEdges.forEach(e=>{

          const col='#7c3aed', mx=(e.sn.x+e.tn.x)/2

          const dx=e.tn.x-e.sn.x, dy=e.tn.y-e.sn.y, len=Math.sqrt(dx*dx+dy*dy)||1

          const tx2=e.tn.x-(dx/len)*14, ty2=e.tn.y-(dy/len)*14

          ctx.save(); ctx.shadowColor=col; ctx.shadowBlur=10

          ctx.strokeStyle=col; ctx.lineWidth=2.5; ctx.globalAlpha=0.6

          ctx.beginPath(); ctx.moveTo(e.sn.x,e.sn.y); ctx.bezierCurveTo(mx,e.sn.y,mx,e.tn.y,tx2,ty2); ctx.stroke()

          ctx.restore(); ctx.globalAlpha=1

        })

        const predDir=best?.predicted_direction||'up'

        const predMag=best?.predicted_magnitude||0.003

        const divAmp=Math.min(36,Math.max(8,predMag*5000))

        // Determine alignment: compare predicted direction vs actual price movement.

        // Use latest tick price_return sign if available; otherwise the hypothesis

        // reaching TWIN already implies a valid signal so default to aligned.

        const latestReturn=ld?.price_return ?? (

          propsRef.current.ticks?.length >= 2

            ? (propsRef.current.ticks.at(-1)?.price||0) - (propsRef.current.ticks.at(-2)?.price||0)

            : null

        )

        const actualDir = latestReturn == null ? null

          : latestReturn > 0 ? 'up' : latestReturn < 0 ? 'down' : null

        // sameDir: true = aligned (small offset), false = diverging (large offset)

        // Default to aligned when we can't measure ΓÇö the hypothesis passed ranking to reach here

        const sameDir = actualDir == null ? true : actualDir === predDir

        const actualOff=sameDir?divAmp*0.18:divAmp

        // ΓÇö EXPECTED path: indigo particle

        displayEdges.forEach((e,ei)=>{

          const t=((sT*1.4+ei*0.24)%1)

          const pos=bzPt(e.sn.x,e.sn.y,e.tn.x,e.tn.y,t)

          ctx.save()

          ctx.beginPath(); ctx.arc(pos.x,pos.y,12,0,Math.PI*2)

          ctx.fillStyle='#7c3aed1a'; ctx.fill()

          ctx.shadowColor='#7c3aed'; ctx.shadowBlur=22

          ctx.beginPath(); ctx.arc(pos.x,pos.y,5,0,Math.PI*2)

          ctx.fillStyle='#7c3aed'; ctx.fill()

          ctx.restore(); ctx.globalAlpha=1

        })

        // ΓÇö ACTUAL path: amber particle, offset in y

        displayEdges.forEach((e,ei)=>{

          const t=((sT*1.4+ei*0.24+0.15)%1)

          const offY=actualOff*Math.sin((t+ei*0.3)*Math.PI)

          const pos=bzPt(e.sn.x,e.sn.y+offY*0.4,e.tn.x,e.tn.y+offY,t)

          ctx.save()

          ctx.beginPath(); ctx.arc(pos.x,pos.y,11,0,Math.PI*2)

          ctx.fillStyle='#f59e0b1a'; ctx.fill()

          ctx.shadowColor='#f59e0b'; ctx.shadowBlur=20

          ctx.beginPath(); ctx.arc(pos.x,pos.y,5,0,Math.PI*2)

          ctx.fillStyle='#f59e0b'; ctx.fill()

          ctx.restore(); ctx.globalAlpha=1

        })

        // Legend (top-left)

        const lx=16, ly=H*0.10

        ctx.save()

        ctx.fillStyle='rgba(255,255,255,0.96)'; ctx.shadowColor='#00000015'; ctx.shadowBlur=12

        ctx.beginPath(); ctx.roundRect(lx,ly,200,78,8); ctx.fill()

        ctx.shadowBlur=0; ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=1; ctx.stroke()

        ctx.restore()

        ctx.fillStyle='#7c3aed'; ctx.beginPath(); ctx.arc(lx+16,ly+20,5,0,Math.PI*2); ctx.fill()

        ctx.font='600 10px JetBrains Mono,monospace'; ctx.fillStyle='#1e1b4b'; ctx.textAlign='left'

        ctx.fillText('Expected (causal chain)',lx+28,ly+24)

        ctx.fillStyle='#f59e0b'; ctx.beginPath(); ctx.arc(lx+16,ly+44,5,0,Math.PI*2); ctx.fill()

        ctx.fillStyle=sameDir?'#16a34a':'#ef4444'

        ctx.fillText('Actual path',lx+28,ly+48)

        ctx.font='600 8px JetBrains Mono,monospace'; ctx.fillStyle='#64748b'

        ctx.fillText(`pred: ${predDir}  |  conf ┬▒${(predMag*200).toFixed(3)}%`,lx+10,ly+66)

        // Prediction badge near price_return node

        const pn=nodes['price_return']

        if(pn&&best){

          const pulse=Math.sin(sT*Math.PI*6)*0.12+0.9

          ctx.save(); ctx.globalAlpha=pulse

          ctx.font='700 13px JetBrains Mono,monospace'; ctx.fillStyle='#7c3aed'; ctx.textAlign='center'

          ctx.fillText(`${predDir==='up'?'Γåæ':'Γåô'} ${((predMag||0)*100).toFixed(3)}%  `,pn.x,pn.y-30)

          ctx.restore(); ctx.globalAlpha=1

        }

      }

      stageLabel(ctx,W,H,'TWIN ΓÇö Expected vs Actual',sT)

    }

    // ├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É

    // STAGE 7: DECISION ΓÇö trade committed or hold (with reason)

    // ├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É

    else if(vsName==='DECISION'){

      drawLayerBg(ctx,W,H,nodes)

      drawGraph(ctx,nodes,edges,cs,0.38,0.85)

      const best=activeHyps[0]

      const traded=ld?.hold_reason==='traded'||cs.size>0

      const isEV=best?.is_escape_valve

      const reason=ld?.hold_reason||'observing'

      const dcol=traded?'#16a34a':isEV?'#f59e0b':'#64748b'

      const dw=360,dh=100,dx2=(W-dw)/2,dy2=(H-dh)/2

      ctx.save()

      ctx.shadowColor=dcol+'55'; ctx.shadowBlur=28

      ctx.fillStyle='rgba(255,255,255,0.97)'

      ctx.beginPath(); ctx.roundRect(dx2,dy2,dw,dh,12); ctx.fill()

      ctx.shadowBlur=0; ctx.strokeStyle=dcol+'99'; ctx.lineWidth=2; ctx.stroke()

      ctx.fillStyle=dcol; ctx.fillRect(dx2,dy2,dw,18)

      ctx.restore()

      ctx.font='700 11px JetBrains Mono,monospace'; ctx.fillStyle='#fff'; ctx.textAlign='left'

      ctx.fillText(traded?'├óΓÇô┬▓ TRADE EXECUTED':isEV?'├ó┼í┬í ESCAPE VALVE FIRED':'├óΓÇö┬É HOLD',dx2+12,dy2+13)

      ctx.font='600 10px JetBrains Mono,monospace'; ctx.fillStyle='#334155'

      ctx.fillText(reason.slice(0,44),dx2+12,dy2+36)

      if(best){

        ctx.fillText(

          `score=${best.composite_score?.toFixed(4)}  dir=${best.predicted_direction}  mag=${((best.predicted_magnitude||0)*100).toFixed(3)}%`,

          dx2+12,dy2+56

        )

        ctx.font='600 9px JetBrains Mono,monospace'; ctx.fillStyle='#64748b'

        ctx.fillText(`capital=${isEV?'1% (escape valve)':'10% (normal)'}  ┬╖  min_score=0.02`,dx2+12,dy2+74)

      }

      // Pulse ring at price_return if trade executed

      if(traded){

        const pn=nodes['price_return']

        if(pn){

          const pulse=Math.sin(sT*Math.PI*8)*0.28+0.72

          ctx.save(); ctx.globalAlpha=pulse

          ctx.shadowColor='#16a34a'; ctx.shadowBlur=32

          ctx.beginPath(); ctx.arc(pn.x,pn.y,24,0,Math.PI*2)

          ctx.strokeStyle='#16a34a'; ctx.lineWidth=3; ctx.stroke()

          ctx.restore(); ctx.globalAlpha=1

        }

      }

      stageLabel(ctx,W,H,'DECISION ΓÇö Trade or Hold',sT)

    }

    // ├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É

    // STAGE 8: LEARNING ΓÇö ML1 retrain, Layer2 trust update, MetaOptimizer step

    // ├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É├óΓÇó┬É

    else if(vsName==='LEARNING'){

      // Draw the causal graph at 40% opacity as the base

      drawLayerBg(ctx,W,H,nodes)

      drawGraph(ctx,nodes,edges,cs,0.18,0.35)

      const pn=nodes['price_return']

      const trust=propsRef.current.agentState?.trust

      const scores=(trust?.scores||[]).slice(0,8)

      const optReg=propsRef.current.optimizer?.regimes?.[0]

      const recentScores=optReg?.history_scores||[]

      const vec2=propsRef.current.agentState?.active_do?.algo_health_vector||[]

      const lastScore2=recentScores[recentScores.length-1]

      const improving2=lastScore2!=null&&recentScores.length>1&&lastScore2>recentScores[recentScores.length-2]

      // Header

      ctx.font='700 12px JetBrains Mono,monospace'; ctx.fillStyle='#7c3aed'; ctx.textAlign='center'

      ctx.fillText('LEARNING  ┬╖  Outcome ΓåÆ Feedback Loop',W/2,22)

      ctx.font='600 8px JetBrains Mono,monospace'; ctx.fillStyle='#94a3b8'

      ctx.fillText('price_return resolves ΓåÆ ML1 retrains + edge trust updates + MetaOpt hill-climbs',W/2,36)

      // ΓöÇΓöÇ Animated feedback arcs: from price_return backwards to learning targets ΓöÇΓöÇ

      // Targets: algo_health nodes (ML1), any edge source nodes (L2 trust), random node (MetaOpt)

      if(pn){

        const feedbackTargets=[

          {name:'algo_health_p_normal',  col:'#4f46e5', label:'ML1 retrain'},

          {name:'regime_volatile',       col:'#7c3aed', label:'L2 trust'},

          {name:'btc_return',            col:'#0891b2', label:'MetaOpt params'},

        ]

        feedbackTargets.forEach((ft,fi)=>{

          const tn=nodes[ft.name]; if(!tn) return

          const col=ft.col

          // Dashed curved arc from price_return back to target

          const t2=sT+fi*0.22

          const mx=(pn.x+tn.x)/2-38 // offset control point

          const my=(pn.y+tn.y)/2-40

          // Draw dark dashed arc (feedback loop)

          ctx.save()

          ctx.setLineDash([5,6])

          ctx.strokeStyle=col

          ctx.lineWidth=1.8

          ctx.globalAlpha=0.55+Math.sin(sT*Math.PI*3+fi*1.8)*0.20

          ctx.shadowColor=col; ctx.shadowBlur=6

          ctx.beginPath(); ctx.moveTo(pn.x,pn.y)

          ctx.quadraticCurveTo(mx,my,tn.x,tn.y)

          ctx.stroke()

          ctx.setLineDash([]); ctx.shadowBlur=0; ctx.restore(); ctx.globalAlpha=1

          // Feedback particles travelling FROM price_return TO target

          for(let pi=0;pi<4;pi++){

            const frac=((t2*0.55+pi*0.22)%1)

            // Quadratic bezier interpolation

            const bx=(1-frac)*(1-frac)*pn.x+2*(1-frac)*frac*mx+frac*frac*tn.x

            const by=(1-frac)*(1-frac)*pn.y+2*(1-frac)*frac*my+frac*frac*tn.y

            const a=Math.sin(frac*Math.PI)*0.90

            ctx.save(); ctx.globalAlpha=a

            ctx.shadowColor=col; ctx.shadowBlur=10

            ctx.beginPath(); ctx.arc(bx,by,3.5,0,Math.PI*2)

            ctx.fillStyle=col; ctx.fill()

            ctx.shadowBlur=0; ctx.restore(); ctx.globalAlpha=1

          }

          // Highlight target node ΓÇö glow halo + bright fill + name
          if(tn){
            const nodeR = tn.name==='price_return' ? 20 : 14
            const pulse2 = 0.7 + Math.sin(sT*Math.PI*4 + fi*1.4)*0.3
            ctx.save()
            // Outer halo
            ctx.globalAlpha = pulse2 * 0.22
            ctx.shadowColor = col; ctx.shadowBlur = 28
            ctx.beginPath(); ctx.arc(tn.x, tn.y, nodeR+10, 0, Math.PI*2)
            ctx.fillStyle = col; ctx.fill()
            ctx.shadowBlur = 0
            // Node circle
            ctx.globalAlpha = pulse2 * 0.85
            ctx.beginPath(); ctx.arc(tn.x, tn.y, nodeR, 0, Math.PI*2)
            ctx.fillStyle = col+'30'; ctx.fill()
            ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke()
            // Node name below
            ctx.globalAlpha = pulse2
            ctx.font = '700 8px JetBrains Mono,monospace'
            ctx.fillStyle = col; ctx.textAlign = 'center'
            ctx.fillText(abbr(tn.name), tn.x, tn.y + nodeR + 13)
            // Action label (pill above)
            const lx = tn.x - 31
            const ly = tn.y - nodeR - 20
            ctx.globalAlpha = pulse2 * 0.9
            ctx.fillStyle = 'rgba(255,255,255,0.95)'
            ctx.beginPath(); ctx.roundRect(lx, ly, 62, 15, 4); ctx.fill()
            ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.stroke()
            ctx.font = '700 7.5px JetBrains Mono,monospace'; ctx.fillStyle = col; ctx.textAlign = 'center'
            ctx.fillText(ft.label, tn.x, ly + 10)
            ctx.restore(); ctx.globalAlpha = 1
          }

        })

        // Pulse glow on price_return node (the outcome node)

        const pr=nodes['price_return']

        if(pr){

          const pulse=Math.sin(sT*Math.PI*5)*0.3+0.7

          ctx.save()

          ctx.globalAlpha=pulse*0.35

          ctx.shadowColor='#16a34a'; ctx.shadowBlur=28

          ctx.beginPath(); ctx.arc(pr.x,pr.y,22,0,Math.PI*2)

          ctx.fillStyle='#16a34a22'; ctx.fill()

          ctx.shadowBlur=0; ctx.restore(); ctx.globalAlpha=1

          // "OUTCOME" label

          ctx.font='700 8px JetBrains Mono,monospace'; ctx.fillStyle='#16a34a'; ctx.textAlign='center'

          ctx.fillText('OUTCOME',pr.x,pr.y+32)

        }

      }

      // ΓöÇΓöÇ HUD panel: slim 3-row info strip ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

      const hudX=12, hudY=H-90, hudW=Math.min(W-24,580), hudH=72

      ctx.save()

      ctx.fillStyle='rgba(255,255,255,0.93)'

      ctx.shadowColor='rgba(0,0,0,0.08)'; ctx.shadowBlur=14

      ctx.beginPath(); ctx.roundRect(hudX,hudY,hudW,hudH,8); ctx.fill()

      ctx.shadowBlur=0; ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=1; ctx.stroke()

      ctx.restore()

      // HUD divider top bar (colour strip)

      ctx.fillStyle='#7c3aed'; ctx.beginPath(); ctx.roundRect(hudX,hudY,hudW,5,{upperLeft:8,upperRight:8,lowerLeft:0,lowerRight:0}); ctx.fill()

      // Row 1: ML1

      const ml1x=hudX+14, ml1y=hudY+22

      ctx.font='700 8px JetBrains Mono,monospace'; ctx.fillStyle='#4f46e5'; ctx.textAlign='left'

      ctx.fillText('ML1',ml1x,ml1y)

      ctx.font='600 8px JetBrains Mono,monospace'; ctx.fillStyle='#334155'

      ctx.fillText(`N=${(vec2[0]*100||0).toFixed(0)}% S=${(vec2[1]*100||0).toFixed(0)}% D=${(vec2[2]*100||0).toFixed(0)}%`,ml1x+26,ml1y)

      // ML1 mini bar showing N/S/D proportions

      const mlW=100,mlY=ml1y+5,mlX=ml1x+26

      const nW=(vec2[0]||0)*mlW, sW=(vec2[1]||0)*mlW, dW=(vec2[2]||0)*mlW

      ctx.fillStyle='#f1f5f9'; ctx.beginPath(); ctx.roundRect(mlX,mlY,mlW,5,3); ctx.fill()

      ctx.fillStyle='#16a34a'; ctx.beginPath(); ctx.roundRect(mlX,mlY,nW,5,3); ctx.fill()

      ctx.fillStyle='#f59e0b'; ctx.beginPath(); ctx.roundRect(mlX+nW,mlY,sW,5,3); ctx.fill()

      ctx.fillStyle='#ef4444'; ctx.beginPath(); ctx.roundRect(mlX+nW+sW,mlY,dW,5,3); ctx.fill()

      // Row 2: L2 Trust + MetaOpt

      const r2y=hudY+44

      ctx.font='700 8px JetBrains Mono,monospace'; ctx.fillStyle='#7c3aed'; ctx.textAlign='left'

      ctx.fillText('L2',ml1x,r2y)

      const topEdge2=scores[0]

      ctx.font='600 8px JetBrains Mono,monospace'; ctx.fillStyle='#334155'

      ctx.fillText(topEdge2?`top: ${topEdge2.source}ΓåÆ${topEdge2.target} trust=${topEdge2.trust?.toFixed(3)||'ΓÇö'}$}`:'no outcomes yet',ml1x+20,r2y)

      // MetaOpt on same row, right section

      const moX=hudX+hudW/2+20

      ctx.font='700 8px JetBrains Mono,monospace'; ctx.fillStyle='#0891b2'; ctx.textAlign='left'

      ctx.fillText('META',moX,r2y)

      ctx.font='600 8px JetBrains Mono,monospace'; ctx.fillStyle='#334155'

      ctx.fillText(lastScore2!=null?`score ${lastScore2.toFixed(4)} ${improving2?'Γåæ':'ΓåÆ'}`:'exploringΓÇª',moX+32,r2y)

      // Row 3: sparkline if scores exist

      if(recentScores.length>2){

        const slX=moX+110, slY=hudY+14, slW=hudW-(slX-hudX)-14, slH=50

        const mn2=Math.min(...recentScores),mx22=Math.max(...recentScores),rng2=mx22-mn2||0.01

        ctx.strokeStyle='#0891b2'; ctx.lineWidth=1.5; ctx.beginPath()

        recentScores.forEach((s,i)=>{

          const rx=slX+i*(slW)/(recentScores.length-1)

          const ry=slY+slH-((s-mn2)/rng2)*slH

          i===0?ctx.moveTo(rx,ry):ctx.lineTo(rx,ry)

        }); ctx.stroke()

        const li2=recentScores.length-1

        const lx3=slX+li2*slW/Math.max(1,li2)

        const ly3=slY+slH-((recentScores[li2]-mn2)/rng2)*slH

        const dotP=Math.sin(sT*Math.PI*6)*0.25+0.75

        ctx.globalAlpha=dotP

        ctx.shadowColor='#0891b2'; ctx.shadowBlur=8

        ctx.beginPath(); ctx.arc(lx3,ly3,3.5,0,Math.PI*2); ctx.fillStyle='#0891b2'; ctx.fill()

        ctx.shadowBlur=0; ctx.restore(); ctx.globalAlpha=1

      }

      stageLabel(ctx,W,H,'LEARNING ΓÇö Feedback & Weight Updates',sT)

    }

  },[])

  useEffect(()=>{

    let rafId=0

    const loop=(ts)=>{ drawFn.current(ts); rafId=requestAnimationFrame(loop) }

    rafId=requestAnimationFrame(loop)

    return()=>cancelAnimationFrame(rafId)

  },[])

  useEffect(()=>{

    const w=cvs.current?.parentElement?.clientWidth

    const h=cvs.current?.parentElement?.clientHeight

    layout(graph||MOCK_GRAPH,w,h)

  },[graph,layout])

  const handleCanvasClick=useCallback((e)=>{

    const canvas=cvs.current; if(!canvas) return

    const rect=canvas.getBoundingClientRect()

    const x=(e.clientX-rect.left)*(canvas.width/rect.width)

    const W2=canvas.width

    const vsName=VSTAGES[vsIdxRef.current]

    if(!['PCMCI','HYPOTHESIS','OPTIMIZER','TWIN','DECISION'].includes(vsName)) return

    let matched=null

    for(const l of LO){ if(Math.abs(x-W2*LX[l])<W2*0.12){ matched=l; break } }

    clickedLayer.current=clickedLayer.current===matched?null:matched

  },[])

  return(

    <div style={{position:'relative',width:'100%',height:'100%',background:'#fafafa',fontFamily:"'JetBrains Mono',monospace"}}>

      <canvas ref={cvs} onClick={handleCanvasClick}

        style={{position:'absolute',inset:0,width:'100%',height:'100%',cursor:'crosshair'}}/>

      <div style={{position:'absolute',bottom:8,left:'50%',transform:'translateX(-50%)',

        fontSize:'9px',color:'#94a3b8',letterSpacing:'0.1em',textTransform:'uppercase',pointerEvents:'none'}}>

        {vsLabel}

      </div>

    </div>

  )
}
