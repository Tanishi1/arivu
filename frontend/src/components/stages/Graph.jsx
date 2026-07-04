import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'

// ── Layer assignment ──────────────────────────────────────
// Fallback: anything matching these patterns → layer
const LAYER_PATTERNS = {
  macro:          [/btc/,/eth/,/session/,/macro/],
  market:         [/volatil/,/spread/,/volume(?!_imb)/,/rsi/,/trend/,/ema_spread/,/bollinger/,/atr/,/sma/,/ema(?!_v)/,/stoch/],
  microstructure: [/order_book/,/depth/,/trade_press/,/liquidity/,/volume_imb/,/ema_vol/,/vwap/,/price_mom/,/imbalance/],
  health:         [/health/,/integrity/,/regime/,/trading/,/ml1/,/ml2/],
  target:         [/price_return/],
}
const LAYER_ORDER = ['macro','market','microstructure','health','target']
const LAYER_X_PCT = { macro: 0.06, market: 0.28, microstructure: 0.55, health: 0.78, target: 0.95 }
const LAYER_COLORS = {
  macro: '#6d28d9', market: '#4f46e5', microstructure: '#2563eb', health: '#475569', target: '#16a34a',
}

function assignLayer(name) {
  for (const [layer, patterns] of Object.entries(LAYER_PATTERNS))
    if (patterns.some(p => p.test(name))) return layer
  return 'market' // fallback
}

const FILTERS = ['All','Macro','Market','Micro','Health','Target','Validated','Hyp Only']

export default function Graph({ graph, hypotheses = [], selectedChain }) {
  const svgRef = useRef(null)
  const [filter, setFilter] = useState('All')

  useEffect(() => {
    if (!graph || !svgRef.current) return
    drawGraph(svgRef.current, graph, hypotheses, selectedChain, filter)
  }, [graph, hypotheses, selectedChain, filter])

  if (!graph) return (
    <div className="stage-panel fade-in">
      <div className="stage-hdr">
        <span className="stage-tag t-graph">GRAPH</span>
        <span className="stage-title">Causal Graph</span>
      </div>
      <div className="empty"><div className="empty-icon">◇</div><p>Waiting for first PCMCI run — needs 40+ bars</p></div>
    </div>
  )

  return (
    <div className="stage-panel fade-in">
      <div className="stage-hdr">
        <span className="stage-tag t-graph">GRAPH</span>
        <span className="stage-title">Causal Graph</span>
        <span className="stage-meta">v{graph.version_id?.slice(0,8)} · {graph.edge_count} edges · {graph.algorithm}</span>
      </div>

      <div className="graph-filters">
        {FILTERS.map(f => (
          <button key={f} className={`gfbtn ${filter===f?'on':''}`} onClick={() => setFilter(f)}>{f}</button>
        ))}
      </div>

      <div className="graph-canvas">
        <svg ref={svgRef} style={{ width:'100%', display:'block' }} />
      </div>

      <div className="legend" style={{ marginTop: 10 }}>
        {Object.entries(LAYER_COLORS).map(([l, c]) => (
          <span className="legend-item" key={l}>
            <span className="legend-dot" style={{ background: c }} />
            {l}
          </span>
        ))}
        <span className="legend-item">
          <span className="legend-line" style={{ background:'#2563eb', height:2 }} />
          validated
        </span>
        <span className="legend-item">
          <span className="legend-line" style={{ background:'#c2410c', borderTop:'2px dashed' }} />
          escape valve
        </span>
      </div>
    </div>
  )
}

function drawGraph(el, graph, hypotheses, selectedChain, filter) {
  const d3svg = d3.select(el)
  d3svg.selectAll('*').remove()

  // ── Build node set ──
  const nodeSet = new Set()
  graph.edges?.forEach(e => { nodeSet.add(e.source); nodeSet.add(e.target) })
  graph.variables?.forEach(v => nodeSet.add(v))
  if (nodeSet.size === 0) return

  // ── Assign layers ──
  const layerNodes = {}
  LAYER_ORDER.forEach(l => layerNodes[l] = [])
  nodeSet.forEach(n => layerNodes[assignLayer(n)].push(n))

  // ── Calculate dimensions ──
  const maxNodesInLayer = Math.max(...Object.values(layerNodes).map(a => a.length))
  const NODE_SPACING = 28
  const MIN_H = 220
  const neededH = Math.max(MIN_H, maxNodesInLayer * NODE_SPACING + 40)
  const W = el.clientWidth || 700
  const H = neededH
  d3svg.attr('height', H)

  const margin = { top: 20, bottom: 20, left: 10, right: 60 }
  const usableW = W - margin.left - margin.right
  const usableH = H - margin.top - margin.bottom

  // ── Compute node positions ──
  const nodes = {}
  LAYER_ORDER.forEach(layer => {
    const ns = layerNodes[layer]
    if (!ns.length) return
    const x = margin.left + usableW * LAYER_X_PCT[layer]
    const step = usableH / (ns.length + 1)
    ns.forEach((n, i) => {
      nodes[n] = { x, y: margin.top + step * (i + 1), layer, name: n }
    })
  })

  // ── Build hypothesis edge set ──
  const hypEdges = new Set()
  ;(hypotheses || []).forEach(h => {
    const parts = (h.chain || '').split(/\s*→\s*|\s*->\s*/)
    parts.forEach((p, i) => {
      if (parts[i + 1]) hypEdges.add(`${p.trim()}|${parts[i+1].trim()}`)
    })
  })
  const selEdges = new Set()
  if (selectedChain) {
    const parts = selectedChain.split(/\s*→\s*|\s*->\s*|\s*\(lag=\d+\)\s*→?\s*/)
      .map(p => p.replace(/\(.*\)/,'').trim()).filter(Boolean)
    parts.forEach((p, i) => { if (parts[i+1]) selEdges.add(`${p}|${parts[i+1]}`) })
  }

  // ── Filter edges ──
  let edges = graph.edges || []
  if (filter === 'Validated')  edges = edges.filter(e => e.validated)
  if (filter === 'Macro')      edges = edges.filter(e => assignLayer(e.source) === 'macro')
  if (filter === 'Market')     edges = edges.filter(e => assignLayer(e.source) === 'market')
  if (filter === 'Micro')      edges = edges.filter(e => assignLayer(e.source) === 'microstructure')
  if (filter === 'Health')     edges = edges.filter(e => assignLayer(e.source) === 'health')
  if (filter === 'Target')     edges = edges.filter(e => e.target === 'price_return')
  if (filter === 'Hyp Only')   edges = edges.filter(e => hypEdges.has(`${e.source}|${e.target}`) || selEdges.has(`${e.source}|${e.target}`))

  const g = d3svg.append('g')

  // ── Draw edges ──
  edges.forEach(e => {
    const s = nodes[e.source], t = nodes[e.target]
    if (!s || !t) return

    const isSel  = selEdges.has(`${e.source}|${e.target}`)
    const isHyp  = hypEdges.has(`${e.source}|${e.target}`)
    const isVal  = e.validated

    const stroke  = isSel ? '#2563eb' : isVal ? '#4f46e5' : isHyp ? '#6d9feb' : '#d4cfc8'
    const opacity = isSel ? 0.95 : isVal ? 0.65 : isHyp ? 0.50 : 0.25
    const sw      = isSel ? 2.5 : isVal ? 1.8 : 1

    const mx = (s.x + t.x) / 2
    const path = g.append('path')
      .attr('d', `M${s.x},${s.y} C${mx},${s.y} ${mx},${t.y} ${t.x},${t.y}`)
      .attr('stroke', stroke).attr('stroke-width', sw)
      .attr('fill', 'none').attr('opacity', opacity)
    if (e.is_escape_valve) path.attr('stroke-dasharray', '5,3')

    // Animated dot along selected edges
    if (isSel) {
      const totalLen = 200
      g.append('circle')
        .attr('r', 3).attr('fill', '#2563eb').attr('opacity', 0.8)
        .append('animateMotion')
        .attr('dur', '2s').attr('repeatCount', 'indefinite')
        .append('mpath').attr('href', null) // fallback — simple transform animation
      // Simpler: animate opacity pulsing on the path itself
      path.append('animate')
        .attr('attributeName', 'opacity')
        .attr('values', '0.95;0.3;0.95')
        .attr('dur', '1.8s')
        .attr('repeatCount', 'indefinite')
    }
  })

  // ── Draw nodes ──
  const nodeG = g.append('g')
  Object.values(nodes).forEach(n => {
    const color = LAYER_COLORS[n.layer]
    const isTarget = n.name === 'price_return'
    const r = isTarget ? 13 : 9
    const inHyp = hypotheses?.some(h => (h.chain || '').includes(n.name))
    const inSel = selectedChain?.includes(n.name)

    const ng = nodeG.append('g')
      .attr('transform', `translate(${n.x},${n.y})`)
      .style('cursor', 'default')

    ng.append('circle')
      .attr('r', r)
      .attr('fill', inSel ? color : (inHyp ? color : color))
      .attr('fill-opacity', inSel ? 0.25 : inHyp ? 0.18 : 0.10)
      .attr('stroke', color)
      .attr('stroke-width', inSel ? 2.5 : inHyp ? 2 : 1.2)

    // Pulse on selected node
    if (inSel) {
      ng.append('circle')
        .attr('r', r + 5).attr('fill', 'none').attr('stroke', color)
        .attr('stroke-width', 1).attr('opacity', 0.3)
        .append('animate')
          .attr('attributeName', 'r').attr('values', `${r};${r+10}`)
          .attr('dur', '1.5s').attr('repeatCount', 'indefinite')
      ng.select('circle:first-child').append('animate')
        .attr('attributeName', 'opacity').attr('values', '1;0')
        .attr('dur', '1.5s').attr('repeatCount', 'indefinite')
    }

    // Label — abbreviate long names
    const label = n.name.length > 14
      ? n.name.replace('_imbalance','_imb').replace('_return','_ret').replace('microstructure','μ').replace('volatility','vol')
      : n.name
    const labelX = n.layer === 'target' ? r + 5 : 0
    const anchor = n.layer === 'target' ? 'start' : 'middle'
    const dy     = n.layer === 'target' ? 4 : r + 11

    ng.append('text')
      .attr('x', labelX).attr('y', dy)
      .attr('text-anchor', anchor)
      .attr('font-size', 9)
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('fill', color)
      .attr('font-weight', inSel ? 700 : 500)
      .text(label)
  })

  // ── Layer dividers ──
  LAYER_ORDER.forEach(layer => {
    if (!layerNodes[layer].length) return
    const x = margin.left + usableW * LAYER_X_PCT[layer]
    g.append('text')
      .attr('x', x).attr('y', margin.top - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', 8).attr('font-family', 'JetBrains Mono, monospace')
      .attr('fill', LAYER_COLORS[layer]).attr('opacity', 0.7)
      .attr('font-weight', 600).attr('letter-spacing', 1)
      .text(layer.toUpperCase().slice(0, 5))
  })
}
