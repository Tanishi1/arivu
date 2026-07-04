import { useState, useEffect, useCallback } from 'react'

const POLL_MS = 10000

export function useSystemState() {
  const [state, setState] = useState(null)
  const [graph, setGraph] = useState(null)
  const [trust, setTrust] = useState(null)
  const [optimizer, setOptimizer] = useState(null)
  const [ledger, setLedger] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)

  const fetchAll = useCallback(async () => {
    try {
      const [s, g, t, o, l] = await Promise.all([
        fetch('/api/state').then(r => r.json()),
        fetch('/api/graph/latest').then(r => r.json()),
        fetch('/api/trust').then(r => r.json()),
        fetch('/api/optimizer').then(r => r.json()),
        fetch('/api/ledger?limit=10').then(r => r.json()),
      ])
      setState(s)
      setGraph(g.graph)
      setTrust(t)
      setOptimizer(o)
      setLedger(l)
      setLastUpdate(new Date())
      setError(null)
    } catch (e) {
      setError('API unreachable — is the API server running?')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchAll()
    const id = setInterval(fetchAll, POLL_MS)
    return () => clearInterval(id)
  }, [fetchAll])

  return { state, graph, trust, optimizer, ledger, loading, error, lastUpdate, refetch: fetchAll }
}

export function useLiveFeed(onMessage) {
  useEffect(() => {
    let ws
    let retry
    const connect = () => {
      ws = new WebSocket(`ws://${location.host}/ws/feed`)
      ws.onmessage = e => {
        try { onMessage(JSON.parse(e.data)) } catch {}
      }
      ws.onclose = () => { retry = setTimeout(connect, 3000) }
      ws.onerror  = () => ws.close()
    }
    connect()
    return () => { ws?.close(); clearTimeout(retry) }
  }, [onMessage])
}
