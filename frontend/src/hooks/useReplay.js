import { useState, useEffect, useRef } from 'react'

export function useReplay() {
  const [replay, setReplay] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeStage, setActiveStage] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(3000) // ms per stage
  const timerRef = useRef(null)

  const STAGES = ['STREAMS','BARS','PCMCI','HYPOTHESIS','OPTIMIZER','TWIN','DECISION','LEARNING']

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const r = await fetch('/api/replay/last').then(r => r.json())
      if (r.error || !r.replay) {
        setError(r.reason || r.error || 'No completed experiment found yet.')
      } else {
        setReplay(r.replay)
        setActiveStage(0)
        setPlaying(false)
      }
    } catch {
      setError('API unreachable')
    } finally {
      setLoading(false)
    }
  }

  // Auto-advance stages when playing
  useEffect(() => {
    if (!playing || !replay) return
    timerRef.current = setInterval(() => {
      setActiveStage(s => {
        if (s >= STAGES.length - 1) { setPlaying(false); return s }
        return s + 1
      })
    }, speed)
    return () => clearInterval(timerRef.current)
  }, [playing, replay, speed])

  const play  = () => { setActiveStage(0); setPlaying(true) }
  const pause = () => setPlaying(false)
  const step  = () => setActiveStage(s => Math.min(s + 1, STAGES.length - 1))
  const reset = () => { setPlaying(false); setActiveStage(0) }

  return { replay, loading, error, activeStage, playing, speed, STAGES,
           load, play, pause, step, reset, setSpeed, setActiveStage }
}
