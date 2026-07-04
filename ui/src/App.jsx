import React from 'react'
import { AgentCockpit } from './components/cockpit/AgentCockpit'
import './index.css'

function App() {
  return (
    <div style={{ width: '100vw', height: '100vh', background: 'var(--bg-base)' }}>
      <AgentCockpit />
    </div>
  )
}

export default App
