import { BrowserRouter, Routes, Route } from 'react-router-dom'
import CockpitPage    from './pages/CockpitPage'
import ComparisonPage from './pages/ComparisonPage'
import LedgerPage     from './pages/LedgerPage'
import ResearchPage   from './pages/ResearchPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"         element={<CockpitPage />} />
        <Route path="/compare"  element={<ComparisonPage />} />
        <Route path="/ledger"   element={<LedgerPage />} />
        <Route path="/research" element={<ResearchPage />} />
        <Route path="*"         element={<CockpitPage />} />
      </Routes>
    </BrowserRouter>
  )
}
