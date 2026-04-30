import { useState } from 'react'

import { clearStoredHistory, readHistory, writeHistory } from './services/historyStorage'
import { HistoryView } from './views/HistoryView'
import { ProductsView } from './views/ProductsView'
import { SearchView } from './views/SearchView'

function App() {
  const [activeTab, setActiveTab] = useState('search')
  const [history, setHistory] = useState(() => readHistory())

  function saveHistory(fileName, data) {
    const entry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      created_at: new Date().toISOString(),
      file_name: fileName,
      detected_items: data.detected_items || [],
      message: data.message || null,
    }
    setHistory((current) => {
      const nextHistory = [entry, ...current].slice(0, 20)
      writeHistory(nextHistory)
      return nextHistory
    })
  }

  function clearHistory() {
    setHistory([])
    clearStoredHistory()
  }

  return (
    <main className="app-shell">
      <nav className="tabs" aria-label="Main views">
        <button type="button" className={activeTab === 'search' ? 'active' : ''} onClick={() => setActiveTab('search')}>
          Search
        </button>
        <button type="button" className={activeTab === 'products' ? 'active' : ''} onClick={() => setActiveTab('products')}>
          Products
        </button>
        <button type="button" className={activeTab === 'history' ? 'active' : ''} onClick={() => setActiveTab('history')}>
          History
        </button>
      </nav>

      {activeTab === 'search' && <SearchView onSaveHistory={saveHistory} />}
      {activeTab === 'products' && <ProductsView />}
      {activeTab === 'history' && <HistoryView history={history} onClearHistory={clearHistory} />}
    </main>
  )
}

export default App
