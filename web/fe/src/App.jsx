import { useState } from 'react'

import { ClockIcon, GridIcon, HangerIcon, SearchIcon } from './components/Icons'
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

  const tabs = [
    { id: 'search', label: 'Search', icon: SearchIcon },
    { id: 'products', label: 'Catalog', icon: GridIcon },
    { id: 'history', label: 'History', icon: ClockIcon, count: history.length },
  ]

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="/" onClick={(event) => { event.preventDefault(); setActiveTab('search') }}>
            <span className="brand-mark"><HangerIcon size={18} /></span>
            <span className="brand-name">Fashion<em>Visual</em>Search</span>
          </a>

          <nav className="tabs" aria-label="Main views">
            {tabs.map(({ id, label, icon: TabIcon, count }) => (
              <button
                key={id}
                type="button"
                className={activeTab === id ? 'tab active' : 'tab'}
                aria-current={activeTab === id ? 'page' : undefined}
                onClick={() => setActiveTab(id)}
              >
                <TabIcon size={17} />
                <span>{label}</span>
                {count > 0 && <span className="tab-count">{count}</span>}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="page">
        {activeTab === 'search' && <SearchView onSaveHistory={saveHistory} />}
        {activeTab === 'products' && <ProductsView />}
        {activeTab === 'history' && <HistoryView history={history} onClearHistory={clearHistory} />}
      </main>

      <footer className="footer">
        <span>Fashion Visual Search</span>
        <span>YOLOv8 detection · FashionCLIP + HSV · pgvector ranking</span>
      </footer>
    </div>
  )
}

export default App
