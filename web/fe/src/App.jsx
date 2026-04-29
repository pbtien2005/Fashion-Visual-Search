import { useEffect, useMemo, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const HISTORY_KEY = 'fashion_visual_search_history'

function formatPercent(value) {
  if (typeof value !== 'number') return null
  return `${(value * 100).toFixed(1)}%`
}

function formatRgb(rgb) {
  if (!Array.isArray(rgb) || rgb.length !== 3) return null
  return `rgb(${rgb.map((value) => Number(value) || 0).join(', ')})`
}

function formatColorLabel(name, rgb) {
  const rgbText = Array.isArray(rgb) ? rgb.join(', ') : null
  if (name && rgbText) return `${name} (${rgbText})`
  return name || rgbText
}

function isPublicImageUrl(url) {
  return typeof url === 'string' && (url.startsWith('http://') || url.startsWith('https://'))
}

function resolveApiUrl(pathOrUrl) {
  if (!pathOrUrl) return null
  if (isPublicImageUrl(pathOrUrl)) return pathOrUrl
  if (pathOrUrl.startsWith('/')) return `${API_URL}${pathOrUrl}`
  return null
}

function readHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function ProductCard({ product }) {
  const imageUrl = resolveApiUrl(product.usable_image_url)
  const productUrl = isPublicImageUrl(product.product_url) ? product.product_url : null
  const similarity = formatPercent(product.similarity)
  const productColor = formatRgb(product.dominant_color_rgb)
  const productColorLabel = formatColorLabel(product.dominant_color_name, product.dominant_color_rgb)
  const scoreRows = [
    ['Image', formatPercent(product.image_similarity)],
    ['Category', formatPercent(product.category_score)],
    ['Color', formatPercent(product.color_score)],
    ['Final', formatPercent(product.final_score)],
    ['Reason', product.match_reason],
  ].filter(([, value]) => value)

  const content = (
    <>
      <div className="product-media">
        {imageUrl ? (
          <img src={imageUrl} alt={product.title || product.id || 'Product'} />
        ) : (
          <div className="product-placeholder">No public image</div>
        )}
      </div>
      <div className="product-body">
        <h3>{product.title || product.name || product.id || 'Untitled product'}</h3>
        <div className="product-meta">
          {product.brand && <span>{product.brand}</span>}
          {product.price !== null && product.price !== undefined && <span>${product.price}</span>}
          {similarity && <span>{similarity} match</span>}
        </div>
        {product.category && <p className="category">{product.category}</p>}
        {scoreRows.length > 0 && (
          <details className="score-panel">
            <summary>Score</summary>
            <div className="score-grid">
              {scoreRows.map(([label, value]) => (
                <div className="score-row" key={label}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
              {productColorLabel && (
                <div className="score-row">
                  <span>Product color</span>
                  <strong className="color-value">
                    {productColor && <span className="color-swatch" style={{ backgroundColor: productColor }} />}
                    {productColorLabel}
                  </strong>
                </div>
              )}
            </div>
          </details>
        )}
        {!imageUrl && product.image_url && <p className="raw-url">{product.image_url}</p>}
      </div>
    </>
  )

  if (productUrl) {
    return (
      <a className="product-card product-link" href={productUrl} target="_blank" rel="noreferrer">
        {content}
      </a>
    )
  }

  return <article className="product-card">{content}</article>
}

function DetectedItem({ item }) {
  const confidence = formatPercent(item.confidence)
  const cropSrc = `data:image/jpeg;base64,${item.crop_image_base64}`
  const hasProducts = Boolean(item.similar_products?.length)
  const noConfidentMatch = item.is_confident_match === false
  const queryColor = formatRgb(item.query_color_rgb)
  const queryColorLabel = formatColorLabel(item.query_color_name, item.query_color_rgb)
  const emptyMessage = noConfidentMatch
    ? item.message || 'Detected, but no confident product match found'
    : 'No similar products found for this item.'
  const debugRows = [
    ['Class', item.class_name],
    ['Confidence', confidence],
    ['Original bbox', item.original_bbox?.join(', ') || item.bbox?.join(', ')],
    ['Expanded bbox', item.expanded_bbox?.join(', ')],
    ['Padded bbox', item.padded_bbox?.join(', ')],
    ['Output crop', item.crop_width && item.crop_height ? `${item.crop_width} x ${item.crop_height}` : null],
    [
      'Source crop',
      item.source_crop_width && item.source_crop_height ? `${item.source_crop_width} x ${item.source_crop_height}` : null,
    ],
    ['Strategy', item.crop_strategy],
    ['Query color', queryColorLabel],
  ].filter(([, value]) => value)

  return (
    <section className="detected-item">
      <div className="crop-panel">
        <img src={cropSrc} alt={item.class_name_vi || item.class_name} />
      </div>
      <div className="item-results">
        <div className="item-header">
          <div>
            <h2>{item.class_name_vi || item.class_name}</h2>
            <p>{item.class_name}</p>
          </div>
          <div className="badges">
            {confidence && <span>{confidence}</span>}
            <span>[{item.bbox?.join(', ')}]</span>
          </div>
        </div>
        <details className="debug-panel">
          <summary>Detection debug</summary>
          <div className="debug-grid">
            {debugRows.map(([label, value]) => (
              <div className="debug-row" key={label}>
                <span>{label}</span>
                <strong className={label === 'Query color' ? 'color-value' : undefined}>
                  {label === 'Query color' && queryColor && (
                    <span className="color-swatch" style={{ backgroundColor: queryColor }} />
                  )}
                  {value}
                </strong>
              </div>
            ))}
          </div>
        </details>
        {hasProducts ? (
          <div className="product-grid">
            {item.similar_products.map((product, index) => (
              <ProductCard key={`${product.id || 'product'}-${index}`} product={product} />
            ))}
          </div>
        ) : (
          <div className={noConfidentMatch ? 'empty-inline threshold-message' : 'empty-inline'}>{emptyMessage}</div>
        )}
      </div>
    </section>
  )
}

function SearchView({ onSaveHistory }) {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const detectedItems = result?.detected_items || []
  const canSubmit = useMemo(() => Boolean(file) && !loading, [file, loading])

  function handleFileChange(event) {
    const nextFile = event.target.files?.[0]
    setResult(null)
    setError('')
    setFile(nextFile || null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : null)
  }

  async function handleSubmit(event) {
    event.preventDefault()
    if (!file) return

    setLoading(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_URL}/api/search/upload`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || data.message || `Request failed with ${response.status}`)
      }
      setResult(data)
      onSaveHistory(file.name, data)
    } catch (err) {
      setError(err.message || 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <section className="workspace">
        <div className="upload-panel">
          <div>
            <h1>Fashion Visual Search</h1>
            <p>Upload an outfit image to detect garments and find similar products.</p>
          </div>

          <form onSubmit={handleSubmit} className="upload-form">
            <label className="file-drop">
              <input type="file" accept="image/*" onChange={handleFileChange} />
              <span>{file ? file.name : 'Choose outfit image'}</span>
            </label>
            <button type="submit" disabled={!canSubmit}>
              {loading ? 'Searching...' : 'Search'}
            </button>
          </form>

          {error && <div className="error-state">{error}</div>}
        </div>

        <div className="preview-panel">
          {previewUrl ? (
            <img src={previewUrl} alt="Uploaded outfit preview" />
          ) : (
            <div className="preview-empty">Image preview</div>
          )}
        </div>
      </section>

      {loading && <div className="status-panel">Detecting garments and searching products...</div>}

      {!loading && result && detectedItems.length === 0 && (
        <div className="status-panel empty-state">
          {result.message || 'No clothing items were detected in this image.'}
        </div>
      )}

      {!loading && detectedItems.length > 0 && (
        <section className="results-stack">
          {detectedItems.map((item, index) => (
            <DetectedItem key={`${item.class_name}-${index}`} item={item} />
          ))}
        </section>
      )}
    </>
  )
}

function ProductsView() {
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('')
  const [products, setProducts] = useState([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const limit = 24

  async function loadProducts(nextOffset = 0) {
    setLoading(true)
    setError('')
    const params = new URLSearchParams({
      limit: String(limit),
      offset: String(nextOffset),
    })
    if (query.trim()) params.set('q', query.trim())
    if (category.trim()) params.set('category', category.trim())

    try {
      const response = await fetch(`${API_URL}/api/products?${params.toString()}`)
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || `Request failed with ${response.status}`)
      }
      setProducts(data.items || [])
      setTotal(data.total || 0)
      setOffset(nextOffset)
    } catch (err) {
      setError(err.message || 'Product list failed')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadProducts(0)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function handleSubmit(event) {
    event.preventDefault()
    loadProducts(0)
  }

  return (
    <section className="panel-stack">
      <form className="filter-bar" onSubmit={handleSubmit}>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search title or brand" />
        <input value={category} onChange={(event) => setCategory(event.target.value)} placeholder="Category" />
        <button type="submit" disabled={loading}>{loading ? 'Loading...' : 'Filter'}</button>
      </form>

      {error && <div className="error-state">{error}</div>}
      <div className="list-summary">{total} products in database</div>

      {products.length ? (
        <div className="product-grid catalog-grid">
          {products.map((product) => (
            <ProductCard key={product.id} product={product} />
          ))}
        </div>
      ) : (
        !loading && <div className="status-panel empty-state">No products found.</div>
      )}

      <div className="pager">
        <button type="button" disabled={loading || offset === 0} onClick={() => loadProducts(Math.max(0, offset - limit))}>
          Previous
        </button>
        <span>{offset + 1}-{Math.min(offset + limit, total)} of {total}</span>
        <button type="button" disabled={loading || offset + limit >= total} onClick={() => loadProducts(offset + limit)}>
          Next
        </button>
      </div>
    </section>
  )
}

function HistoryView({ history, onClearHistory }) {
  return (
    <section className="panel-stack">
      <div className="section-toolbar">
        <div>
          <h1>Search History</h1>
          <p>Recent searches saved in this browser.</p>
        </div>
        <button type="button" disabled={!history.length} onClick={onClearHistory}>Clear</button>
      </div>

      {history.length ? (
        <div className="history-list">
          {history.map((entry) => (
            <article className="history-card" key={entry.id}>
              <div className="history-card-head">
                <div>
                  <h2>{entry.file_name || 'Uploaded image'}</h2>
                  <p>{new Date(entry.created_at).toLocaleString()}</p>
                </div>
                <span>{entry.detected_items.length} items</span>
              </div>
              {entry.detected_items.length ? (
                <div className="history-crops">
                  {entry.detected_items.map((item, index) => (
                    <div className="history-crop" key={`${entry.id}-${item.class_name}-${index}`}>
                      <img src={`data:image/jpeg;base64,${item.crop_image_base64}`} alt={item.class_name_vi || item.class_name} />
                      <div>
                        <strong>{item.class_name_vi || item.class_name}</strong>
                        <span>{item.similar_products?.length || 0} products</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-inline">No clothing items found.</div>
              )}
            </article>
          ))}
        </div>
      ) : (
        <div className="status-panel empty-state">No search history yet.</div>
      )}
    </section>
  )
}

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
      localStorage.setItem(HISTORY_KEY, JSON.stringify(nextHistory))
      return nextHistory
    })
  }

  function clearHistory() {
    setHistory([])
    localStorage.removeItem(HISTORY_KEY)
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
