import { useEffect, useState } from 'react'

import { ProductCard } from '../components/ProductCard'
import { fetchProducts } from '../services/api'

const PAGE_SIZE = 24

export function ProductsView() {
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('')
  const [products, setProducts] = useState([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function loadProducts(nextOffset = 0) {
    setLoading(true)
    setError('')

    try {
      const data = await fetchProducts({
        query,
        category,
        limit: PAGE_SIZE,
        offset: nextOffset,
      })
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
        <button type="button" disabled={loading || offset === 0} onClick={() => loadProducts(Math.max(0, offset - PAGE_SIZE))}>
          Previous
        </button>
        <span>{offset + 1}-{Math.min(offset + PAGE_SIZE, total)} of {total}</span>
        <button type="button" disabled={loading || offset + PAGE_SIZE >= total} onClick={() => loadProducts(offset + PAGE_SIZE)}>
          Next
        </button>
      </div>
    </section>
  )
}
