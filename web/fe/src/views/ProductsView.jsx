import { useEffect, useState } from 'react'

import { ArrowIcon, GridIcon, SearchIcon } from '../components/Icons'
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
    <section className="view">
      <div className="section-head">
        <div>
          <p className="eyebrow">Browse</p>
          <h1 className="view-title">Product catalog</h1>
        </div>
        <span className="pill pill-lg">{total.toLocaleString()} products</span>
      </div>

      <form className="filter-bar" onSubmit={handleSubmit}>
        <label className="field field-grow">
          <SearchIcon size={17} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search title or brand" />
        </label>
        <label className="field">
          <GridIcon size={17} />
          <input value={category} onChange={(event) => setCategory(event.target.value)} placeholder="Category" />
        </label>
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? 'Loading…' : 'Apply'}
        </button>
      </form>

      {error && <div className="notice notice-error" role="alert">{error}</div>}

      {products.length ? (
        <div className={loading ? 'product-grid catalog-grid is-loading' : 'product-grid catalog-grid'}>
          {products.map((product) => (
            <ProductCard key={product.id} product={product} />
          ))}
        </div>
      ) : (
        !loading && (
          <div className="empty-card">
            <span className="empty-icon"><GridIcon size={24} /></span>
            <h2>No products found</h2>
            <p>Try a different keyword or clear the category filter.</p>
          </div>
        )
      )}

      {total > PAGE_SIZE && (
        <div className="pager">
          <button
            type="button"
            className="btn btn-outline"
            disabled={loading || offset === 0}
            onClick={() => loadProducts(Math.max(0, offset - PAGE_SIZE))}
          >
            <ArrowIcon size={16} className="flip" />
            Previous
          </button>
          <span>{offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total.toLocaleString()}</span>
          <button
            type="button"
            className="btn btn-outline"
            disabled={loading || offset + PAGE_SIZE >= total}
            onClick={() => loadProducts(offset + PAGE_SIZE)}
          >
            Next
            <ArrowIcon size={16} />
          </button>
        </div>
      )}
    </section>
  )
}
