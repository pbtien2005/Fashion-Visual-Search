import { useEffect, useMemo, useRef, useState } from 'react'

import { DetectedItem } from '../components/DetectedItem'
import { ChevronIcon, CloseIcon, ImageIcon, SearchIcon, UploadIcon } from '../components/Icons'
import { searchUpload } from '../services/api'

const MODES = [
  { value: 'hybrid', label: 'Hybrid (style + color)', hint: 'Combines style and color. Most accurate.' },
  { value: 'embedding_only', label: 'Style only', hint: 'FashionCLIP embedding only.' },
  { value: 'histogram_only', label: 'Color only', hint: 'HSV histogram only.' },
]

const STEPS = ['Detect garments', 'Extract style & color', 'Rank similar products']

export function SearchView({ onSaveHistory }) {
  const [file, setFile] = useState(null)
  const [mode, setMode] = useState('hybrid')
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)
  const resultsRef = useRef(null)

  const detectedItems = result?.detected_items || []
  const canSubmit = useMemo(() => Boolean(file) && !loading, [file, loading])

  useEffect(() => {
    if (!loading && result && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [loading, result])

  function selectFile(nextFile) {
    setResult(null)
    setError('')
    setFile(nextFile || null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : null)
  }

  function handleFileChange(event) {
    selectFile(event.target.files?.[0])
  }

  function handleDrop(event) {
    event.preventDefault()
    setDragging(false)
    const dropped = event.dataTransfer.files?.[0]
    if (!dropped) return
    if (!dropped.type.startsWith('image/')) {
      setError('Please drop an image file (JPG, PNG, WebP).')
      return
    }
    selectFile(dropped)
  }

  function clearFile(event) {
    event.preventDefault()
    event.stopPropagation()
    selectFile(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  async function handleSubmit(event) {
    event.preventDefault()
    if (!file) return

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const data = await searchUpload(file, mode)
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
      <section className="hero">
        <p className="eyebrow">Visual search for fashion</p>
        <h1>
          Found a look you love?
          <br />
          <em>Find the pieces.</em>
        </h1>
        <p className="hero-sub">
          Upload any outfit photo. We detect each garment and match it to similar products in the catalog.
        </p>
      </section>

      <form className="studio" onSubmit={handleSubmit}>
        <label
          className={['dropzone', dragging ? 'is-dragging' : '', previewUrl ? 'has-image' : '', loading ? 'is-scanning' : '']
            .filter(Boolean)
            .join(' ')}
          onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <input ref={inputRef} type="file" accept="image/*" onChange={handleFileChange} />
          {previewUrl ? (
            <>
              <img src={previewUrl} alt="Uploaded outfit preview" />
              {loading && <span className="scan-line" aria-hidden="true" />}
              <button type="button" className="dropzone-clear" onClick={clearFile} aria-label="Remove image" disabled={loading}>
                <CloseIcon size={16} />
              </button>
            </>
          ) : (
            <div className="dropzone-empty">
              <span className="dropzone-icon"><UploadIcon size={26} /></span>
              <strong>Drop an outfit photo here</strong>
              <span>or <u>browse</u> from your device</span>
              <small>JPG, PNG or WebP</small>
            </div>
          )}
        </label>

        <div className="controls">
          <div className="control-group">
            <div className="control-label"><span className="step-no">1</span> Image</div>
            <div className="file-line">
              <ImageIcon size={18} />
              <span>{file ? file.name : 'No image selected'}</span>
            </div>
          </div>

          <fieldset className="control-group">
            <legend className="control-label"><span className="step-no">2</span> Matching method</legend>
            <div className="select-wrap">
              <select value={mode} onChange={(event) => setMode(event.target.value)} aria-label="Matching method">
                {MODES.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
              <ChevronIcon size={18} />
            </div>
            <p className="select-hint">{MODES.find((option) => option.value === mode)?.hint}</p>
          </fieldset>

          <button type="submit" className="btn btn-primary btn-lg" disabled={!canSubmit}>
            {loading ? <span className="spinner" aria-hidden="true" /> : <SearchIcon size={18} />}
            {loading ? 'Searching…' : 'Find similar products'}
          </button>

          {error && <div className="notice notice-error" role="alert">{error}</div>}

          <ol className="pipeline">
            {STEPS.map((step, index) => (
              <li key={step}><span>{index + 1}</span>{step}</li>
            ))}
          </ol>
        </div>
      </form>

      <div ref={resultsRef} className="results-anchor">
        {loading && <ResultsSkeleton />}

        {!loading && result && detectedItems.length === 0 && (
          <div className="empty-card">
            <span className="empty-icon"><ImageIcon size={24} /></span>
            <h2>No garments detected</h2>
            <p>{result.message || 'No clothing items were detected in this image. Try a clearer, full-outfit photo.'}</p>
          </div>
        )}

        {!loading && detectedItems.length > 0 && (
          <section className="results">
            <div className="section-head">
              <div>
                <p className="eyebrow">Results</p>
                <h2>{detectedItems.length} {detectedItems.length === 1 ? 'garment' : 'garments'} detected</h2>
              </div>
            </div>
            <div className="results-stack">
              {detectedItems.map((item, index) => (
                <DetectedItem key={`${item.class_name}-${index}`} item={item} index={index} />
              ))}
            </div>
          </section>
        )}
      </div>
    </>
  )
}

function ResultsSkeleton() {
  return (
    <section className="results" aria-busy="true" aria-label="Searching">
      <div className="section-head">
        <div>
          <p className="eyebrow">Analyzing</p>
          <h2>Detecting garments and searching products…</h2>
        </div>
      </div>
      <div className="detected-item skeleton-item">
        <div className="skeleton crop-skeleton" />
        <div className="skeleton-body">
          <div className="skeleton line-lg" />
          <div className="skeleton-grid">
            {[0, 1, 2, 3].map((n) => (
              <div className="skeleton card-skeleton" key={n} />
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
