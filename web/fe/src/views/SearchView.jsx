import { useMemo, useState } from 'react'

import { DetectedItem } from '../components/DetectedItem'
import { searchUpload } from '../services/api'

export function SearchView({ onSaveHistory }) {
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

    try {
      const data = await searchUpload(file)
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
