import { API_URL } from '../config'

async function parseJson(response) {
  return response.json().catch(() => ({}))
}

export async function searchUpload(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_URL}/api/search/upload`, {
    method: 'POST',
    body: formData,
  })
  const data = await parseJson(response)
  if (!response.ok) {
    throw new Error(data.detail || data.message || `Request failed with ${response.status}`)
  }
  return data
}

export async function fetchProducts({ query = '', category = '', limit = 24, offset = 0 }) {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  })
  if (query.trim()) params.set('q', query.trim())
  if (category.trim()) params.set('category', category.trim())

  const response = await fetch(`${API_URL}/api/products?${params.toString()}`)
  const data = await parseJson(response)
  if (!response.ok) {
    throw new Error(data.detail || `Request failed with ${response.status}`)
  }
  return data
}
