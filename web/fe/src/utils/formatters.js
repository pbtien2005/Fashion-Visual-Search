import { API_URL } from '../config'

export function formatPercent(value) {
  if (typeof value !== 'number') return null
  return `${(value * 100).toFixed(1)}%`
}

export function formatRgb(rgb) {
  if (!Array.isArray(rgb) || rgb.length !== 3) return null
  return `rgb(${rgb.map((value) => Number(value) || 0).join(', ')})`
}

export function formatColorLabel(name, rgb) {
  const rgbText = Array.isArray(rgb) ? rgb.join(', ') : null
  if (name && rgbText) return `${name} (${rgbText})`
  return name || rgbText
}

export function formatVariantScores(scores) {
  if (!scores || typeof scores !== 'object') return null
  return Object.entries(scores)
    .map(([name, score]) => `${name}: ${formatPercent(score) || score}`)
    .join(', ')
}

export function isPublicImageUrl(url) {
  return typeof url === 'string' && (url.startsWith('http://') || url.startsWith('https://'))
}

export function resolveApiUrl(pathOrUrl) {
  if (!pathOrUrl) return null
  if (isPublicImageUrl(pathOrUrl)) return pathOrUrl
  if (pathOrUrl.startsWith('/')) return `${API_URL}${pathOrUrl}`
  return null
}
