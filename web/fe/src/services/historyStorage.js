import { HISTORY_KEY } from '../config'

export function readHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function writeHistory(history) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history))
}

export function clearStoredHistory() {
  localStorage.removeItem(HISTORY_KEY)
}
