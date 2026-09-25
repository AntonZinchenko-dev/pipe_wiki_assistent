/**
 * Клиент бэкенда.
 *
 * Никаких обращений к провайдеру модели напрямую: ключ живёт на сервере, и
 * фронт про него не знает. Всё идёт через свои эндпоинты.
 */

import type {
  AlertReport,
  DocumentDetail,
  DocumentSummary,
  ModelChoice,
  Viewer,
} from './contracts'

const BASE = '/api'

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE}${path}`, { signal })
  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new Error(`${response.status}: ${detail.slice(0, 200) || response.statusText}`)
  }
  return (await response.json()) as T
}

export async function fetchDocuments(signal?: AbortSignal): Promise<DocumentSummary[]> {
  const payload = await getJson<{ documents: DocumentSummary[] }>('/documents', signal)
  return payload.documents
}

export function fetchDocument(docId: string, signal?: AbortSignal): Promise<DocumentDetail> {
  return getJson<DocumentDetail>(`/documents/${encodeURIComponent(docId)}`, signal)
}

export type HealthReport = {
  ok: boolean
  index: { documents: number; chunks: number; meta: Record<string, string> }
  providers: Array<Record<string, unknown>>
  prompt_version: string
  settings: Record<string, unknown>
}

export function fetchHealth(signal?: AbortSignal): Promise<HealthReport> {
  return getJson<HealthReport>('/health', signal)
}

export function fetchViewer(signal?: AbortSignal): Promise<Viewer> {
  return getJson<Viewer>('/me', signal)
}

export function fetchAlerts(signal?: AbortSignal): Promise<AlertReport> {
  return getJson<AlertReport>('/alerts', signal)
}

export async function fetchModels(signal?: AbortSignal): Promise<{
  models: ModelChoice[]
  default: { provider: string; model: string }
  agent: { available: boolean; max_steps: number; decision: string; live_data: boolean }
}> {
  return getJson('/models', signal)
}

/**
 * Запрос к ассистенту. Возвращает сырой Response — разбор потока делает
 * вызывающий код через readSse, потому что ему же принадлежит и отмена.
 *
 * Выбранная модель уходит на сервер парой «провайдер + модель», и сервер
 * сверяет её со своим списком. Фронт тут ничего не гарантирует: имя модели
 * из браузера уходит в чужой API, и проверять его на стороне, которую можно
 * открыть в инструментах разработчика, бессмысленно.
 */
export type ChatTurn = { role: 'user' | 'assistant'; content: string }

export async function postChat(
  question: string,
  choice: { provider: string; model: string } | null,
  agent: boolean,
  signal: AbortSignal,
  history: ChatTurn[] = [],
  summary = '',
): Promise<Response> {
  const response = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      ...(choice ? { provider: choice.provider, model: choice.model } : {}),
      ...(agent ? { agent: true } : {}),
      // Пустую историю не отправляем вовсе: тело первого запроса обязано
      // остаться ровно таким, каким было до появления памяти. Лишнее поле
      // здесь — это уже другой запрос, и сравнивать его с прошлыми
      // прогонами было бы нельзя.
      ...(history.length ? { history } : {}),
      ...(summary ? { summary } : {}),
    }),
    signal,
  })

  if (!response.ok) {
    const retryAfter = response.headers.get('Retry-After')
    const detail = await response.text().catch(() => '')
    // Сообщение конкретное, с временем ожидания: «попробуйте через 40 секунд»
    // полезнее, чем «слишком много запросов».
    const suffix = retryAfter ? ` Попробуйте через ${retryAfter} с.` : ''
    throw new Error(`${detail.slice(0, 200) || response.statusText}${suffix}`)
  }

  return response
}
