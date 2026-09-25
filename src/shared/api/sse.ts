/**
 * Разбор потока server-sent events через буфер.
 *
 * Это то место, где ломается почти каждая первая реализация стриминга.
 * Границы сетевых кусков не совпадают с границами событий: одно событие
 * спокойно приезжает разрезанным пополам, а в один кусок попадают три события
 * подряд. Локально этого не видно — на быстром соединении чанк почти всегда
 * приходит целым, — поэтому разбор «каждый кусок отдельно» работает на машине
 * разработчика и разваливается в проде.
 *
 * Отсюда два правила, оба реализованы ниже:
 *   1) накапливать в буфере и вырезать только ЗАКОНЧЕННЫЕ события;
 *   2) битое событие пропускать, а не ронять весь поток: одна аномалия в
 *      середине не должна стоить пользователю всего уже полученного ответа.
 */

export type SseEvent = { name: string; data: unknown }

const EVENT_SEPARATOR = /\r?\n\r?\n/

/** Разбирает один блок «event: … / data: …» в событие. */
function parseBlock(block: string): SseEvent | null {
  let name = 'message'
  const dataLines: string[] = []

  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith(':')) continue // комментарий-heartbeat
    if (line.startsWith('event:')) name = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }

  if (dataLines.length === 0) return null

  try {
    return { name, data: JSON.parse(dataLines.join('\n')) }
  } catch {
    // Битое событие: пропускаем и продолжаем читать поток.
    return null
  }
}

/**
 * Итератор событий из тела ответа fetch.
 *
 * Отмена делается через AbortSignal на самом fetch — не здесь: остановить
 * отрисовку недостаточно, надо порвать соединение, иначе модель продолжает
 * генерировать, а мы продолжаем за это платить.
 */
export async function* readSse(
  response: Response,
  options: { onIdle?: () => void } = {},
): AsyncGenerator<SseEvent> {
  if (!response.body) throw new Error('поток без тела ответа')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break

      // stream: true обязателен — иначе многобайтовый символ, разрезанный
      // между чанками, декодируется в «замену» и текст портится.
      buffer += decoder.decode(value, { stream: true })
      options.onIdle?.()

      for (;;) {
        const match = EVENT_SEPARATOR.exec(buffer)
        if (!match) break
        const block = buffer.slice(0, match.index)
        buffer = buffer.slice(match.index + match[0].length)
        const event = parseBlock(block)
        if (event) yield event
      }
    }

    const tail = buffer.trim()
    if (tail) {
      const event = parseBlock(tail)
      if (event) yield event
    }
  } finally {
    // reader.cancel() освобождает соединение и на стороне сервера роняет
    // генератор — то есть отмена доходит до провайдера, а не только до нас.
    reader.cancel().catch(() => undefined)
  }
}
