import { describe, expect, it } from 'vitest'
import { readSse } from './sse'

/**
 * Главный тест здесь — тот, который режет поток на ОДИН байт.
 *
 * Локально чанки приходят целыми, поэтому баг с разорванной границей события
 * невозможно воспроизвести на нормальных тестовых данных: он ждёт прода.
 * Значит режем искусственно мелко (раздел 5 чек-листа).
 */

function response(text: string, chunkSize: number): Response {
  const bytes = new TextEncoder().encode(text)
  let offset = 0

  const body = {
    getReader() {
      return {
        read: async () => {
          if (offset >= bytes.length) return { done: true, value: undefined }
          const value = bytes.slice(offset, offset + chunkSize)
          offset += chunkSize
          return { done: false, value }
        },
        cancel: async () => undefined,
      }
    },
  }

  return { body } as unknown as Response
}

const STREAM =
  'event: meta\ndata: {"trace_id":"abc"}\n\n' +
  'event: delta\ndata: {"text":"Кэш "}\n\n' +
  'event: delta\ndata: {"text":"включается "}\n\n' +
  ': heartbeat\n\n' +
  'event: delta\ndata: {"text":"переменной."}\n\n' +
  'event: broken\ndata: {не json}\n\n' +
  'event: answer\ndata: {"status":"answered"}\n\n' +
  'event: done\ndata: {"finish_reason":"stop"}\n\n'

async function collect(text: string, chunkSize: number) {
  const events: Array<{ name: string; data: unknown }> = []
  for await (const event of readSse(response(text, chunkSize))) events.push(event)
  return events
}

describe('readSse', () => {
  it.each([1, 2, 3, 7, 16, 64, 4096])('разбирает поток при размере куска %i', async (size) => {
    const events = await collect(STREAM, size)

    expect(events.map((event) => event.name)).toEqual([
      'meta',
      'delta',
      'delta',
      'delta',
      'answer',
      'done',
    ])

    const text = events
      .filter((event) => event.name === 'delta')
      .map((event) => (event.data as { text: string }).text)
      .join('')
    expect(text).toBe('Кэш включается переменной.')
  })

  it('пропускает битое событие, не роняя поток', async () => {
    const events = await collect(STREAM, 5)
    expect(events.some((event) => event.name === 'broken')).toBe(false)
    expect(events.at(-1)?.name).toBe('done')
  })

  it('сохраняет полученное при обрыве посреди события', async () => {
    const truncated =
      'event: delta\ndata: {"text":"начало"}\n\nevent: delta\ndata: {"text":"обры'
    const events = await collect(truncated, 4)

    expect(events).toHaveLength(1)
    expect((events[0].data as { text: string }).text).toBe('начало')
  })

  it('не портит многобайтовые символы на границе кусков', async () => {
    const stream = 'event: delta\ndata: {"text":"привет, мир"}\n\n'
    for (const size of [1, 2, 3]) {
      const events = await collect(stream, size)
      expect((events[0].data as { text: string }).text).toBe('привет, мир')
    }
  })
})
