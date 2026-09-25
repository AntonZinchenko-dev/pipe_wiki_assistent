import { describe, expect, it } from 'vitest'
import { createExchange, type Exchange } from '@/entities/answer'
import type { Dataset } from '@/shared/api/contracts'
import { asOpenTables, asTurns } from './turns'

function done(question: string, answer: string): Exchange {
  return {
    ...createExchange(question),
    phase: 'done',
    answer: {
      status: 'answered',
      answer,
      clarifying_question: '',
      schema_valid: true,
      schema_error: '',
      citations: [],
      citations_failed: 0,
      citations_dropped: 0,
      unknown_refs: [],
      mismatched_refs: [],
    },
  }
}

describe('память диалога', () => {
  it('не отправляет незавершённые обмены', () => {
    // Обрубок хуже отсутствия: модель прочитает его как свои прошлые
    // слова и будет достраивать из половины фразы. Человек при этом
    // видел на экране, что ответ не состоялся, — и не поймёт, откуда
    // ассистент взял продолжение.
    const broken: Exchange = { ...createExchange('оборвали'), phase: 'stopped', streamedText: 'нача' }

    expect(asTurns([broken], 6)).toEqual([])
  })

  it('отправляет пару «вопрос — ответ» на каждый завершённый обмен', () => {
    const turns = asTurns([done('порог?', '80 процентов [1].')], 6)

    expect(turns).toEqual([
      { role: 'user', content: 'порог?' },
      { role: 'assistant', content: '80 процентов [1].' },
    ])
  })

  it('оставляет хвост переписки, а не её начало', () => {
    // «А сколько их?» опирается на предыдущую реплику, а не на ту, что
    // была полчаса назад.
    const turns = asTurns([done('первый', 'раз'), done('второй', 'два')], 1)

    expect(turns.map((turn) => turn.content)).toEqual(['второй', 'два'])
  })

  it('не тащит пустой ответ', () => {
    const empty: Exchange = { ...createExchange('вопрос'), phase: 'done' }

    expect(asTurns([empty], 6)).toEqual([])
  })
})

describe('свёрнутая часть', () => {
  it('не уходит на сервер второй раз', () => {
    // Иначе памятка не экономит ничего: реплики продолжают ехать
    // целиком, а рядом с ними едет ещё и их пересказ.
    const old = { ...done('давний', 'ответ'), startedAt: 1000 }
    const fresh = { ...done('свежий', 'ответ'), startedAt: 2000 }

    const turns = asTurns([old, fresh], 6, 1500)

    expect(turns.map((turn) => turn.content)).toEqual(['свежий', 'ответ'])
  })
})

describe('asOpenTables', () => {
  const dataset = (over: Partial<Dataset> = {}): Dataset => ({
    handle: 'T98D4', kind: 'pipes.fleet', title: 'Топ-5 труб',
    columns: [], rows: [{ pipe_id: 'PP-0035' }], total_found: 39, scanned: 40,
    offset: 0, truncated: true, hint: '', taken_at: '', skipped: [],
    next_cursor: 'fleet|0.0000|5|5', error: null, ...over,
  })

  const withTables = (datasets: Dataset[]): Exchange => ({
    ...createExchange('дай топ 5 труб'),
    phase: 'done',
    datasets,
  })

  it('отдаёт курсор таблицы, которую человек видит', () => {
    // Без этого «следующие 5» — догадка: агент шёл за первой страницей
    // заново и показывал те же строки второй раз.
    const tables = asOpenTables([withTables([dataset()])])

    expect(tables).toHaveLength(1)
    expect(tables[0].next_cursor).toBe('fleet|0.0000|5|5')
    expect(tables[0].shown).toBe(1)
  })

  it('молчит о таблицах, которые листать некуда', () => {
    // Строка состояния стоит токенов на КАЖДОМ шаге агента. Платить за
    // сообщение «дальше ничего нет» незачем.
    expect(asOpenTables([withTables([dataset({ next_cursor: '' })])])).toEqual([])
  })

  it('не предлагает листать отказ инструмента', () => {
    const failed = dataset({
      error: { code: 'stale', message: 'Расчёт устарел.', retriable: true, retry_after_s: 60, hint: '' },
    })
    expect(asOpenTables([withTables([failed])])).toEqual([])
  })

  it('берёт последние таблицы, а не все за день', () => {
    const old = withTables([dataset({ handle: 'TOLD1' }), dataset({ handle: 'TOLD2' })])
    const fresh = withTables([dataset({ handle: 'TNEW1' }), dataset({ handle: 'TNEW2' })])

    expect(asOpenTables([old, fresh]).map((table) => table.handle)).toEqual(['TNEW1', 'TNEW2'])
  })
})
