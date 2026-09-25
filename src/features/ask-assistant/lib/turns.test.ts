import { describe, expect, it } from 'vitest'
import { createExchange, type Exchange } from '@/entities/answer'
import { asTurns } from './turns'

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
