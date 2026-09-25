import type {
  AnswerEvent,
  DoneEvent,
  ErrorEvent,
  MetaEvent,
} from '@/shared/api/contracts'

/**
 * Состояние одного обмена «вопрос — ответ».
 *
 * `phase` — явный конечный автомат, а не набор булевых флагов. Флаги
 * позволяют «невозможные» комбинации (активная кнопка отправки во время
 * идущего потока, спиннер после ошибки), автомат — нет.
 */
export type ExchangePhase = 'idle' | 'streaming' | 'stopped' | 'done' | 'error'

export type Exchange = {
  id: string
  question: string
  phase: ExchangePhase
  /** Текст, пришедший потоком. Сохраняется при отмене и при обрыве. */
  streamedText: string
  meta: MetaEvent | null
  answer: AnswerEvent | null
  done: DoneEvent | null
  error: ErrorEvent | null
  startedAt: number
}

export function createExchange(question: string): Exchange {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    question,
    phase: 'idle',
    streamedText: '',
    meta: null,
    answer: null,
    done: null,
    error: null,
    startedAt: Date.now(),
  }
}

/** Текст, который показываем: структурированный ответ, если он уже пришёл. */
export function displayText(exchange: Exchange): string {
  return exchange.answer?.answer?.trim() || exchange.streamedText
}

export function isBusy(phase: ExchangePhase): boolean {
  return phase === 'streaming'
}
