import type {
  AnswerEvent,
  Dataset,
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
  /**
   * Таблицы из инструментов. Приходят до ответа и живут отдельно от него.
   *
   * Отдельно — потому что они верны независимо от того, что напишет
   * модель и допишет ли вообще. Оборвался поток на полуслове — таблица
   * всё равно на экране и всё равно правильная.
   */
  datasets: Dataset[]
  /**
   * Чем агент занят. Живёт только до ответа: как только текст пошёл,
   * рассказывать о процессе больше незачем — виден результат.
   */
  step: string
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
    datasets: [],
    step: '',
    answer: null,
    done: null,
    error: null,
    startedAt: Date.now(),
  }
}

/**
 * Запись из браузерной базы, приведённая к текущей форме обмена.
 *
 * База переживает выкатки: запись, сохранённая до появления поля, придёт
 * без него, и карточка упадёт на первом же `.map`. Поэтому под
 * прочитанное подкладываем значения по умолчанию для всех полей.
 */
export function restoreExchange(stored: Partial<Exchange> & { id: string }): Exchange {
  return {
    ...createExchange(stored.question ?? ''),
    ...stored,
    datasets: stored.datasets ?? [],
  }
}

/** Текст, который показываем: структурированный ответ, если он уже пришёл. */
export function displayText(exchange: Exchange): string {
  return exchange.answer?.answer?.trim() || exchange.streamedText
}

export function isBusy(phase: ExchangePhase): boolean {
  return phase === 'streaming'
}
