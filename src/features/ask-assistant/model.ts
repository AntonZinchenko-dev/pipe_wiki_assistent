/**
 * Состояние диалога с ассистентом.
 *
 * Здесь собраны четыре требования раздела 5 чек-листа, каждое из которых
 * обычно забывают:
 *
 * 1. Кнопка «Стоп» рвёт СОЕДИНЕНИЕ, а не только останавливает отрисовку.
 *    Иначе модель продолжает генерировать, а мы продолжаем за это платить —
 *    просто пользователь этого больше не видит.
 * 2. Новый запрос отменяет предыдущий. Иначе два потока пишут в одно
 *    состояние, и на экране получается каша из перемешанных ответов.
 * 3. Полученная часть ответа сохраняется при отмене, ошибке и обрыве. Человек
 *    нажал «стоп» именно потому, что уже увидел достаточно.
 * 4. Перерисовка пачкуется по кадрам. Модель отдаёт десятки токенов в
 *    секунду; ререндер на каждый токен тормозит страницу ровно в тот момент,
 *    когда её читают.
 */

import { create } from 'zustand'
import { createExchange, type Exchange, type ExchangePhase } from '@/entities/answer'
import { useAgentMode, useModelChoice } from '@/entities/chat-model'
import { loadRecords, saveRecord, trimThread, useThreads } from '@/entities/thread'
import { postChat } from '@/shared/api/client'
import { asOpenTables, asTurns } from './lib/turns'
import { readSse } from '@/shared/api/sse'
import type {
  AnswerEvent,
  Dataset,
  DoneEvent,
  ErrorEvent,
  MemoryEvent,
  MetaEvent,
} from '@/shared/api/contracts'
import { config } from '@/shared/config'

type AssistantState = {
  exchanges: Exchange[]
  activeId: string | null
  /**
   * Диалог, которому принадлежит лента на экране.
   *
   * Хранится ЗДЕСЬ, а не только в списке диалогов, и это не дублирование.
   * Пока идёт поток, человек может переключить диалог в меню; сохранять
   * пришедший ответ надо в тот диалог, в котором задавали вопрос, а не в
   * тот, что открыт сейчас. Поле отвечает на вопрос «чья эта лента», а
   * `useThreads.currentId` — на вопрос «что человек смотрит».
   */
  threadId: string | null
  ask: (question: string) => Promise<void>
  stop: () => void
  /** Начать новый диалог: лента пустеет, память обнуляется. */
  startNew: () => void
  /** Показать сохранённый диалог. `null` — пустая лента. */
  openThread: (id: string | null) => Promise<void>
}

/** Контроллер живёт вне стора: это не состояние интерфейса, а ресурс. */
let controller: AbortController | null = null

/** Тип функции обновления стора — нужен, чтобы слить буфер вне кадра. */
type SetState = (
  updater: (state: AssistantState) => Partial<AssistantState>,
) => void

/** Буфер текста между кадрами: копим токены, пишем в стор раз в кадр. */
let pending = ''
let flushHandle: number | null = null

/**
 * Отменяет запланированный кадр, НЕ выбрасывая накопленный текст.
 *
 * Раньше здесь стояло `pending = ''`, и это противоречило собственному
 * требованию файла — «полученная часть ответа сохраняется при отмене».
 * Порядок был жёсткий: `controller.abort()` отклоняет чтение только в
 * микротаске, а `cancelFlush()` выполнялся синхронно следующей строкой, то
 * есть раньше, чем `finally` успевал слить буфер.
 *
 * Итог: по «Стопу» терялся весь текст, накопленный с последнего кадра. А на
 * скрытой вкладке, где requestAnimationFrame не вызывается вовсе, буфер
 * держал ВЕСЬ ответ — человек возвращался, задавал новый вопрос, и предыдущий
 * обмен оставался пустым. Без единой ошибки на экране.
 */
function cancelFlush(): void {
  if (flushHandle !== null) {
    cancelAnimationFrame(flushHandle)
    flushHandle = null
  }
}

/** Сливает буфер немедленно, минуя кадр. Для отмены и завершения. */
function flushNow(id: string, set: SetState): void {
  cancelFlush()
  if (!pending) return
  const chunk = pending
  pending = ''
  set((state) => ({
    exchanges: state.exchanges.map((exchange) =>
      exchange.id === id
        ? { ...exchange, streamedText: exchange.streamedText + chunk }
        : exchange,
    ),
  }))
}

export const useAssistant = create<AssistantState>((set, get) => {
  function patch(id: string, changes: Partial<Exchange>): void {
    set((state) => ({
      exchanges: state.exchanges.map((exchange) =>
        exchange.id === id ? { ...exchange, ...changes } : exchange,
      ),
    }))
  }

  function setPhase(id: string, phase: ExchangePhase): void {
    patch(id, { phase })
  }

  /** Сливает накопленные токены в стор. Вызывается не чаще раза в кадр. */
  function scheduleFlush(id: string): void {
    if (flushHandle !== null) return
    flushHandle = requestAnimationFrame(() => {
      flushHandle = null
      if (!pending) return
      const chunk = pending
      pending = ''
      set((state) => ({
        exchanges: state.exchanges.map((exchange) =>
          exchange.id === id
            ? { ...exchange, streamedText: exchange.streamedText + chunk }
            : exchange,
        ),
      }))
    })
  }

  return {
    exchanges: [],
    activeId: null,
    threadId: null,

    stop(): void {
      const { activeId } = get()
      // Порядок важен: сначала фиксируем состояние, потом рвём соединение.
      // Иначе обработчик отмены в ask() успеет выставить свою фазу.
      if (activeId) {
        // Сначала досливаем прочитанное, потом рвём соединение: человек нажал
        // «Стоп» именно потому, что уже увидел достаточно.
        flushNow(activeId, set)
        setPhase(activeId, 'stopped')
      }
      controller?.abort()
      controller = null
      cancelFlush()
    },

    startNew(): void {
      controller?.abort()
      controller = null
      cancelFlush()
      // Здесь буфер выбрасывается сознательно: лента очищается целиком,
      // дописывать текст некуда.
      pending = ''
      set(() => ({ exchanges: [], activeId: null, threadId: null }))
      // Диалог из базы НЕ удаляем: «новый» и «удалить» — разные действия,
      // и человек, нажавший «новый», не ожидает потерять предыдущий.
      useThreads.getState().select(null)
    },

    async openThread(id: string | null): Promise<void> {
      // Уже открыт — выходим молча. Это не микрооптимизация: `ask`
      // заводит диалог сам и сразу выставляет `threadId`, а подписка в
      // интерфейсе срабатывает следом. Без этой проверки она загрузила бы
      // из базы пустой диалог поверх только что заданного вопроса.
      if (get().threadId === id) return

      controller?.abort()
      controller = null
      cancelFlush()
      pending = ''

      if (id === null) {
        set(() => ({ exchanges: [], activeId: null, threadId: null }))
        return
      }

      const stored = await loadRecords<Exchange>(id)
      // Пока читали из базы, человек мог переключиться ещё раз. Тогда
      // прочитанное относится к уже неактуальному диалогу, и класть его
      // на экран нельзя.
      if (useThreads.getState().currentId !== id) return
      set(() => ({
        exchanges: stored.slice(-config.maxThreadItems),
        activeId: null,
        threadId: id,
      }))
    },

    async ask(question: string): Promise<void> {
      const trimmed = question.trim()
      if (!trimmed) return

      // Новый запрос отменяет предыдущий — до того, как что-то поменяется в
      // состоянии, чтобы старый поток не дописал ничего в новый обмен.
      //
      // Но прочитанное у предыдущего обмена сохраняем: раньше буфер здесь
      // просто обнулялся, и вопрос, заданный на скрытой вкладке (где кадры не
      // выполняются вовсе), оставлял предыдущий ответ пустым при том, что он
      // был получен целиком.
      const previousId = get().activeId
      if (previousId) flushNow(previousId, set)
      controller?.abort()
      cancelFlush()

      // Память диалога собирается ДО того, как в ленту добавлен новый
      // обмен: в переписку уходит то, что уже состоялось, а не пустой
      // вопрос, который только что задали.
      // Диалог заводится при первом вопросе, а не при открытии панели.
      // Иначе список засорялся бы пустыми диалогами каждый раз, когда
      // человек открыл ассистента и передумал.
      const threads = useThreads.getState()
      const threadId = get().threadId ?? threads.currentId ?? threads.start(trimmed).id
      const thread = useThreads.getState().threads.find((item) => item.id === threadId)

      const sent = get().exchanges
      const turns = asTurns(sent, config.historyTurns, thread?.foldedAt ?? 0)
      // Курсоры таблиц берём из ВСЕЙ ленты, а не из несвёрнутой части:
      // памятка пересказывает слова, а таблица на экране не исчезает от
      // того, что реплики про неё свернулись.
      const openTables = asOpenTables(sent)
      // До какой отметки разговор окажется свёрнутым, если сервер пришлёт
      // памятку. Считаем ЗДЕСЬ, до ответа: к моменту, когда памятка
      // приедет, в ленте уже будет новый обмен, и по ней получилось бы,
      // что свёрнут и он тоже — то есть текущий вопрос выпал бы из
      // переписки, не попав ни в одну памятку.
      const foldUpTo = sent.length ? sent[sent.length - 1].startedAt : 0

      const exchange = createExchange(trimmed)
      set((state) => ({
        exchanges: [...state.exchanges, exchange].slice(-config.maxThreadItems),
        activeId: exchange.id,
        threadId,
      }))
      // Сохраняем СРАЗУ, ещё до ответа. Вкладку закрывают и на середине
      // генерации, и вопрос без ответа в списке честнее, чем исчезнувший
      // вопрос: по нему хотя бы видно, о чём спрашивали.
      void saveRecord(threadId, exchange, exchange.startedAt)

      controller = new AbortController()
      const signal = controller.signal
      setPhase(exchange.id, 'streaming')

      try {
        // Выбранную модель читаем В МОМЕНТ ОТПРАВКИ, а не подписываемся на
        // неё. Подписка перерисовывала бы всю ленту на каждое переключение в
        // шапке, а вопросу нужен ровно один снимок: чем спрашиваем сейчас.
        // Уже заданный вопрос менять задним числом нельзя — он ушёл.
        const response = await postChat(
          trimmed,
          useModelChoice.getState().chosen,
          useAgentMode.getState().on,
          signal,
          turns,
          thread?.summary ?? '',
          openTables,
        )

        for await (const event of readSse(response)) {
          if (signal.aborted) break

          switch (event.name) {
            case 'meta':
              patch(exchange.id, { meta: event.data as MetaEvent })
              break
            case 'delta':
              // Первый токен ответа — и рассказ о процессе больше не
              // нужен: виден результат.
              if (get().exchanges.find((item) => item.id === exchange.id)?.step) {
                patch(exchange.id, { step: '' })
              }
              pending += (event.data as { text: string }).text
              scheduleFlush(exchange.id)
              break
            case 'answer':
              // Финальная структура приходит одним событием: статус, цитаты с
              // результатом проверки, признак валидности схемы.
              patch(exchange.id, { answer: event.data as AnswerEvent })
              break
            case 'step':
              patch(exchange.id, { step: (event.data as { text: string }).text })
              break
            case 'dataset':
              // Таблица приходит ДО ответа и копится отдельно от него:
              // она верна независимо от того, что напишет модель и
              // допишет ли вообще.
              set((state) => ({
                exchanges: state.exchanges.map((item) =>
                  item.id === exchange.id
                    ? { ...item, datasets: [...item.datasets, event.data as Dataset] }
                    : item,
                ),
              }))
              break
            case 'memory':
              // Сервер свернул накопившуюся переписку в памятку. Кладём
              // её в базу и с этого момента шлём вместо свёрнутых реплик.
              useThreads
                .getState()
                .fold(threadId, (event.data as MemoryEvent).summary, foldUpTo)
              break
            case 'done':
              patch(exchange.id, { done: event.data as DoneEvent, phase: 'done' })
              break
            case 'error':
              patch(exchange.id, { error: event.data as ErrorEvent, phase: 'error' })
              break
            default:
              break
          }
        }

        // Поток закончился, а финального события так и не пришло: соединение
        // оборвалось. Показываем это явно — с сохранением полученного текста,
        // — а не оставляем интерфейс замершим на полуслове.
        const current = get().exchanges.find((item) => item.id === exchange.id)
        if (current && current.phase === 'streaming') {
          patch(exchange.id, {
            phase: 'error',
            error: {
              code: 'stream_truncated',
              message: 'Соединение прервано, ответ неполный.',
            },
          })
        }
      } catch (error) {
        const aborted = signal.aborted || (error as Error)?.name === 'AbortError'
        if (aborted) {
          // Отмена — не ошибка. Фаза уже выставлена в stop() либо перетрётся
          // следующим запросом; текст, который человек успел прочитать,
          // остаётся на экране.
          const current = get().exchanges.find((item) => item.id === exchange.id)
          if (current && current.phase === 'streaming') setPhase(exchange.id, 'stopped')
        } else {
          patch(exchange.id, {
            phase: 'error',
            error: {
              code: 'request_failed',
              message: (error as Error)?.message || 'Запрос не удался.',
            },
          })
        }
      } finally {
        // Досливаем то, что осталось в буфере: иначе последние токены
        // пропадут, если поток закончился между кадрами.
        // Досливаем остаток одной функцией — той же, что и при «Стопе».
        // Две копии этой логики однажды разойдутся.
        flushNow(exchange.id, set)
        if (get().activeId === exchange.id) set(() => ({ activeId: null }))

        // Сохраняем то, чем обмен закончился, — с ответом, метаданными и
        // фазой. Диалог при этом сохраняется в ТОТ, в котором задавали
        // вопрос: человек мог переключиться, пока шёл поток.
        const finished = get().exchanges.find((item) => item.id === exchange.id)
        if (finished) {
          void saveRecord(threadId, finished, finished.startedAt).then((saved) => {
            if (!saved) return
            // Обрезаем хвост только после удачной записи. Иначе при
            // переполнении квоты мы бы удаляли старое, не сумев записать
            // новое, — то есть теряли бы переписку обеими руками.
            void trimThread(threadId, config.maxThreadItems)
          })
        }
        useThreads.getState().touch(threadId, trimmed)
      }
    },
  }
})

export function selectActive(state: AssistantState): Exchange | undefined {
  return state.exchanges.find((exchange) => exchange.id === state.activeId)
}
