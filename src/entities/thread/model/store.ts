import { create } from 'zustand'
import { dropThread, loadThreads, saveThread } from '../api/storage'
import { NEW_TITLE, newThread, TITLE_CHARS, type Thread } from './thread'

/**
 * Список диалогов и текущий выбранный.
 *
 * Здесь НЕТ реплик. Хранилище списка и хранилище ленты разделены нарочно:
 * список нужен всегда (он в меню), а лента — только для открытого диалога.
 * Держи их вместе — и переключение диалога заставляло бы перерисовываться
 * меню, а каждый пришедший токен ответа — весь список.
 *
 * Лентой заведует `features/ask-assistant`: там же живёт отправка запроса,
 * и разрывать «показать» и «сохранить» между двумя слоями значило бы
 * гарантированно их рассинхронизировать.
 */

type ThreadsState = {
  threads: Thread[]
  currentId: string | null
  /** Список ещё не читали из базы: до этого момента «пусто» ничего не значит. */
  ready: boolean
  load: () => Promise<void>
  /** Создаёт диалог и делает его текущим. Возвращает созданный. */
  start: (title: string) => Thread
  select: (id: string | null) => void
  /** Обновляет время последней активности и, если надо, заголовок. */
  touch: (id: string, title?: string) => void
  remove: (id: string) => Promise<void>
  /** Записать памятку и отметить, до какого момента разговор свёрнут. */
  fold: (id: string, summary: string, upTo: number) => void
}

export const useThreads = create<ThreadsState>((set, get) => ({
  threads: [],
  currentId: null,
  ready: false,

  async load(): Promise<void> {
    const threads = await loadThreads()
    set((state) => ({
      threads,
      ready: true,
      // Открываем самый свежий: человек закрыл вкладку посреди разговора
      // и вернулся — разговор должен быть на месте. Но если диалог уже
      // выбран (страницу успели потыкать), выбор не перебиваем.
      currentId: state.currentId ?? threads[0]?.id ?? null,
    }))
  },

  start(title: string): Thread {
    const thread = newThread(title)
    set((state) => ({ threads: [thread, ...state.threads], currentId: thread.id }))
    void saveThread(thread)
    return thread
  },

  select(id: string | null): void {
    set({ currentId: id })
  },

  touch(id: string, title?: string): void {
    const found = get().threads.find((thread) => thread.id === id)
    if (!found) return
    const updated: Thread = {
      ...found,
      updatedAt: Date.now(),
      // Заголовок задаётся ОДИН раз, первым вопросом, и дальше не
      // переписывается. Иначе диалог, который человек нашёл в списке по
      // первому вопросу, завтра назывался бы последним — и найти его
      // снова было бы нечем.
      title: found.title === NEW_TITLE && title
        ? title.trim().slice(0, TITLE_CHARS)
        : found.title,
    }
    set((state) => ({
      threads: [updated, ...state.threads.filter((thread) => thread.id !== id)],
    }))
    void saveThread(updated)
  },

  fold(id: string, summary: string, upTo: number): void {
    const found = get().threads.find((thread) => thread.id === id)
    if (!found || !summary) return
    // Отметку двигаем только ВПЕРЁД. Два ответа могут прийти не в том
    // порядке, в каком их спрашивали (второй короче первого), и старая
    // памятка, приехавшая последней, вернула бы в переписку уже свёрнутое.
    const updated: Thread = {
      ...found,
      summary,
      foldedAt: Math.max(found.foldedAt, upTo),
    }
    set((state) => ({
      threads: state.threads.map((thread) => (thread.id === id ? updated : thread)),
    }))
    void saveThread(updated)
  },

  async remove(id: string): Promise<void> {
    set((state) => ({
      threads: state.threads.filter((thread) => thread.id !== id),
      currentId: state.currentId === id ? null : state.currentId,
    }))
    await dropThread(id)
  },
}))
