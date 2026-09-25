import { EXCHANGES, THREADS, getAll, getByThread, putAll, remove } from '@/shared/lib/idb'
import type { Thread } from '../model/thread'

/**
 * Диалоги и их записи в браузерной базе.
 *
 * Сущность «диалог» НЕ ЗНАЕТ, что в нём лежит. Записи здесь — параметр
 * типа, а не конкретный обмен с ассистентом, и это не абстракция ради
 * абстракции: диалог и ответ — две соседние сущности одного слоя, и
 * прямая ссылка одной на другую запрещена правилами раскладки. Правило
 * не формальное — оно ровно про то, чтобы через полгода нельзя было
 * удалить ни одну из двух. Кто кладёт в диалог обмены, решает слой выше.
 *
 * Хранится запись ЦЕЛИКОМ, вместе с метаданными и найденными фрагментами,
 * — чтобы после перезагрузки страницы работало всё, включая блок «как
 * получен ответ» и переходы в документ. Хранить только текст было бы
 * дешевле, но тогда перезагрузка превращала бы полноценную ленту в её
 * выцветшую копию, и человек не понимал бы, куда делись источники.
 *
 * Отсюда и размер: один ответ с двумя десятками фрагментов — это десятки
 * килобайт. Поэтому число записей в диалоге ограничено, а переполнение
 * квоты обрабатывается как обычная ситуация, а не как поломка.
 */

/** Запись в базе: сама запись плюс к какому диалогу и когда она относится. */
type Stored<T> = T & { threadId: string; createdAt: number }

export async function loadThreads(): Promise<Thread[]> {
  const threads = await getAll<Thread>(THREADS)
  return threads.sort((a, b) => b.updatedAt - a.updatedAt)
}

export async function saveThread(thread: Thread): Promise<boolean> {
  return putAll(THREADS, [thread])
}

export async function loadRecords<T>(threadId: string): Promise<T[]> {
  const stored = await getByThread<Stored<T>>(EXCHANGES, threadId)
  return stored.map((record) => {
    const copy = { ...record } as Stored<T>
    delete (copy as { threadId?: string }).threadId
    delete (copy as { createdAt?: number }).createdAt
    return copy as T
  })
}

export async function saveRecord<T extends { id: string }>(
  threadId: string,
  record: T,
  at: number,
): Promise<boolean> {
  // Время берут СНАРУЖИ, а не `Date.now()` здесь: одна и та же запись
  // сохраняется дважды — при отправке вопроса и по завершении ответа, — и
  // вторая запись не должна перебрасывать её в конец ленты.
  return putAll(EXCHANGES, [{ ...record, threadId, createdAt: at }])
}

export async function dropThread(threadId: string): Promise<void> {
  const stored = await getByThread<{ id: string }>(EXCHANGES, threadId)
  await remove(EXCHANGES, stored.map((item) => item.id))
  await remove(THREADS, [threadId])
}

/**
 * Удаляет самые старые записи диалога сверх предела.
 *
 * Предел нужен не ради аккуратности, а ради квоты: без него один длинный
 * диалог однажды упрётся в лимит хранилища, и тогда перестанет
 * сохраняться ВСЁ, включая другие диалоги. Лучше потерять сотый вопрос
 * осознанно, чем первый — случайно.
 */
export async function trimThread(threadId: string, keep: number): Promise<void> {
  const stored = await getByThread<{ id: string }>(EXCHANGES, threadId)
  if (stored.length <= keep) return
  await remove(
    EXCHANGES,
    stored.slice(0, stored.length - keep).map((item) => item.id),
  )
}
