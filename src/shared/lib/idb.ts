/**
 * Тонкая обёртка над IndexedDB: промисы вместо событий.
 *
 * ПОЧЕМУ IndexedDB, А НЕ localStorage
 *
 * localStorage держит строки, ограничен примерно пятью мегабайтами и
 * работает СИНХРОННО — то есть блокирует поток отрисовки. Диалог с
 * ассистентом хранит объекты: ответ, метаданные, найденные фрагменты. Через
 * десяток вопросов это сотни килобайт, а через сотню — предел, после
 * которого запись начинает падать. Причём падать молча, если не ловить:
 * человек закрывает вкладку в уверенности, что переписка сохранена.
 *
 * IndexedDB асинхронна, хранит объекты как есть, имеет индексы и квоту в
 * сотни мегабайт. Цена — многословный API на событиях, и вот его мы и
 * прячем здесь.
 *
 * ПОЧЕМУ НЕ БИБЛИОТЕКА
 *
 * Нужны четыре операции над двумя хранилищами. Библиотека принесла бы
 * ещё одну зависимость в сборку ради кода, который здесь помещается на
 * экран.
 *
 * БАЗЫ МОЖЕТ НЕ БЫТЬ ВООБЩЕ
 *
 * В приватном окне Firefox, при запрете хранилища в настройках, при
 * переполненной квоте IndexedDB недоступна или падает на записи. Это не
 * повод ломать приложение: ассистент обязан работать и без памяти —
 * просто без памяти. Поэтому каждая операция здесь возвращает значение по
 * умолчанию при отказе, а не бросает исключение наверх.
 */

const DB_NAME = 'pipewiki'
const DB_VERSION = 1

export const THREADS = 'threads'
export const EXCHANGES = 'exchanges'

/** Индекс по диалогу: без него выборка реплик была бы перебором всей базы. */
export const BY_THREAD = 'by_thread'

let handle: Promise<IDBDatabase | null> | null = null

/**
 * Открывает базу один раз на вкладку.
 *
 * Результат кэшируется вместе с неудачей: если хранилище запрещено, оно
 * запрещено до конца сессии, и пробовать снова на каждую запись значит
 * дёргать браузер впустую.
 */
export function openDb(): Promise<IDBDatabase | null> {
  if (handle) return handle

  handle = new Promise((resolve) => {
    if (typeof indexedDB === 'undefined') {
      resolve(null)
      return
    }

    let request: IDBOpenDBRequest
    try {
      request = indexedDB.open(DB_NAME, DB_VERSION)
    } catch {
      resolve(null)
      return
    }

    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains(THREADS)) {
        db.createObjectStore(THREADS, { keyPath: 'id' })
      }
      if (!db.objectStoreNames.contains(EXCHANGES)) {
        const store = db.createObjectStore(EXCHANGES, { keyPath: 'id' })
        // Составной ключ индекса: сначала диалог, потом время. Так выборка
        // по одному диалогу приходит уже отсортированной, и сортировать
        // руками не надо.
        store.createIndex(BY_THREAD, ['threadId', 'createdAt'])
      }
    }

    request.onsuccess = () => resolve(request.result)
    request.onerror = () => resolve(null)
    // Другая вкладка держит старую версию базы и не даёт обновиться.
    // Молчать здесь нельзя, но и ломаться незачем: работаем без памяти.
    request.onblocked = () => resolve(null)
  })

  return handle
}

/** Оборачивает запрос IndexedDB в промис. Отказ — это `null`, не исключение. */
function ask<T>(request: IDBRequest<T>): Promise<T | null> {
  return new Promise((resolve) => {
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => resolve(null)
  })
}

export async function putAll(store: string, records: unknown[]): Promise<boolean> {
  if (!records.length) return true
  const db = await openDb()
  if (!db) return false

  return new Promise((resolve) => {
    let transaction: IDBTransaction
    try {
      transaction = db.transaction(store, 'readwrite')
    } catch {
      resolve(false)
      return
    }
    const target = transaction.objectStore(store)
    for (const record of records) target.put(record)
    transaction.oncomplete = () => resolve(true)
    // Сюда приходит и QuotaExceededError — самая вероятная ошибка записи.
    // Возвращаем `false`, а не бросаем: вызывающий код решит, стоит ли
    // сообщать человеку, и ответ на экране от этого не пострадает.
    transaction.onerror = () => resolve(false)
    transaction.onabort = () => resolve(false)
  })
}

export async function getAll<T>(store: string): Promise<T[]> {
  const db = await openDb()
  if (!db) return []
  try {
    const result = await ask(db.transaction(store, 'readonly').objectStore(store).getAll())
    return (result as T[] | null) ?? []
  } catch {
    return []
  }
}

/** Все записи одного диалога, уже по возрастанию времени — так устроен индекс. */
export async function getByThread<T>(store: string, threadId: string): Promise<T[]> {
  const db = await openDb()
  if (!db) return []
  try {
    const index = db.transaction(store, 'readonly').objectStore(store).index(BY_THREAD)
    const range = IDBKeyRange.bound([threadId, -Infinity], [threadId, Infinity])
    const result = await ask(index.getAll(range))
    return (result as T[] | null) ?? []
  } catch {
    return []
  }
}

export async function remove(store: string, keys: IDBValidKey[]): Promise<void> {
  if (!keys.length) return
  const db = await openDb()
  if (!db) return
  try {
    const target = db.transaction(store, 'readwrite').objectStore(store)
    for (const key of keys) target.delete(key)
  } catch {
    // Удалять нечего или база недоступна — обе ситуации не стоят ошибки
    // на экране.
  }
}
