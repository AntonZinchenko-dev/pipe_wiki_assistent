import type { Exchange } from '@/entities/answer'
import type { OpenTableRef } from '@/shared/api/client'

/**
 * Что из ленты уходит на сервер как память диалога.
 *
 * Живёт ЗДЕСЬ, а не в сущности «диалог», и по делу: диалог не знает, что
 * такое обмен с ассистентом, и знать не должен — иначе две соседние
 * сущности склеиваются намертво. Превращение обменов в реплики — задача
 * того, кто отправляет запрос.
 *
 * Берём ТОЛЬКО завершённые обмены. Оборванный или ошибочный ответ в
 * переписке хуже его отсутствия: модель прочитает обрубок как свои прошлые
 * слова и будет достраивать из него, а человек уже видел на экране, что
 * ответ не состоялся.
 *
 * Берём ТЕКСТ ответа, а не структуру: ни цитат, ни метаданных, ни номеров
 * фрагментов. Номер [2] во вчерашнем ответе указывает на фрагмент из
 * вчерашнего поиска, и сегодня под этим номером лежит другой текст.
 * Отправить такую ссылку значит дать модели заведомо неверное
 * соответствие — причём выглядящее совершенно правдоподобно.
 */
export function asTurns(
  exchanges: Exchange[],
  pairs: number,
  foldedAt = 0,
): Array<{ role: 'user' | 'assistant'; content: string }> {
  const turns: Array<{ role: 'user' | 'assistant'; content: string }> = []
  for (const exchange of exchanges) {
    if (exchange.phase !== 'done') continue
    // Свёрнутое в памятку второй раз не отправляем. Иначе памятка не
    // экономит ничего: реплики продолжают ехать целиком, а рядом с ними
    // едет ещё и их пересказ.
    if (exchange.startedAt <= foldedAt) continue
    const text = (exchange.answer?.answer ?? exchange.streamedText).trim()
    if (!text) continue
    turns.push({ role: 'user', content: exchange.question })
    turns.push({ role: 'assistant', content: text })
  }
  // Хвост, а не начало: свежие реплики нужнее давних. То же правило, что на
  // сервере, — и дублирование сознательное. Сервер обязан резать сам,
  // потому что клиенту верить нельзя; клиент режет затем, чтобы не гонять
  // по сети то, что на той стороне всё равно выбросят.
  return turns.slice(-pairs * 2)
}

/**
 * Какие таблицы человек видит и чем листать их дальше.
 *
 * Здесь же, по той же причине, что и `asTurns`: превращение ленты в поля
 * запроса — дело отправителя. Берём только последние: смысл имеет то, что
 * на экране, а не всё, что показывали за день.
 *
 * Таблицы БЕЗ метки следующей страницы отбрасываем. Листать в них нечего, а
 * агенту каждая строка состояния стоит токенов на каждом шаге.
 */
export function asOpenTables(exchanges: Exchange[], limit = 2): OpenTableRef[] {
  const tables: OpenTableRef[] = []
  for (const exchange of exchanges) {
    if (exchange.phase !== 'done') continue
    for (const dataset of exchange.datasets) {
      if (!dataset.next_cursor || dataset.error) continue
      tables.push({
        handle: dataset.handle,
        title: dataset.title,
        offset: dataset.offset,
        shown: dataset.rows.length,
        total_found: dataset.total_found,
        next_cursor: dataset.next_cursor,
      })
    }
  }
  return tables.slice(-limit)
}
