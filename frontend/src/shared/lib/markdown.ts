/**
 * Разбор ответа модели в дерево элементов. Белый список ПО ПОСТРОЕНИЮ.
 *
 * Зачем вообще разметка. Ответ вида «порядок такой: 1) … 2) … 3) …» сплошным
 * абзацем читается заметно хуже списка, а модель размечает его сама и
 * бесплатно. До сих пор мы просто показывали её вывод текстом.
 *
 * ПОЧЕМУ НЕ «ОТРЕНДЕРИТЬ MARKDOWN И ПОЧИСТИТЬ HTML»
 *
 * Обычный путь такой: markdown → HTML → санитайзер по белому списку тегов →
 * вставка в DOM. Он работает, им пользуются, и он же требует доверять
 * санитайзеру: список запрещённого приходится держать полным, а обходы
 * находят регулярно — через атрибуты, через `javascript:` в ссылке, через
 * вложенные теги, которые парсер и санитайзер разбирают по-разному.
 *
 * Здесь ход другой. Текст разбирается в НАШЕ дерево из пяти видов узлов, и
 * каждый узел рисуется заранее известным компонентом. HTML не собирается
 * нигде и ни на каком шаге: `dangerouslySetInnerHTML` в проекте не
 * встречается вовсе. Белый список тут не список запретов, который можно
 * обойти, а множество того, что вообще способно появиться на выходе.
 *
 * Разница практическая: чтобы пробить санитайзер, нужно найти дырку в его
 * правилах. Чтобы пробить это, нужно дописать сюда новый вид узла.
 *
 * ССЫЛОК НЕТ СОЗНАТЕЛЬНО
 *
 * `[текст](адрес)` не поддерживается, и `<a>` тут построить нечем. Адрес в
 * ответе модели берётся из документов вики, то есть из текста, который
 * пишут люди; кликабельная ссылка оттуда — это увод пользователя куда
 * угодно с видом «так написано в регламенте». Единственные ссылки в ответе —
 * номера фрагментов, и адрес для них собирает интерфейс из метаданных
 * поиска, а не модель из своего текста.
 */

export type Inline =
  | { kind: 'text'; text: string }
  | { kind: 'bold'; text: string }
  | { kind: 'italic'; text: string }
  | { kind: 'code'; text: string }
  | { kind: 'citation'; number: number }

export type Block =
  | { kind: 'paragraph'; content: Inline[] }
  | { kind: 'list'; ordered: boolean; items: Inline[][] }

const BULLET = /^\s*[-*•]\s+(.*)$/
const NUMBERED = /^\s*(\d+)[.)]\s+(.*)$/

/**
 * Инлайн-разметка одним проходом.
 *
 * Порядок вариантов в чередовании важен: код идёт первым, потому что внутри
 * обратных кавычек ничего разбирать нельзя — там показывают текст как есть,
 * включая звёздочки и квадратные скобки. Жирный идёт раньше курсива, иначе
 * `**` разберётся как две курсивные звёздочки.
 */
const INLINE = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\*[^*\n]+\*|\[\d{1,3}(?:\s*[,;]\s*\d{1,3})*\])/g

export function parseInline(text: string): Inline[] {
  const parts = text.split(INLINE).filter((part) => part !== '')
  // flatMap, а не map: одна скобка «[1, 2]» — это один кусок текста и ДВА
  // узла. Заводить ради этого отдельный вид узла значило бы обязать
  // каждого, кто рисует ответ, помнить про второй случай; развернуть здесь
  // — значит на выходе по-прежнему только одиночные ссылки.
  return parts.flatMap((part): Inline | Inline[] => {
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return { kind: 'code', text: part.slice(1, -1) }
    }
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return { kind: 'bold', text: part.slice(2, -2) }
    }
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
      return { kind: 'italic', text: part.slice(1, -1) }
    }
    // Одна скобка на несколько номеров: [1, 2].
    //
    // Так модели сносят источники, и запретить это промптом до конца не
    // выходит. Пока разбирался только одиночный номер, «[1, 2]» оставалось
    // обычным текстом: не ссылка, не кликается, и в разборе выглядело
    // ответом без источников. Первый номер берём здесь, остальные
    // распаковывает `expandCitations` — разбор инлайна отдаёт по узлу на
    // кусок текста, а тут кусок один, а узлов нужно несколько.
    const citation = /^\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]$/.exec(part)
    if (citation) {
      return citation[1]
        .split(/[,;]/)
        .map((value): Inline => ({ kind: 'citation', number: Number(value.trim()) }))
    }
    // Всё прочее — текст. Сюда попадают и угловые скобки, и незакрытые
    // звёздочки, и то, что выглядит как markdown-ссылка: ни один из этих
    // случаев не создаёт узла, кроме текстового.
    return { kind: 'text', text: part }
  })
}

export function parseAnswer(text: string): Block[] {
  const blocks: Block[] = []
  let paragraph: string[] = []
  let list: { ordered: boolean; items: string[] } | null = null

  const closeParagraph = () => {
    if (paragraph.length) {
      blocks.push({ kind: 'paragraph', content: parseInline(paragraph.join(' ')) })
      paragraph = []
    }
  }
  const closeList = () => {
    if (list) {
      blocks.push({
        kind: 'list',
        ordered: list.ordered,
        items: list.items.map(parseInline),
      })
      list = null
    }
  }

  for (const line of text.split('\n')) {
    const bullet = BULLET.exec(line)
    const numbered = NUMBERED.exec(line)

    if (bullet || numbered) {
      closeParagraph()
      const ordered = Boolean(numbered)
      const item = (bullet ? bullet[1] : numbered![2]).trim()
      // Смена вида списка закрывает предыдущий: маркированный и
      // нумерованный подряд — это два списка, а не один с разнобоем.
      if (list && list.ordered !== ordered) closeList()
      list = list ?? { ordered, items: [] }
      list.items.push(item)
      continue
    }

    if (!line.trim()) {
      closeParagraph()
      closeList()
      continue
    }

    closeList()
    paragraph.push(line.trim())
  }

  closeParagraph()
  closeList()
  return blocks
}
