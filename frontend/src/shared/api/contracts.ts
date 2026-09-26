/**
 * Контракт с бэкендом.
 *
 * Ответ модели приходит СТРУКТУРОЙ, а не текстом: статус, текст, цитаты с
 * номерами фрагментов и результатом дословной проверки. Именно поэтому
 * интерфейс может показать отказ как отдельное состояние, а не как абзац,
 * неотличимый от ответа, и не обязан угадывать эвристиками, что перед ним.
 */

export type AnswerStatus =
  | 'answered'
  | 'not_found'
  | 'need_clarification'
  | 'no_context'
  | 'error'

export type FinishReason = 'stop' | 'length' | 'cancelled' | 'error' | 'timeout'

export type Citation = {
  fragment: number
  /**
   * Текст цитаты, вырезанный СЕРВЕРОМ из фрагмента. Модель его не пишет — она
   * только указывает место, — поэтому текст дословен по построению. При
   * неудачной опоре поле пустое.
   */
  quote: string
  /** Нашлось ли указанное моделью место в том фрагменте, на который она ссылается. */
  ok: boolean
  reason: string
  /** Что именно указала модель. Нужно для разбора неудач, не для показа. */
  handle: string
}

export type RetrievedFragment = {
  number: number
  /** Номер куска в базе — по нему интерфейс ставит ссылку в точное место. */
  chunk_id: number
  /**
   * Живые данные из сервиса, а не кусок документа.
   *
   * Разница не косметическая. Цитата из документа верна, пока документ не
   * переписали; число из API верно на момент снятия и через час может быть
   * другим. И вести по такой ссылке некуда: документа с этим текстом не
   * существует.
   */
  live: boolean
  doc_id: string
  doc_title: string
  heading_path: string
  page_from: number
  page_to: number
  source_label: string
  found_by: string
  cosine: number | null
  rrf: number
  doc_status: string
  doc_version: string
  body: string
}

/**
 * Что делал агентский шаг.
 *
 * `null` означает «агент не участвовал вовсе» и это НЕ то же самое, что
 * «участвовал и решил, что найденного хватает». Первое — обычная цепочка,
 * второе — работа агента с нулевым результатом. Свести их в одно значило бы
 * лишить себя возможности понять, включался ли режим вообще.
 */
export type AgentTrace = {
  used_tools: boolean
  /** Цикл дошёл до предела шагов. Сам по себе не повод для тревоги. */
  hit_limit: boolean
  /**
   * Шаги кончились, когда след был ЖИВОЙ: последний вызов принёс новые
   * фрагменты, значит следующий мог принести ещё.
   *
   * Предупреждаем пользователя ровно по этому полю, а не по `hit_limit`.
   * Раньше по `hit_limit` — и «контекст может быть неполным» висело под
   * полными правильными ответами каждый раз, когда модель последним шагом
   * сходила впустую. Предупреждение, которое врёт через раз, перестают
   * читать за неделю, и дальше оно не работает уже никогда.
   */
  trail_cut: boolean
  /**
   * Чем принято решение: «инструменты», «схема» или «не спрашивали».
   *
   * Нужно затем, что «агент ничего не сделал» выглядит одинаково и когда
   * модель решила, что хватает, и когда она вообще не умеет вызывать
   * инструменты. Без различия чинишь наугад — мы на этом потеряли вечер.
   */
  mechanism: string
  /** Что модель ответила вместо вызова. Пусто, если вызвала. */
  declined_with: string
  steps: Array<{
    number: number
    tool: string
    arguments: Record<string, unknown>
    result: string
    added: number
    ok: boolean
  }>
}

export type MetaEvent = {
  trace_id: string
  prompt_version: string
  provider: string
  model: string
  embed_model: string
  agent: AgentTrace | null
  /**
   * Что система знала о прошлом разговоре и чем в итоге искала.
   *
   * `search_question` — ВТОРОЙ запрос поиска, а не замена первого. Вопрос
   * человека не подменяется нигде: он всегда уходит в поиск как есть.
   * Обрывок вроде «а 10?» дополнительно подпирается склейкой с прошлым
   * вопросом, и вот она здесь. Пусто — значит вопрос был понятен сам.
   *
   * Показывать обязательно: без этого «нашлось не то» не разобрать —
   * человек видит короткий вопрос, а искали ещё и длинным.
   */
  history: { turns: number; search_question: string }
  retrieval: {
    best_cosine: number
    floor: number
    passed_floor: boolean
    hits_total: number
    duplicates_dropped: number
  }
  fragments: RetrievedFragment[]
}

/**
 * Памятка по свёрнутой части разговора.
 *
 * Приходит ПОСЛЕ ответа и далеко не в каждом запросе: сворачивание — это
 * отдельный вызов модели, и делается он пачкой, раз в несколько вопросов.
 * Браузер кладёт памятку в свою базу и с этого момента шлёт её вместо
 * свёрнутых реплик.
 */
/**
 * Чем агент занят прямо сейчас.
 *
 * Приходит по мере работы, до ответа. Агент работает секунды — решает,
 * идёт в базу, решает снова, — и всё это время человек смотрел в пустой
 * экран, не зная, думает система или зависла.
 */
export type StepEvent = { text: string }

export type MemoryEvent = {
  summary: string
  folded_turns: number
}

/**
 * Таблица, добытая инструментом. Приходит ДО ответа, отдельным событием.
 *
 * Строк этой таблицы модель не видела. Это не деталь реализации, а самое
 * важное свойство: то, что не сгенерировано, не может быть сгенерировано
 * неправильно. Раньше модель переписывала таблицу в текст ответа — и
 * путала цифры («PP-0001, выработка 7%» вместо 77.0), выдумывала строки,
 * бросала список на середине.
 *
 * `total_found` против длины `rows` — то, что первым терялось в пересказе:
 * «нашлось 40, показано 5» превращалось в «топ-5», и человек делал вывод
 * по выборке, приняв её за весь парк.
 */
export type Dataset = {
  handle: string
  kind: string
  title: string
  columns: Array<{ key: string; title: string; unit: string; kind: string }>
  rows: Array<Record<string, string | number | null>>
  total_found: number
  scanned: number
  /**
   * С какой по счёту записи начинается показанное.
   *
   * Оболочка, которая объявляет о скрытых строках, обязана давать способ
   * их достать. Без этого поля на просьбу «следующие пять» модель
   * запрашивала список побольше и пересказывала из него хвост — то самое
   * переписывание таблицы, от которого мы уходили.
   */
  offset: number
  truncated: boolean
  hint: string
  taken_at: string
  /** Строки, по которым данных получить не удалось. Молчать о них нельзя. */
  skipped: string[]
  /** Метка следующей страницы. Пусто — дальше ничего нет. */
  next_cursor: string
  /**
   * Отказ инструмента. `retriable` решает, предлагать ли повтор: «расчёт
   * устарел, через минуту» и «такой трубы нет» — разные ответы человеку.
   */
  error: {
    code: string
    message: string
    retriable: boolean
    retry_after_s: number | null
    hint: string
  } | null
}

export type AnswerEvent = {
  status: AnswerStatus
  answer: string
  clarifying_question: string
  schema_valid: boolean
  schema_error: string
  citations: Citation[]
  citations_failed: number
  /**
   * Сколько ссылок сервер снял с отказа.
   *
   * Модель иногда пишет «сведений нет» и тут же ссылается на фрагмент.
   * Подтверждать у отказа нечего, поэтому такие ссылки до интерфейса не
   * доходят, но число остаётся: по нему видно, что это не идеальный отказ, а
   * отказ с противоречием внутри.
   */
  citations_dropped: number
  /**
   * Идентификаторы из ответа, которых нет ни в источниках, ни в вопросе.
   *
   * Закрывает дыру, которую проверка цитат закрыть не может: схема
   * разрешает четыре опоры, а в ответе списком строк бывает десять.
   * Остальные шесть проходили как есть — и «PP-0029, выработка 72%»
   * выглядело на экране точно так же, как настоящая строка из сервиса.
   */
  unknown_refs: string[]
  /**
   * Обозначения, у которых числа в ответе разошлись с источником.
   *
   * Отдельно от `unknown_refs`, потому что это другая беда: там «такой
   * трубы нет», здесь «труба есть, а цифра не та». Вторая тише и
   * опаснее — «PP-0001, выработка 7%» вместо 77% читается как обычная
   * строка отчёта.
   */
  mismatched_refs: string[]
}

export type DoneEvent = {
  finish_reason: FinishReason
  stop_reason: string
  /**
   * Кто ответил НА САМОМ ДЕЛЕ.
   *
   * В событии meta стоит намерение: оно уходит до генерации, когда ответчик
   * ещё не известен. Если основной провайдер откажет и вступит резерв, пары
   * разойдутся — и это надо показать, а не сгладить. Молчаливая подмена
   * модели — ровно та ошибка, из-за которой потом не сходятся замеры.
   */
  provider: string
  model: string
  usage: { prompt_tokens: number; completion_tokens: number; cost_rub: number }
  timing: { ttft_ms: number | null }
  trace_id: string
}

/**
 * Одна строчка выпадающего списка моделей.
 *
 * Список приходит с сервера, а не зашит во фронте: зашитый устаревает молча.
 * Мы на этом уже обожглись — в настройках стояло имя модели, которой у
 * сервиса больше нет, и отказ выглядел как «облако не работает».
 */
export type ModelChoice = {
  provider: string
  model: string
  is_default: boolean
}

export type ErrorEvent = {
  code: string
  message: string
  detail?: string
  retry_after_s?: number | null
}

export type DocumentSummary = {
  doc_id: string
  title: string
  project: string
  owner: string
  updated: string
  version: string
  status: string
  page_count: number
  chunk_count: number
}

export type DocumentChunk = {
  chunk_id: number
  ordinal: number
  heading_path: string
  body: string
  page_from: number
  page_to: number
}

export type DocumentDetail = DocumentSummary & {
  source_path: string
  pdf_path: string
  chunks: DocumentChunk[]
}

/**
 * Показание одной метрики оповещения.
 *
 * Пороги приходят вместе со значением сознательно: цифра «0.07» сама по
 * себе не говорит ничего, а «0.07 при пороге внимания 0.01» говорит всё —
 * и человеку не надо искать регламент, чтобы понять, плохо это или нет.
 *
 * `value` равен null, когда считать не по чему: мало наблюдений или не
 * задан бюджет. Это НЕ ноль. Ноль означал бы «всё хорошо», а null — «не с
 * чем сравнить», и подменять второе первым нельзя.
 */
export type AlertMetric = {
  metric: string
  title: string
  value: number | null
  level: 'ok' | 'warn' | 'crit'
  warn_at: number
  crit_at: number
  detail: string
  source: string
}

export type AlertReport = {
  level: 'ok' | 'warn' | 'crit'
  window_minutes: number
  samples: number
  metrics: AlertMetric[]
  /** Переходы уровня. Текущее значение говорит «что сейчас», история — «что было ночью». */
  events: Array<{
    at: number
    metric: string
    level: string
    previous: string
    value: number | null
    text: string
  }>
  note?: string
}

/**
 * Кто спрашивает и что ему можно.
 *
 * Права приходят с сервера и нужны интерфейсу ровно для одного: не
 * показывать кнопку, которая вернёт отказ. Само ограничение держит сервер —
 * проверка в браузере это удобство, а не защита, потому что браузер
 * открывается инструментами разработчика.
 *
 * Здесь ЧИСЛО закрытых проектов, а не их имена. Перечень закрытого — сам по
 * себе сведение о том, что в системе есть: «вам закрыт SURVEYD» сообщает о
 * существовании SURVEYD.
 */
export type Viewer = {
  name: string
  roles: string[]
  rights: { agent: boolean; live_data: boolean }
  closed_projects: number
  auth_mode: string
}
