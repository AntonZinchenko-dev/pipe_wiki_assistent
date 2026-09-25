import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import type { AnswerStatus } from '@/shared/api/contracts'
import { Badge, Disclosure, Field, Spinner } from '@/shared/ui/primitives'
import { IconCloud, IconCpu, IconDoc, IconSparkle } from '@/shared/ui/icons'
import { cost, ms, similarity, tokens } from '@/shared/lib/format'
import { parseAnswer, type Inline } from '@/shared/lib/markdown'
import { cn } from '@/shared/lib/cn'
import { displayText, type Exchange } from '../model/exchange'

/**
 * Карточка одного ответа.
 *
 * Отказ здесь — отдельное СОСТОЯНИЕ интерфейса, а не абзац текста. Разница
 * практическая: отказ, оформленный как обычный ответ, читается как «система
 * не справилась», а оформленный явно — как осознанное решение системы. И
 * только во втором случае люди начинают ему доверять.
 *
 * Переехала из `widgets/answer-card` сюда, в сущность. Причина не в красоте
 * раскладки по папкам: виджет ленты импортировал виджет карточки, а это
 * связь между слайсами одного слоя — та самая, из-за которой через полгода
 * нельзя удалить ни один. Карточка рисует сущность «ответ», её место рядом
 * с типами этой сущности.
 */

const STATUS_VIEW: Record<AnswerStatus, { label: string; tone: 'good' | 'warn' | 'bad' | 'info' }> =
  {
    answered: { label: 'ответ найден', tone: 'good' },
    not_found: { label: 'в документации нет ответа', tone: 'warn' },
    need_clarification: { label: 'нужно уточнение', tone: 'info' },
    no_context: { label: 'подходящих фрагментов нет', tone: 'warn' },
    error: { label: 'ошибка', tone: 'bad' },
  }

/**
 * Текст ответа рисуется деревом ИЗВЕСТНЫХ узлов, а не вставкой разметки.
 *
 * Вывод модели — недоверенный ввод: он собран из документов, которые кто-то
 * написал, и попадает прямиком в DOM. Разбор в `shared/lib/markdown` даёт
 * пять видов узлов, и каждый рисуется заранее известным компонентом. HTML
 * не собирается нигде: `dangerouslySetInnerHTML` в проекте не встречается
 * вовсе. Белый список здесь — не список запретов, который можно обойти, а
 * множество того, что вообще способно появиться на выходе.
 *
 * Номера [4] кликабельны, но ссылку строим МЫ, а не модель. Из текста
 * модели берётся только число, всё остальное — адрес документа и номер
 * куска — подставляется из метаданных поиска, то есть из того, что система
 * нашла сама. Модель не может ни привести на чужой адрес, ни сослаться на
 * фрагмент, которого не было в контексте: на номер без пары просто не
 * вешается ссылка.
 */
function InlineRun({
  nodes,
  target,
}: {
  nodes: Inline[]
  target: (number: number) => string | null
}) {
  const badge = 'mx-0.5 rounded bg-sky-50 px-1 text-xs font-semibold text-sky-700'
  return (
    <>
      {nodes.map((node, index) => {
        switch (node.kind) {
          case 'bold':
            return (
              <strong key={index} className="font-semibold">
                {node.text}
              </strong>
            )
          case 'italic':
            return <em key={index}>{node.text}</em>
          case 'code':
            return (
              <code
                key={index}
                className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[13px] text-slate-800"
              >
                {node.text}
              </code>
            )
          case 'citation': {
            const href = target(node.number)
            const label = `[${node.number}]`
            return href ? (
              <Link key={index} to={href} className={cn(badge, 'hover:bg-sky-100 hover:underline')}>
                {label}
              </Link>
            ) : (
              <span key={index} className={badge}>
                {label}
              </span>
            )
          }
          default:
            return <span key={index}>{node.text}</span>
        }
      })}
    </>
  )
}

function AnswerText({
  text,
  target,
}: {
  text: string
  target: (number: number) => string | null
}) {
  const blocks = parseAnswer(text)
  return (
    <div className="space-y-2 text-[15px] leading-relaxed text-slate-800">
      {blocks.map((block, index) =>
        block.kind === 'list' ? (
          block.ordered ? (
            <ol key={index} className="list-decimal space-y-1 pl-5">
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex}>
                  <InlineRun nodes={item} target={target} />
                </li>
              ))}
            </ol>
          ) : (
            <ul key={index} className="list-disc space-y-1 pl-5">
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex}>
                  <InlineRun nodes={item} target={target} />
                </li>
              ))}
            </ul>
          )
        ) : (
          <p key={index} className="whitespace-pre-wrap">
            <InlineRun nodes={block.content} target={target} />
          </p>
        ),
      )}
    </div>
  )
}

function Row({ children }: { children: ReactNode }) {
  return <div className="flex flex-wrap items-center gap-1.5">{children}</div>
}

const TOOL_NAMES: Record<string, string> = {
  search_wiki: 'поискал ещё раз',
  read_section: 'дочитал раздел',
  live_pipe: 'спросил сервис про трубу',
  live_fleet: 'спросил сервис про парк',
}

/**
 * Что делал агент, человеческими словами.
 *
 * Это не украшение и не отладка. Агентский шаг меняет состав контекста, то
 * есть меняет ответ, — и делает это невидимо. Скрытый шаг, влияющий на
 * результат, человек рано или поздно обнаруживает по расхождению цифр и
 * перестаёт доверять всей системе. Дешевле показать сразу.
 *
 * Аргументы вызова печатаем как есть: именно по ним видно, что модель
 * искала своими словами, а не словами вопроса, — ради чего шаг и заведён.
 */
function AgentSteps({ agent }: { agent: NonNullable<Exchange['meta']>['agent'] }) {
  if (!agent) return null
  if (!agent.used_tools) {
    // Раньше здесь стояло безусловное «посмотрел и решил, что достаточно»
    // — и это была неправда в половине случаев: модель могла просто не
    // уметь вызывать инструменты. Пять запросов подряд выглядели как
    // осознанные решения, а решений не было вовсе.
    const decided = agent.mechanism !== 'не спрашивали'
    return (
      <div className="space-y-1 text-xs text-slate-500">
        <p>
          {decided
            ? 'Агент посмотрел на найденное и решил, что инструменты не нужны.'
            : 'Агент не дошёл до решения.'}{' '}
          <span className="text-slate-400">способ: {agent.mechanism}</span>
        </p>
        {agent.declined_with ? (
          <p className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600">
            ответ модели вместо вызова: «{agent.declined_with}»
          </p>
        ) : null}
      </div>
    )
  }

  return (
    <div className="space-y-1">
      {agent.steps.map((step) => (
        <div
          key={step.number}
          className={cn(
            'rounded-lg border px-2.5 py-1.5 text-xs',
            step.ok ? 'border-slate-200 bg-white' : 'border-amber-200 bg-amber-50',
          )}
        >
          <span className="font-medium text-slate-700">
            {step.number}. {TOOL_NAMES[step.tool] ?? step.tool}
          </span>
          <span className="text-slate-500">
            {' '}
            {Object.entries(step.arguments)
              // С именем поля, а не одним значением. «0.8» в строке про
              // чтение раздела не объяснял ничего; «min_damage: 0.8»
              // сразу показал бы, что в вызов уехал чужой аргумент.
              .map(([name, value]) => `${name}: «${String(value)}»`)
              .join(', ')}
          </span>
          <span className="ml-1 text-slate-500">
            → {step.added > 0 ? `+${step.added} фрагм.` : 'ничего нового'}
          </span>
        </div>
      ))}
      <p className="text-xs text-slate-400">способ решения: {agent.mechanism}</p>
      {agent.trail_cut ? (
        <p className="text-xs text-amber-700">
          Шаги кончились на полпути: последний вызов ещё приносил новое.
        </p>
      ) : null}
    </div>
  )
}

export function AnswerCard({ exchange }: { exchange: Exchange }) {
  const { phase, answer, meta, done, error } = exchange
  const text = displayText(exchange)
  const status = answer?.status
  const view = status ? STATUS_VIEW[status] : null

  // Кто ответил на самом деле — из финального события, а не из meta: meta
  // уходит до генерации и знает только намерение. Если вступил резерв, эти
  // двое разойдутся, и показать надо настоящего.
  const provider = done?.provider || meta?.provider || ''
  const model = done?.model || meta?.model || ''
  const local = provider === 'ollama'

  // Документы, на которые реально сослался ответ. Не все найденные, а
  // именно процитированные: список «что нашлось» лежит ниже, в разборе.
  const citedNumbers = new Set((answer?.citations ?? []).map((citation) => citation.fragment))
  const sources = (meta?.fragments ?? []).filter((fragment) => citedNumbers.has(fragment.number))

  // Адрес конкретного фрагмента внутри документа. Якорь обязателен: в
  // документе на сорок кусков ссылка «в документ» означает «ищи сам», и
  // проверять цитату никто не станет.
  const byNumber = new Map((meta?.fragments ?? []).map((fragment) => [fragment.number, fragment]))
  function linkTo(number: number): string | null {
    const fragment = byNumber.get(number)
    if (!fragment) return null
    // Живые данные никуда не ведут: документа с этим текстом нет. Ссылка,
    // открывающая пустую страницу, хуже её отсутствия — человек решит, что
    // документ потерялся, и пойдёт искать несуществующую поломку.
    if (fragment.live) return null
    return `/wiki/${encodeURIComponent(fragment.doc_id)}#chunk-${fragment.chunk_id}`
  }

  return (
    <article className="space-y-3">
      <div className="flex justify-end">
        <p className="max-w-[85%] rounded-2xl rounded-br-md bg-slate-100 px-3.5 py-2 text-[15px] text-slate-800">
          {exchange.question}
        </p>
      </div>

      <div className="flex gap-2.5">
        <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-sky-50 text-sky-600">
          <IconSparkle className="size-4" />
        </span>

        <div className="min-w-0 flex-1 space-y-2.5">
          <Row>
            {view ? <Badge tone={view.tone}>{view.label}</Badge> : null}
            {phase === 'streaming' ? <Spinner label="модель отвечает" /> : null}
            {phase === 'stopped' ? <Badge tone="neutral">остановлено вами</Badge> : null}
            {done?.finish_reason === 'length' ? (
              <Badge tone="warn">обрезано по лимиту токенов</Badge>
            ) : null}
            {answer && !answer.schema_valid ? (
              <Badge tone="warn">схема: {answer.schema_error}</Badge>
            ) : null}
            {answer && answer.citations_failed > 0 ? (
              <Badge tone="bad">ссылок не подтверждено: {answer.citations_failed}</Badge>
            ) : null}
            {meta?.agent?.used_tools ? (
              <Badge tone="info">агент: {meta.agent.steps.length} шаг(а)</Badge>
            ) : null}
            {meta?.fragments.some((fragment) => fragment.live) ? (
              <Badge tone="warn">есть живые данные сервиса</Badge>
            ) : null}
          </Row>

          {/* Ошибка — это состояние, а не текст поверх ответа: показываем её
              отдельно и НЕ стираем то, что человек уже прочитал. */}
          {error ? (
            <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
              <p className="font-medium">{error.message}</p>
              {error.detail ? <p className="mt-1 text-xs opacity-80">{error.detail}</p> : null}
              {error.retry_after_s ? (
                <p className="mt-1 text-xs">Повторить можно через {error.retry_after_s} с.</p>
              ) : null}
            </div>
          ) : null}

          {text ? <AnswerText text={text} target={linkTo} /> : null}

          {/* Выдуманные идентификаторы — красным и над источниками.
              Это не придирка к оформлению: «PP-0029, выработка 72%» в
              ответе про бурильный инструмент читается как факт, а такой
              трубы в парке нет. Проверка цитат её пропустила — опор
              всего четыре, а строк в списке десять. */}
          {answer && answer.mismatched_refs?.length ? (
            <p className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
              Числа рядом с этими обозначениями не совпадают с источником:{' '}
              <span className="font-semibold">{answer.mismatched_refs.join(', ')}</span>. Сверьте
              их по блоку источников ниже — там значения такие, какими их отдал сервис.
            </p>
          ) : null}

          {answer && answer.unknown_refs?.length ? (
            <p className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
              В ответе названы обозначения, которых нет в источниках:{' '}
              <span className="font-semibold">{answer.unknown_refs.join(', ')}</span>. Модель их
              не нашла, а составила — проверьте эти строки по сервису вручную.
            </p>
          ) : null}

          {/* Уточняющий вопрос — отдельной плашкой, но только если он НЕ
              повторяет текст ответа. Модель регулярно задаёт вопрос прямо
              в тексте и оставляет поле пустым; сервер переносит его в
              поле, и без этой проверки одна и та же фраза печаталась на
              экране дважды подряд. */}
          {answer?.status === 'need_clarification' &&
          answer.clarifying_question &&
          answer.clarifying_question.trim() !== text.trim() ? (
            <p className="rounded-lg border border-sky-200 bg-sky-50 p-3 text-sm text-sky-900">
              {answer.clarifying_question}
            </p>
          ) : null}

          {answer && answer.citations.length > 0 ? (
            <ul className="space-y-1">
              {answer.citations.map((citation, index) => (
                <li
                  key={`${citation.fragment}-${index}`}
                  className={cn(
                    'rounded-lg border px-2.5 py-1.5 text-xs',
                    citation.ok
                      ? 'border-emerald-200 bg-emerald-50/60 text-emerald-900'
                      : 'border-red-200 bg-red-50/60 text-red-900',
                  )}
                >
                  <span className="font-semibold">[{citation.fragment}]</span>{' '}
                  {citation.ok ? 'из документа' : `не подтверждена — ${citation.reason}`}
                  {/*
                    Показываем только то, что вырезано из документа. При
                    неудачной опоре цитаты нет вообще — и это правильнее, чем
                    показать человеку текст, который сочинила модель: он
                    выглядит ровно как настоящий, и отличить их на экране
                    нельзя.
                  */}
                  {citation.quote ? (
                    <span className="mt-0.5 block opacity-80">«{citation.quote}»</span>
                  ) : null}
                </li>
              ))}
            </ul>
          ) : null}

          {sources.length > 0 ? (
            <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-2.5">
              <p className="px-0.5 pb-1.5 text-xs font-semibold text-slate-500">Источники</p>
              <ul className="space-y-1">
                {sources.map((fragment) => {
                  const inside = (
                    <>
                      {fragment.live ? (
                        <IconCloud className="mt-0.5 size-4 shrink-0 text-amber-500" />
                      ) : (
                        <IconDoc className="mt-0.5 size-4 shrink-0 text-slate-400" />
                      )}
                      <span className="min-w-0">
                        <span className="block truncate text-xs font-medium text-slate-800">
                          [{fragment.number}] {fragment.doc_title}
                        </span>
                        <span className="block truncate text-xs text-slate-500">
                          {fragment.heading_path}
                          {fragment.live ? '' : ` · ред. ${fragment.doc_version}`}
                        </span>
                      </span>
                    </>
                  )
                  // Живой источник — не ссылка, а карточка с пометкой. Он
                  // никуда не ведёт и означает «снимок на момент запроса»,
                  // а не «так написано в документе».
                  return (
                    <li key={fragment.number}>
                      {fragment.live ? (
                        <div className="flex items-start gap-2 rounded-lg bg-amber-50/70 px-2.5 py-2 ring-1 ring-amber-200">
                          {inside}
                        </div>
                      ) : (
                        <Link
                          to={`/wiki/${encodeURIComponent(fragment.doc_id)}#chunk-${fragment.chunk_id}`}
                          className="flex items-start gap-2 rounded-lg bg-white px-2.5 py-2 ring-1 ring-slate-200 transition hover:ring-slate-300"
                        >
                          {inside}
                        </Link>
                      )}
                    </li>
                  )
                })}
              </ul>
            </div>
          ) : null}

          {meta?.agent ? (
            <Disclosure summary="что делал агент">
              <AgentSteps agent={meta.agent} />
            </Disclosure>
          ) : null}

          {meta ? (
            <Disclosure summary="как получен ответ">
              <dl className="space-y-1">
                <Field label="лучшая близость">
                  {similarity(meta.retrieval.best_cosine)} при пороге{' '}
                  {similarity(meta.retrieval.floor)}{' '}
                  {meta.retrieval.passed_floor ? '(прошло)' : '(ниже порога)'}
                </Field>
                <Field label="найдено фрагментов">
                  {meta.retrieval.hits_total}, дублей убрано {meta.retrieval.duplicates_dropped}
                </Field>
                <Field label="отвечала модель">
                  <span className="inline-flex items-center gap-1.5">
                    {local ? <IconCpu className="size-3.5" /> : <IconCloud className="size-3.5" />}
                    {model || '—'} · провайдер {provider || '—'}
                  </span>
                  {/* Расхождение показываем прямо здесь: подмена ответчика
                      молча — это то, из-за чего потом не сходятся замеры. */}
                  {meta.model && model && meta.model !== model ? (
                    <span className="mt-0.5 block text-amber-700">
                      выбирали {meta.model}, но ответил резерв
                    </span>
                  ) : null}
                </Field>
                <Field label="эмбеддинги">{meta.embed_model}</Field>
                <Field label="версия промпта">{meta.prompt_version}</Field>
                {/* Чем НА САМОМ ДЕЛЕ искали. Показываем, только если вопрос
                    переписали с опорой на переписку: иначе строка «искали
                    тем же, что вы написали» была бы шумом на каждом
                    первом вопросе. А вот когда переписали — без неё
                    «нашлось не то» не разобрать: человек видит свой
                    вопрос, а поиск шёл по другому. */}
                {meta.history?.search_question ? (
                  <Field label="искали по">
                    «{meta.history.search_question}»
                    <span className="mt-0.5 block text-slate-400">
                      вопрос дополнен по переписке ({meta.history.turns} реплик)
                    </span>
                  </Field>
                ) : null}
                {done ? (
                  <>
                    <Field label="токены">
                      {tokens(done.usage.prompt_tokens, done.usage.completion_tokens)}
                    </Field>
                    <Field label="стоимость">{cost(done.usage.cost_rub, local)}</Field>
                    <Field label="до первого токена">{ms(done.timing.ttft_ms)}</Field>
                    <Field label="причина остановки">
                      {done.finish_reason} / {done.stop_reason}
                    </Field>
                    <Field label="трейс">{done.trace_id}</Field>
                  </>
                ) : null}
              </dl>

              {meta.fragments.length > 0 ? (
                <ol className="mt-3 space-y-2">
                  {meta.fragments.map((fragment) => (
                    <li
                      key={fragment.number}
                      className="rounded-lg border border-slate-200 bg-white p-2"
                    >
                      <p className="text-xs font-medium text-slate-700">
                        [{fragment.number}] {fragment.source_label}
                      </p>
                      <p className="mt-0.5 text-xs text-slate-500">
                        найден: {fragment.found_by} · косинус {similarity(fragment.cosine)} · RRF{' '}
                        {fragment.rrf.toFixed(4)}
                      </p>
                      <p className="mt-1 line-clamp-4 text-xs text-slate-700">{fragment.body}</p>
                    </li>
                  ))}
                </ol>
              ) : null}
            </Disclosure>
          ) : null}
        </div>
      </div>
    </article>
  )
}
