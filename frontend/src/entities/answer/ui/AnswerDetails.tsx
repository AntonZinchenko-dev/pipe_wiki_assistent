import { Link } from 'react-router-dom'
import { Field } from '@/shared/ui/primitives'
import { IconCloud, IconCpu, IconDoc } from '@/shared/ui/icons'
import { cost, ms, similarity, tokens } from '@/shared/lib/format'
import { cn } from '@/shared/lib/cn'
import type { Exchange } from '../model/exchange'

/**
 * Разбор ответа: откуда он взялся.
 *
 * Всё это раньше стояло прямо в ленте под каждым ответом — цитаты,
 * источники, шаги агента, замеры поиска и полные тексты всех найденных
 * фрагментов. Полезно и нужно, но нужно РЕДКО, а места занимало больше,
 * чем сам ответ: два вопроса подряд — и переписка перестаёт читаться.
 *
 * Здесь оно целиком, в выдвижной панели. В ленте остаётся то, ради чего
 * человек пришёл: вопрос, таблица, ответ — и предупреждения, если с
 * ответом что-то не так. Предупреждения НЕ прячем: то, что человек должен
 * увидеть не глядя, не кладут за кнопку.
 */

const TOOL_NAMES: Record<string, string> = {
  search_wiki: 'поискал ещё раз',
  read_section: 'дочитал раздел',
  live_pipe: 'спросил сервис про трубу',
  live_fleet: 'спросил сервис про парк',
  live_inspections: 'спросил сервис про инспекции',
  live_wells: 'спросил сервис про скважины',
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2">
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">{title}</h3>
      {children}
    </section>
  )
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
      <div className="space-y-1 text-xs text-ink-soft">
        <p>
          {decided
            ? 'Агент посмотрел на найденное и решил, что инструменты не нужны.'
            : 'Агент не дошёл до решения.'}{' '}
          <span className="text-ink-faint">способ: {agent.mechanism}</span>
        </p>
        {agent.declined_with ? (
          <p className="rounded border border-line bg-sunken px-2 py-1 text-ink-soft">
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
            step.ok ? 'border-line bg-surface' : 'border-live/30 bg-live-soft',
          )}
        >
          <span className="font-medium text-ink">
            {step.number}. {TOOL_NAMES[step.tool] ?? step.tool}
          </span>
          <span className="text-ink-soft">
            {' '}
            {Object.entries(step.arguments)
              // С именем поля, а не одним значением. «0.8» в строке про
              // чтение раздела не объяснял ничего; «min_damage: 0.8»
              // сразу показал бы, что в вызов уехал чужой аргумент.
              .map(([name, value]) => `${name}: «${String(value)}»`)
              .join(', ')}
          </span>
          <span className="ml-1 text-ink-soft">
            → {step.added > 0 ? `+${step.added} фрагм.` : 'ничего нового'}
          </span>
        </div>
      ))}
      <p className="text-xs text-ink-faint">способ решения: {agent.mechanism}</p>
      {agent.trail_cut ? (
        <p className="text-xs text-live-ink">
          Шаги кончились на полпути: последний вызов ещё приносил новое.
        </p>
      ) : null}
    </div>
  )
}

export function AnswerDetails({ exchange }: { exchange: Exchange }) {
  const { answer, meta, done } = exchange

  const provider = done?.provider || meta?.provider || ''
  const model = done?.model || meta?.model || ''
  const local = provider === 'ollama'

  const byNumber = new Map((meta?.fragments ?? []).map((fragment) => [fragment.number, fragment]))
  const citedNumbers = new Set((answer?.citations ?? []).map((citation) => citation.fragment))
  const sources = (meta?.fragments ?? []).filter((fragment) => citedNumbers.has(fragment.number))

  return (
    <div className="space-y-6">
      {sources.length > 0 ? (
        <Section title="Источники">
          <ul className="space-y-1.5">
            {sources.map((fragment) => {
              const inside = (
                <>
                  {fragment.live ? (
                    <IconCloud className="mt-0.5 size-4 shrink-0 text-live" />
                  ) : (
                    <IconDoc className="mt-0.5 size-4 shrink-0 text-ink-faint" />
                  )}
                  <span className="min-w-0">
                    <span className="block text-xs font-medium text-ink">
                      [{fragment.number}] {fragment.doc_title}
                    </span>
                    <span className="block text-[11px] text-ink-faint">
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
                    <div className="flex items-start gap-2 rounded-lg bg-live-soft px-2.5 py-2 ring-1 ring-live/30">
                      {inside}
                    </div>
                  ) : (
                    <Link
                      to={`/wiki/${encodeURIComponent(fragment.doc_id)}#chunk-${fragment.chunk_id}`}
                      className="flex items-start gap-2 rounded-lg bg-surface px-2.5 py-2 ring-1 ring-line transition hover:ring-accent-line"
                    >
                      {inside}
                    </Link>
                  )}
                </li>
              )
            })}
          </ul>
        </Section>
      ) : null}

      {answer && (!answer.schema_valid || answer.citations_dropped > 0) ? (
        <Section title="Что заметила проверка">
          <div className="space-y-1.5 text-xs">
            {answer.schema_error ? (
              <p className="rounded-lg border border-warn/40 bg-warn-soft px-2.5 py-1.5 text-warn-ink">
                {answer.schema_error}
              </p>
            ) : null}
            {/* Снятые ссылки — отдельная строка, а не молчание.
                Модель иногда пишет «сведений нет» и тут же ссылается на
                фрагмент. Подтверждать у отказа нечего, ссылки до экрана не
                доходят — но то, что они были, говорит о противоречии
                внутри ответа, и это стоит знать. */}
            {answer.citations_dropped > 0 ? (
              <p className="text-ink-soft">
                снято ссылок с отказа: {answer.citations_dropped}
              </p>
            ) : null}
          </div>
        </Section>
      ) : null}

      {answer && answer.citations.length > 0 ? (
        <Section title="Что подтверждено дословно">
          <ul className="space-y-1.5">
            {answer.citations.map((citation, index) => (
              <li
                key={`${citation.fragment}-${index}`}
                // ПОДТВЕРЖДЁННАЯ ЦИТАТА — НЕ СОБЫТИЕ.
                //
                // Зелёным она была логична и неверна: подтверждаются почти
                // все, и под каждым ответом висела стопка зелёных плашек.
                // Зелёное «всё хорошо», повторённое четыре раза, не
                // сообщает ничего, зато рядом теряется единственное красное
                // «не подтверждена» — то самое, ради которого проверка и
                // делалась. Норма молчит, отклонение кричит.
                className={cn(
                  'rounded-lg border px-2.5 py-1.5 text-xs',
                  citation.ok
                    ? 'border-line bg-surface text-ink-soft'
                    : 'border-bad/30 bg-bad-soft text-bad-ink',
                )}
              >
                <span className={cn('font-semibold', citation.ok && 'text-ink')}>
                  [{citation.fragment}]
                </span>{' '}
                {/* «Из документа» про строку живой таблицы — неправда:
                    документа с этим текстом не существует, есть снимок
                    сервиса, сделанный минуту назад. Мелочь, но из таких
                    мелочей складывается доверие к блоку источников. */}
                {citation.ok
                  ? byNumber.get(citation.fragment)?.live
                    ? 'из данных сервиса'
                    : 'из документа'
                  : `не подтверждена — ${citation.reason}`}
                {/*
                  Показываем только то, что вырезано из документа. При
                  неудачной опоре цитаты нет вообще — и это правильнее, чем
                  показать человеку текст, который сочинила модель: он
                  выглядит ровно как настоящий, и отличить их на экране
                  нельзя.
                */}
                {citation.quote ? (
                  <span className="mt-1 block border-l-2 border-line pl-2 text-ink">
                    {citation.quote}
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
        </Section>
      ) : null}

      {meta?.agent ? (
        <Section title="Что делал агент">
          <AgentSteps agent={meta.agent} />
        </Section>
      ) : null}

      {meta ? (
        <Section title="Как получен ответ">
          <dl className="space-y-1">
            <Field label="лучшая близость">
              {similarity(meta.retrieval.best_cosine)} при пороге {similarity(meta.retrieval.floor)}{' '}
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
                <span className="mt-0.5 block text-live-ink">
                  выбирали {meta.model}, но ответил резерв
                </span>
              ) : null}
            </Field>
            <Field label="эмбеддинги">{meta.embed_model}</Field>
            <Field label="версия промпта">{meta.prompt_version}</Field>
            {/* Чем НА САМОМ ДЕЛЕ искали. Показываем, только если вопрос
                переписали с опорой на переписку: иначе строка «искали тем
                же, что вы написали» была бы шумом на каждом первом
                вопросе. А вот когда переписали — без неё «нашлось не то»
                не разобрать: человек видит свой вопрос, а поиск шёл по
                другому. */}
            {meta.history?.search_question ? (
              <Field label="искали ещё и по">
                «{meta.history.search_question}»
                <span className="mt-0.5 block text-ink-faint">
                  вопрос сам по себе непонятен, поэтому вторым запросом ушла
                  склейка с прошлым ({meta.history.turns} реплик в памяти)
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
                <Field label="трейс">
                  <code className="font-mono text-[11px]">{done.trace_id}</code>
                </Field>
              </>
            ) : null}
          </dl>
        </Section>
      ) : null}

      {meta && meta.fragments.length > 0 ? (
        <Section title={`Весь контекст — ${meta.fragments.length} фрагм.`}>
          <ol className="space-y-2">
            {meta.fragments.map((fragment) => (
              <li key={fragment.number} className="rounded-lg border border-line bg-surface p-2.5">
                <p className="text-xs font-medium text-ink">
                  [{fragment.number}] {fragment.source_label}
                </p>
                <p className="mt-0.5 text-[11px] text-ink-faint">
                  найден: {fragment.found_by} · косинус {similarity(fragment.cosine)} · RRF{' '}
                  {fragment.rrf.toFixed(4)}
                </p>
                <p className="mt-1.5 line-clamp-4 text-xs leading-relaxed text-ink-soft">
                  {fragment.body}
                </p>
              </li>
            ))}
          </ol>
        </Section>
      ) : null}
    </div>
  )
}
