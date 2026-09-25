import type { ReactNode } from 'react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { AnswerStatus } from '@/shared/api/contracts'
import { Badge, Spinner } from '@/shared/ui/primitives'
import { Drawer } from '@/shared/ui/Drawer'
import { IconInfo, IconSparkle } from '@/shared/ui/icons'
import { parseAnswer, type Inline } from '@/shared/lib/markdown'
import { cn } from '@/shared/lib/cn'
import { displayText, type Exchange } from '../model/exchange'
import { AnswerDetails } from './AnswerDetails'
import { DataTable } from './DataTable'

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

/**
 * Ярлык статуса. Обычный ответ — НЕЙТРАЛЬНЫМ тоном.
 *
 * «Ответ найден» зелёным висело над каждым удачным ответом, то есть почти
 * всегда. Цвет, который горит при норме, не сообщает о норме — он просто
 * приучает глаз его не видеть, и вместе с ним перестают замечать жёлтое
 * «в документации нет ответа». Здесь цветом отмечены только те четыре
 * состояния, в которых с ответом что-то не так.
 */
const STATUS_VIEW: Record<
  AnswerStatus,
  { label: string; tone: 'neutral' | 'good' | 'warn' | 'bad' | 'info' }
> =
  {
    answered: { label: 'ответ найден', tone: 'neutral' },
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
  // Отступ ТОЛЬКО СЛЕВА.
  //
  // Симметричный отступ отрывал ссылку от следующей за ней точки, и в
  // тексте появлялось «выработано 97.7 % ресурса [1] . Это выше» — пробел
  // перед точкой. Мелочь ровно до того момента, пока её не увидишь: после
  // этого она видна в каждом предложении.
  const badge =
    'ml-0.5 rounded bg-accent-soft px-1 align-baseline text-[11px] font-semibold text-accent-ink'
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
                className="rounded bg-sunken px-1 py-0.5 font-mono text-[13px] text-ink"
              >
                {node.text}
              </code>
            )
          case 'citation': {
            const href = target(node.number)
            const label = `[${node.number}]`
            return href ? (
              <Link key={index} to={href} className={cn(badge, 'hover:bg-accent-soft hover:underline')}>
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
    <div className="space-y-2 text-[15px] leading-relaxed text-ink">
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

export function AnswerCard({ exchange }: { exchange: Exchange }) {
  const [details, setDetails] = useState(false)
  const { phase, answer, meta, done, error } = exchange
  const text = displayText(exchange)
  const status = answer?.status
  const view = status ? STATUS_VIEW[status] : null

  // Подпись к кнопке разбора: сколько источников и сколько шагов сделал
  // агент. Числа стоят на кнопке, а не в ленте отдельными плашками —
  // «2 источника» это не событие, о котором надо сообщать строкой.
  const citedNumbers = new Set((answer?.citations ?? []).map((citation) => citation.fragment))
  const steps = meta?.agent?.used_tools ? meta.agent.steps.length : 0
  const summary = [
    citedNumbers.size ? `${citedNumbers.size} источн.` : '',
    steps ? `агент: ${steps}` : '',
  ]
    .filter(Boolean)
    .join(' · ')

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
        <p className="max-w-[85%] rounded-2xl rounded-br-md border border-line bg-surface px-3.5 py-2 text-[14.5px] leading-relaxed text-ink shadow-card">
          {exchange.question}
        </p>
      </div>

      <div className="flex gap-2.5">
        <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-accent-soft text-accent">
          <IconSparkle className="size-4" />
        </span>

        <div className="min-w-0 flex-1 space-y-2.5">
          <Row>
            {view ? <Badge tone={view.tone}>{view.label}</Badge> : null}
            {phase === 'streaming' ? (
              <Spinner label={exchange.step || 'модель отвечает'} />
            ) : null}
            {phase === 'stopped' ? <Badge tone="neutral">остановлено вами</Badge> : null}
            {done?.finish_reason === 'length' ? (
              <Badge tone="warn">обрезано по лимиту токенов</Badge>
            ) : null}
            {/* Текст замечания уехал в разбор.
                «схема: статус «ответил» без единой цитаты — исправлен на
                not_found» — это записка разработчика самому себе. В ленте
                человеку нужен факт «проверка что-то поправила», а сама
                формулировка — там же, где остальные подробности. */}
            {answer && !answer.schema_valid ? (
              <Badge tone="warn">есть замечания проверки</Badge>
            ) : null}
            {answer && answer.citations_failed > 0 ? (
              <Badge tone="bad">ссылок не подтверждено: {answer.citations_failed}</Badge>
            ) : null}
            {/* Плашка «агент: N шагов» отсюда ушла на кнопку разбора.
                Это описание того, как всё прошло, а не событие; в ленте
                остаются только ярлык состояния и то, с чем что-то не так. */}
            {exchange.datasets.length === 0 &&
            meta?.fragments.some((fragment) => fragment.live) ? (
              <Badge tone="live">есть живые данные сервиса</Badge>
            ) : null}
          </Row>

          {/* Ошибка — это состояние, а не текст поверх ответа: показываем её
              отдельно и НЕ стираем то, что человек уже прочитал. */}
          {error ? (
            <div className="rounded-lg border border-bad/30 bg-bad-soft p-3 text-sm text-bad-ink">
              <p className="font-medium">{error.message}</p>
              {error.detail ? <p className="mt-1 text-xs opacity-80">{error.detail}</p> : null}
              {error.retry_after_s ? (
                <p className="mt-1 text-xs">Повторить можно через {error.retry_after_s} с.</p>
              ) : null}
            </div>
          ) : null}

          {/* Таблицы ПЕРЕД текстом, и порядок здесь содержательный.
              Человек спросил «дай топ 5 труб» — ему нужны трубы, а не
              рассуждение о них. Данные приходят раньше ответа и по
              времени: пока модель думает, таблица уже на экране. */}
          {exchange.datasets.map((data) => (
            <DataTable key={data.handle} data={data} />
          ))}

          {text ? <AnswerText text={text} target={linkTo} /> : null}

          {/* Выдуманные идентификаторы — красным и над источниками.
              Это не придирка к оформлению: «PP-0029, выработка 72%» в
              ответе про бурильный инструмент читается как факт, а такой
              трубы в парке нет. Проверка цитат её пропустила — опор
              всего четыре, а строк в списке десять. */}
          {answer && answer.mismatched_refs?.length ? (
            <p className="rounded-lg border border-bad/30 bg-bad-soft p-3 text-sm text-bad-ink">
              Числа рядом с этими обозначениями не совпадают с источником:{' '}
              <span className="font-semibold">{answer.mismatched_refs.join(', ')}</span>. Сверьте
              их по блоку источников ниже — там значения такие, какими их отдал сервис.
            </p>
          ) : null}

          {answer && answer.unknown_refs?.length ? (
            <p className="rounded-lg border border-bad/30 bg-bad-soft p-3 text-sm text-bad-ink">
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
            <p className="rounded-lg border border-accent-line bg-accent-soft p-3 text-sm text-accent-ink">
              {answer.clarifying_question}
            </p>
          ) : null}

          {/* ОДНА КНОПКА ВМЕСТО ДВУХСОТ СТРОК.
              Цитаты, источники, шаги агента, замеры поиска и полные тексты
              всех найденных фрагментов стояли прямо здесь, под каждым
              ответом. Всё это нужно — и нужно редко, а места занимало
              больше, чем сам ответ: после двух вопросов до нужного места в
              переписке было не докрутить. */}
          {meta || (answer && answer.citations.length > 0) ? (
            <div>
              <button
                type="button"
                onClick={() => setDetails(true)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 py-1 text-[11px] text-ink-soft transition hover:border-accent-line hover:text-ink"
              >
                <IconInfo className="size-3.5" />
                разбор
                {summary ? <span className="text-ink-faint">· {summary}</span> : null}
              </button>
            </div>
          ) : null}

          <Drawer
            open={details}
            title={`Разбор: «${exchange.question}»`}
            onClose={() => setDetails(false)}
          >
            <AnswerDetails exchange={exchange} />
          </Drawer>
        </div>
      </div>
    </article>
  )
}
