import { useState } from 'react'
import type { Dataset } from '@/shared/api/contracts'
import { datasetFileName, datasetToCsv } from '@/shared/lib/csv'
import { IconCheck, IconCopy, IconDownload } from '@/shared/ui/icons'

/**
 * Таблица из инструмента — данные, которых модель не видела.
 *
 * ЭТО САМОЕ НАДЁЖНОЕ МЕСТО НА ЭКРАНЕ, и стоит понимать почему. Всё
 * остальное в карточке — текст, который модель сгенерировала: его
 * проверяют цитатами, сверяют обозначения, ищут подменённые числа. Здесь
 * проверять нечего: строки пришли из сервиса и дошли до экрана, ни разу
 * не пройдя через генерацию.
 *
 * Раньше их пересказывала модель, и это стоило нам трёх разных поломок на
 * трёх разных моделях: «PP-0001, выработка 7%» вместо 77.0, выдуманная
 * труба PP-0029, оборванный на середине список. Все три исчезли не
 * потому, что мы научили модель аккуратности, а потому, что перестали
 * просить её делать механическую работу.
 *
 * `total_found` рядом с числом строк — не украшение. «Нашлось 40,
 * показано 5» первым терялось в пересказе: «топ-5» читалось как «в парке
 * пять труб», и человек делал вывод по выборке, приняв её за весь парк.
 */

/**
 * Разряды разделяем, точность НЕ дорисовываем.
 *
 * «1 990 578» читается, «1990578» — нет. А вот дописать «87» до «87.0»
 * нельзя: это данные о состоянии бурильного инструмента, и лишний знак
 * после запятой — обещание точности, которой в исходном числе не было.
 */
function show(value: unknown): string {
  if (typeof value === 'number') {
    return value.toLocaleString('ru-RU', { maximumFractionDigits: 3 })
  }
  return String(value)
}

function Cell({ value, kind }: { value: unknown; kind: string }) {
  const empty = value === null || value === undefined || value === ''
  return (
    <td
      className={
        kind === 'number'
          ? 'whitespace-nowrap px-2 py-1 text-right font-mono text-[13px] tabular-nums text-ink'
          : 'whitespace-nowrap px-2 py-1 text-ink'
      }
    >
      {empty ? <span className="text-ink-faint">—</span> : show(value)}
    </td>
  )
}

function Failure({ data }: { data: Dataset }) {
  const error = data.error
  if (!error) return null
  return (
    <div className="rounded-lg border border-live/30 bg-live-soft p-3 text-sm">
      <p className="font-medium text-live-ink">{data.title}: запрос к сервису не удался</p>
      <p className="mt-1 text-live-ink">
        {error.message} <span className="text-live-ink">({error.code})</span>
      </p>
      {error.hint ? <p className="mt-1 text-live-ink">{error.hint}</p> : null}
      {/* Повторять или нет — решает сервер, а не догадка интерфейса.
          «Расчёт устарел, через минуту» и «такой трубы нет» выглядели
          одинаково, пока признак не поехал отдельным полем. */}
      <p className="mt-1.5 text-xs text-live-ink">
        {error.retriable
          ? `Отказ временный${
              error.retry_after_s ? `, повторить можно через ${error.retry_after_s} с` : ''
            }.`
          : 'Отказ постоянный: повтор того же запроса даст тот же результат.'}
      </p>
    </div>
  )
}

/**
 * Забрать таблицу с собой.
 *
 * Не украшение и не «ещё одна кнопочка». Эти строки — единственное на
 * экране, что не проходило через генерацию, и работать с ними человек
 * будет дальше: отсортирует, сравнит с прошлой выгрузкой, приложит к
 * заявке. Без выгрузки он перепечатает их руками — то есть вернёт ровно ту
 * ошибку переписывания, ради ухода от которой таблица сюда и приехала.
 *
 * Отдаём файлом, а не открываем ссылкой: ссылка на blob ведёт в никуда
 * через минуту, и вкладка с ней после перезагрузки показывает пустоту.
 */
function TakeAway({ data }: { data: Dataset }) {
  const [copied, setCopied] = useState(false)

  function download() {
    const blob = new Blob([datasetToCsv(data)], {
      type: 'text/csv;charset=utf-8',
    })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = datasetFileName(data)
    link.click()
    // Ссылку освобождаем сразу: браузер держит blob в памяти, пока её не
    // отозвали, а таблиц за разговор набирается десяток.
    URL.revokeObjectURL(url)
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(datasetToCsv(data))
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Буфер обмена закрыт настройками браузера или страница без
      // защищённого соединения. Не повод падать: выгрузка файлом рядом и
      // работает всегда.
    }
  }

  // ЗНАЧКИ, А НЕ СЛОВА.
  //
  // «копировать» и «CSV» занимали девяносто пикселей той же строки, где
  // стоит заголовок, и заголовок обрезался многоточием в панели шириной
  // 440. Название таблицы человеку нужнее, чем подписи к двум кнопкам,
  // смысл которых виден по значку и написан во всплывающей подсказке.
  const style =
    'flex size-6 items-center justify-center rounded-md text-ink-faint transition hover:bg-surface hover:text-ink'

  return (
    <span className="flex shrink-0 items-center gap-0.5">
      <button
        type="button"
        onClick={copy}
        className={style}
        title={copied ? 'скопировано' : 'копировать таблицу'}
        aria-label="копировать таблицу"
      >
        {copied ? <IconCheck className="size-3.5 text-good" /> : <IconCopy className="size-3.5" />}
      </button>
      <button
        type="button"
        onClick={download}
        className={style}
        title="скачать таблицу в CSV"
        aria-label="скачать таблицу в CSV"
      >
        <IconDownload className="size-3.5" />
      </button>
      <span className="ml-0.5 rounded bg-surface px-1.5 py-0.5 font-mono text-[11px] text-ink-faint">
        {data.handle}
      </span>
    </span>
  )
}

export function DataTable({ data }: { data: Dataset }) {
  if (data.error) return <Failure data={data} />
  if (!data.rows.length) {
    return (
      <div className="rounded-lg border border-line bg-sunken p-3 text-sm text-ink-soft">
        <p className="font-medium text-ink">{data.title}</p>
        <p className="mt-1">{data.hint || 'Сервис не вернул ни одной записи.'}</p>
      </div>
    )
  }

  return (
    <figure className="overflow-hidden rounded-lg border border-line bg-surface">
      <figcaption className="flex items-start gap-2 border-b border-line bg-sunken px-3 py-2">
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium text-ink">{data.title}</span>
          <span className="block truncate text-[11px] text-ink-faint">
            данные сервиса, снято {data.taken_at}
          </span>
        </span>
        <TakeAway data={data} />
      </figcaption>

      {/* Тень у правого края — единственный признак того, что таблица
          шире панели.
          В панели шириной 440 пикселей пять колонок не помещаются, и
          последняя обрывается ровно по границе — выглядит это не как
          «прокрути вбок», а как «колонку обрезало». Человек не прокручивает
          то, о чём не знает, и решает, что данных нет. */}
      <div className="relative">
        <div className="max-h-96 overflow-auto">
          <table className="w-full border-collapse text-sm">
            <thead className="sticky top-0 bg-surface">
              <tr className="border-b border-line text-left">
                {data.columns.map((column) => (
                  <th
                    key={column.key}
                    className={
                      column.kind === 'number'
                        ? 'px-2 py-1.5 text-right text-xs font-medium text-ink-soft'
                        : 'px-2 py-1.5 text-xs font-medium text-ink-soft'
                    }
                  >
                    {column.title}
                    {column.unit ? (
                      <span className="font-normal text-ink-faint">, {column.unit}</span>
                    ) : null}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row, index) => (
                <tr key={index} className="border-b border-line-soft last:border-0">
                  {data.columns.map((column) => (
                    <Cell key={column.key} value={row[column.key]} kind={column.kind} />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-surface to-transparent"
        />
      </div>

      <div className="space-y-0.5 border-t border-line bg-sunken px-3 py-2 text-xs text-ink-soft">
        <p>
          {/* Номера строк, а не только их число. «5 из 39» на второй
              странице не говорит, КАКИЕ пять, и читается как начало
              списка. */}
          {data.offset
            ? `строки ${data.offset + 1}–${data.offset + data.rows.length} из ${data.total_found}`
            : `строк: ${data.rows.length}` +
              (data.total_found > data.rows.length ? ` из ${data.total_found}` : '')}
          {data.scanned ? ` · просмотрено ${data.scanned}` : ''}
        </p>
        {data.hint ? <p className="text-ink-soft">{data.hint}</p> : null}
        {/* Пропущенные строки называем поимённо: «две трубы за порогом» и
            «две за порогом, по одной расчёт устарел» — разные ответы. */}
        {data.skipped.length ? (
          <p className="text-live-ink">нет данных по: {data.skipped.join(', ')}</p>
        ) : null}
      </div>
    </figure>
  )
}
