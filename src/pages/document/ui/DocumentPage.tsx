import { useEffect } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { useDocument } from '@/entities/document'
import { Badge, Spinner } from '@/shared/ui/primitives'
import { IconChevronRight, IconDoc } from '@/shared/ui/icons'
import { cn } from '@/shared/lib/cn'
import { pages } from '@/shared/lib/format'

/**
 * Документ в том виде, в котором его видит поиск — по фрагментам.
 *
 * Это сделано специально: человек, читающий ответ ассистента, должен иметь
 * возможность открыть ровно тот фрагмент, на который сослалась модель, и
 * увидеть его целиком. Красивый рендер PDF тут был бы хуже: он показывал бы
 * не то, что реально лежит в индексе.
 */
export function DocumentPage() {
  const { docId } = useParams<{ docId: string }>()
  const { data, isLoading, error } = useDocument(docId)
  const { hash } = useLocation()
  const target = hash.startsWith('#chunk-') ? hash.slice(1) : ''

  // Прокрутка к нужному фрагменту после того, как документ ПРИШЁЛ.
  //
  // Браузер сам к якорю не прокрутит: в момент перехода нужного элемента на
  // странице ещё нет, документ только загружается. Поэтому ждём данные и
  // зависим от них — иначе переход по ссылке [4] работал бы через раз, в
  // зависимости от того, лежит документ в кэше запросов или нет. Такие
  // «иногда не срабатывает» ловятся потом неделями.
  useEffect(() => {
    if (!target || !data) return
    document.getElementById(target)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [target, data])

  if (isLoading) return <Spinner label="загружаю документ" />
  if (error) {
    return (
      <p className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
        {(error as Error).message}
      </p>
    )
  }
  if (!data) return null

  return (
    <article className="space-y-5">
      <nav className="flex items-center gap-1 text-xs text-slate-500">
        <Link to="/wiki" className="hover:text-slate-800">
          Документы
        </Link>
        <IconChevronRight className="size-3.5" />
        <Link
          to={`/wiki?project=${encodeURIComponent(data.project)}`}
          className="hover:text-slate-800"
        >
          {data.project}
        </Link>
        <IconChevronRight className="size-3.5" />
        <span className="truncate text-slate-700">{data.doc_id}</span>
      </nav>

      <header className="space-y-3">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 flex size-10 shrink-0 items-center justify-center rounded-xl bg-sky-50 text-sky-600">
            <IconDoc className="size-5" />
          </span>
          <h1 className="text-2xl font-semibold leading-snug text-slate-900">{data.title}</h1>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <Badge>{data.doc_id}</Badge>
          <Badge>ред. {data.version}</Badge>
          <Badge tone={data.status === 'действующий' ? 'good' : 'warn'}>{data.status}</Badge>
          <Badge>обновлён {data.updated || '—'}</Badge>
          <Badge>{data.owner}</Badge>
        </div>
      </header>

      <ol className="space-y-2.5">
        {data.chunks.map((chunk) => (
          <li
            key={chunk.chunk_id}
            id={`chunk-${chunk.chunk_id}`}
            className={cn(
              'rounded-xl border bg-white p-4 transition',
              // Подсветка держится, пока в адресе стоит якорь, и не гаснет
              // по таймеру. Человек пришёл сюда проверить цитату: если
              // подсветка исчезнет, пока он читает, он потеряет место и
              // будет искать заново.
              target === `chunk-${chunk.chunk_id}`
                ? 'border-sky-300 ring-2 ring-sky-200'
                : 'border-slate-200',
            )}
          >
            <p className="text-xs font-medium text-slate-500">
              {chunk.heading_path} · {pages(chunk.page_from, chunk.page_to)}
              {target === `chunk-${chunk.chunk_id}` ? (
                <span className="ml-1.5 text-sky-700">· фрагмент из ответа</span>
              ) : null}
            </p>
            <p className="mt-1.5 whitespace-pre-wrap text-[15px] leading-relaxed text-slate-800">
              {chunk.body}
            </p>
          </li>
        ))}
      </ol>
    </article>
  )
}
