import { Link, useSearchParams } from 'react-router-dom'
import { DocumentRow, useDocuments } from '@/entities/document'
import { Spinner } from '@/shared/ui/primitives'
import { IconClose } from '@/shared/ui/icons'

/**
 * Список документов вики.
 *
 * Фильтры живут в адресе страницы, а не в состоянии компонента. Это значит,
 * что отфильтрованный список можно послать коллеге ссылкой и что кнопка
 * «назад» работает так, как человек ожидает. Состояние внутри компонента
 * выглядит проще ровно до первого «скинь, что ты там нашёл».
 */
export function WikiPage() {
  const { data, isLoading, error } = useDocuments()
  const [params] = useSearchParams()
  const project = params.get('project') ?? ''
  const query = (params.get('q') ?? '').trim().toLowerCase()

  if (isLoading) return <Spinner label="загружаю список документов" />
  if (error) {
    return (
      <p className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
        Не удалось загрузить список: {(error as Error).message}
      </p>
    )
  }

  const all = data ?? []
  const filtered = all.filter((document) => {
    if (project && document.project !== project) return false
    if (!query) return true
    return (
      document.title.toLowerCase().includes(query) ||
      document.doc_id.toLowerCase().includes(query)
    )
  })

  const byProject = new Map<string, typeof filtered>()
  for (const document of filtered) {
    const group = byProject.get(document.project) ?? []
    group.push(document)
    byProject.set(document.project, group)
  }

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold text-slate-900">
          {project || 'Документы вики'}
        </h1>
        <p className="text-sm text-slate-600">
          {filtered.length} из {all.length} документов
          {query ? ` по запросу «${params.get('q')}»` : ''}
        </p>
        {project || query ? (
          <Link
            to="/wiki"
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-600 transition hover:bg-slate-50"
          >
            <IconClose className="size-3.5" />
            сбросить фильтр
          </Link>
        ) : null}
      </header>

      {filtered.length === 0 ? (
        <p className="rounded-lg border border-dashed border-slate-300 p-6 text-sm text-slate-500">
          Ничего не нашлось. Обычный поиск ищет по названию и коду документа — если нужен
          поиск по смыслу, спросите ассистента справа.
        </p>
      ) : null}

      {[...byProject.entries()].map(([group, documents]) => (
        <section key={group} className="space-y-2">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            {group}
          </h2>
          <div className="space-y-1.5">
            {documents.map((document) => (
              <DocumentRow key={document.doc_id} document={document} />
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}
