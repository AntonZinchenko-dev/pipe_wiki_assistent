import { Link } from 'react-router-dom'
import type { DocumentSummary } from '@/shared/api/contracts'
import { Badge } from '@/shared/ui/primitives'
import { IconDoc } from '@/shared/ui/icons'

/**
 * Строка документа в списке.
 *
 * Карточка, а не строка таблицы: у документа есть иконка, название и
 * подпись, и всё это читается сверху вниз одним взглядом. Таблица была бы
 * плотнее, но в ней название и метаданные выравниваются по одной высоте, и
 * глаз перестаёт различать, что здесь главное.
 */
export function DocumentRow({ document }: { document: DocumentSummary }) {
  const archived = document.status !== 'действующий'
  return (
    <Link
      to={`/wiki/${encodeURIComponent(document.doc_id)}`}
      className="group flex items-center gap-3 rounded-xl border border-line bg-surface px-3 py-2.5 shadow-card transition hover:border-accent-line hover:shadow-raised"
    >
      <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-sunken text-ink-faint transition group-hover:bg-accent-soft group-hover:text-accent">
        <IconDoc className="size-4" />
      </span>

      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium text-ink">{document.title}</span>
        <span className="block truncate text-xs text-ink-faint">
          {document.doc_id} · {document.project} · ред. {document.version} · обновлён{' '}
          {document.updated || '—'}
        </span>
      </span>

      <span className="flex shrink-0 items-center gap-1.5">
        {archived ? <Badge tone="warn">{document.status}</Badge> : null}
        <Badge>{document.chunk_count} фрагм.</Badge>
      </span>
    </Link>
  )
}
