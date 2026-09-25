import { Link } from 'react-router-dom'
import type { DocumentSummary } from '@/shared/api/contracts'
import { Badge } from '@/shared/ui/primitives'

export function DocumentRow({ document }: { document: DocumentSummary }) {
  const archived = document.status !== 'действующий'
  return (
    <Link
      to={`/wiki/${encodeURIComponent(document.doc_id)}`}
      className="flex items-baseline justify-between gap-4 rounded-md border border-slate-200 px-3 py-2 hover:border-slate-400 hover:bg-slate-50"
    >
      <span className="min-w-0">
        <span className="block truncate text-sm font-medium text-slate-900">
          {document.title}
        </span>
        <span className="block text-xs text-slate-500">
          {document.doc_id} · {document.project} · ред. {document.version} · обновлён{' '}
          {document.updated || '—'}
        </span>
      </span>
      <span className="flex shrink-0 items-center gap-2">
        {archived ? <Badge tone="warn">{document.status}</Badge> : null}
        <Badge>{document.chunk_count} фрагм.</Badge>
      </span>
    </Link>
  )
}
