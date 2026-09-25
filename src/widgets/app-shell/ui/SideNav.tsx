import { NavLink, useSearchParams } from 'react-router-dom'
import { useDocuments, useHealth } from '@/entities/document'
import { useViewer } from '@/entities/viewer'
import { IconDoc, IconFolder } from '@/shared/ui/icons'
import { cn } from '@/shared/lib/cn'
import { ThreadList } from './ThreadList'

/**
 * Левое меню.
 *
 * «Пространства» здесь — НЕ выдуманные разделы под макет, а проекты из
 * самих документов: то поле, по которому вики уже сгруппирована. Придумать
 * красивые пункты меню было бы быстрее, но меню, которое ведёт не туда, где
 * лежат данные, — это вторая навигация поверх первой, и они расходятся на
 * первой же новой папке.
 */

// Пункта «Ассистент» здесь больше нет, и это не упрощение меню.
//
// Он вёл на отдельный раздел, который рисовал ту же переписку, что и
// панель справа, — на широком экране получалось два окна в один диалог.
// У ассистента одно место: панель, которая разворачивается на весь экран
// кнопкой в своей же шапке.
const LINKS = [{ to: '/wiki', label: 'Все документы', icon: IconDoc, end: true }]

export function SideNav() {
  const { data: documents } = useDocuments()
  const { data: viewer } = useViewer()
  const { data: health } = useHealth()
  const [params] = useSearchParams()
  const activeProject = params.get('project') ?? ''

  const counts = new Map<string, number>()
  for (const document of documents ?? []) {
    counts.set(document.project, (counts.get(document.project) ?? 0) + 1)
  }
  const projects = [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0], 'ru'))

  const chunks = health?.index.chunks ?? 0

  return (
    <aside className="hidden w-60 shrink-0 flex-col border-r border-slate-200 bg-white lg:flex">
      <nav className="flex flex-col gap-0.5 p-3">
        {LINKS.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            end={link.end}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition',
                isActive && !activeProject
                  ? 'bg-sky-50 font-medium text-sky-700'
                  : 'text-slate-600 hover:bg-slate-100',
              )
            }
          >
            <link.icon className="size-4 shrink-0" />
            {link.label}
          </NavLink>
        ))}
      </nav>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        <ThreadList />

        <p className="px-2.5 pb-1 pt-4 text-xs font-semibold uppercase tracking-wide text-slate-400">
          Пространства
        </p>
        <div className="flex flex-col gap-0.5">
          {projects.map(([project, count]) => (
            <NavLink
              key={project}
              to={`/wiki?project=${encodeURIComponent(project)}`}
              className={cn(
                'flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition',
                activeProject === project
                  ? 'bg-sky-50 font-medium text-sky-700'
                  : 'text-slate-600 hover:bg-slate-100',
              )}
            >
              <IconFolder className="size-4 shrink-0 text-slate-400" />
              <span className="min-w-0 flex-1 truncate">{project}</span>
              <span className="shrink-0 text-xs text-slate-400">{count}</span>
            </NavLink>
          ))}
          {projects.length === 0 ? (
            <p className="px-2.5 py-2 text-xs text-slate-400">пока пусто</p>
          ) : null}
        </div>
      </div>

      {/* Состояние индекса внизу, а не в логах: пустой индекс — самая частая
          причина «ассистент ничего не знает», и она должна быть видна. */}
      <div className="space-y-1.5 border-t border-slate-200 px-4 py-3 text-xs">
        {/* Где лежит переписка. Человек имеет право знать, что диалоги
            живут в браузере: на другом компьютере их не будет, а очистка
            данных сайта сотрёт их без предупреждения. */}
        <p className="text-slate-400">диалоги хранятся в этом браузере</p>

        {viewer ? (
          <div>
            <p className="font-medium text-slate-700">{viewer.name}</p>
            <p className="text-slate-500">
              {viewer.roles.length ? viewer.roles.join(', ') : 'без ролей'}
            </p>
            {/* Сколько проектов закрыто — но НЕ какие. Перечень закрытого
                сам по себе сведение о том, что в системе есть. А знать,
                что поиск идёт не по всему корпусу, человеку нужно: иначе
                «в документации нет ответа» читается как отсутствие
                документа, а не как отсутствие доступа. */}
            {viewer.closed_projects > 0 ? (
              <p className="text-amber-700">
                закрыто проектов: {viewer.closed_projects}
              </p>
            ) : null}
          </div>
        ) : null}

        {chunks > 0 ? (
          <p className="text-slate-500">
            индекс: {health?.index.documents} док. · {chunks} фрагм.
          </p>
        ) : (
          <p className="font-medium text-red-700">
            индекс пуст — выполните <code className="text-[11px]">scripts/ingest.py</code>
          </p>
        )}
      </div>
    </aside>
  )
}
