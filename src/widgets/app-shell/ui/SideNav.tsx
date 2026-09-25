import { NavLink, useSearchParams } from 'react-router-dom'
import { useDocuments, useHealth } from '@/entities/document'
import { useThreads } from '@/entities/thread'
import { useViewer } from '@/entities/viewer'
import { IconDoc, IconFolder, IconSparkle } from '@/shared/ui/icons'
import { GroupLabel } from '@/shared/ui/primitives'
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

/** Одинаковая строка меню для ссылок и для пространств. */
const ROW = 'flex items-center gap-2.5 rounded-lg px-2.5 py-[7px] text-[13.5px] transition'
const ROW_ACTIVE = 'bg-accent-soft font-medium text-accent-ink'
const ROW_IDLE = 'text-ink-soft hover:bg-sunken hover:text-ink'

export function SideNav({ onAskAssistant }: { onAskAssistant: () => void }) {
  const { data: documents } = useDocuments()
  const { data: viewer } = useViewer()
  const { data: health } = useHealth()
  const [params] = useSearchParams()
  const select = useThreads((state) => state.select)
  const activeProject = params.get('project') ?? ''

  const counts = new Map<string, number>()
  for (const document of documents ?? []) {
    counts.set(document.project, (counts.get(document.project) ?? 0) + 1)
  }
  const projects = [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0], 'ru'))

  const chunks = health?.index.chunks ?? 0

  return (
    <aside className="hidden w-60 shrink-0 flex-col border-r border-line bg-chrome lg:flex">
      {/* Главное действие — наверху и одно.
          Заметная кнопка в меню означает «вот с чего начинают», и в этом
          приложении начинают с вопроса, а не с листания документов. */}
      <div className="p-3 pb-1">
        <button
          type="button"
          onClick={() => {
            select(null)
            onAskAssistant()
          }}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white shadow-card transition hover:bg-accent-hover"
        >
          <IconSparkle className="size-4" />
          Новый вопрос
        </button>
      </div>

      <nav className="flex flex-col gap-0.5 px-3 pt-2">
        {LINKS.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            end={link.end}
            className={({ isActive }) =>
              cn(ROW, isActive && !activeProject ? ROW_ACTIVE : ROW_IDLE)
            }
          >
            <link.icon className="size-4 shrink-0" />
            {link.label}
          </NavLink>
        ))}
      </nav>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        <ThreadList />

        <GroupLabel>Пространства</GroupLabel>
        <div className="flex flex-col gap-0.5">
          {projects.map(([project, count]) => (
            <NavLink
              key={project}
              to={`/wiki?project=${encodeURIComponent(project)}`}
              className={cn(ROW, activeProject === project ? ROW_ACTIVE : ROW_IDLE)}
            >
              <IconFolder className="size-4 shrink-0 text-ink-faint" />
              <span className="min-w-0 flex-1 truncate">{project}</span>
              <span className="shrink-0 text-[11px] text-ink-faint">{count}</span>
            </NavLink>
          ))}
          {projects.length === 0 ? (
            <p className="px-2.5 py-2 text-xs text-ink-faint">пока пусто</p>
          ) : null}
        </div>
      </div>

      {/* Состояние индекса внизу, а не в логах: пустой индекс — самая частая
          причина «ассистент ничего не знает», и она должна быть видна. */}
      <div className="border-t border-line px-3 py-3 text-xs">
        {viewer ? (
          <div className="mb-2.5 flex items-center gap-2.5 px-1">
            {/* Инициал вместо картинки: аватары взять неоткуда, а пустой
                круг-заглушка выглядит как несработавшая загрузка. */}
            <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-accent-soft text-[11px] font-semibold text-accent-ink">
              {viewer.name.slice(0, 1).toUpperCase()}
            </span>
            <span className="min-w-0">
              <span className="block truncate font-medium text-ink">{viewer.name}</span>
              <span className="block truncate text-ink-faint">
                {viewer.roles.length ? viewer.roles.join(', ') : 'без ролей'}
              </span>
            </span>
          </div>
        ) : null}

        <div className="space-y-1 px-1">
          {/* Сколько проектов закрыто — но НЕ какие. Перечень закрытого
              сам по себе сведение о том, что в системе есть. А знать,
              что поиск идёт не по всему корпусу, человеку нужно: иначе
              «в документации нет ответа» читается как отсутствие
              документа, а не как отсутствие доступа. */}
          {viewer && viewer.closed_projects > 0 ? (
            // Спокойным тоном, а не жёлтым.
            //
            // Это постоянное состояние, а не событие: у человека просто
            // нет доступа к части проектов, и так будет всегда. Яркая
            // строчка, которая горит в углу каждый день, через неделю
            // перестаёт читаться — и заодно приучает не замечать жёлтый
            // вообще, в том числе там, где он значит «сломалось».
            <p className="text-ink-faint">
              закрыто проектов: <span className="text-ink-soft">{viewer.closed_projects}</span>
            </p>
          ) : null}

          {chunks > 0 ? (
            <p className="text-ink-faint">
              индекс: {health?.index.documents} док. · {chunks} фрагм.
            </p>
          ) : (
            <p className="font-medium text-bad-ink">
              индекс пуст — выполните <code className="font-mono text-[11px]">scripts/ingest.py</code>
            </p>
          )}

          {/* Где лежит переписка. Человек имеет право знать, что диалоги
              живут в браузере: на другом компьютере их не будет, а очистка
              данных сайта сотрёт их без предупреждения. */}
          <p className="text-ink-faint">диалоги хранятся в этом браузере</p>
        </div>
      </div>
    </aside>
  )
}
