import { useThreads } from '@/entities/thread'
import { IconChat, IconPlus, IconTrash } from '@/shared/ui/icons'
import { cn } from '@/shared/lib/cn'

/**
 * Список диалогов с ассистентом.
 *
 * Здесь только ВЫБОР диалога — ни загрузки ленты, ни отправки запроса.
 * Переключение состоит из одной строчки `select(id)`, а лента следует за
 * ним сама (см. `useThreadSync`). Это не разделение ради разделения:
 * меню перерисовывается редко, лента — на каждый токен потока, и держать
 * их в одном состоянии значило бы перерисовывать меню тридцать раз в
 * секунду во время ответа.
 *
 * Диалоги лежат в базе БРАУЗЕРА и на сервер не уезжают. Про это сказано
 * прямо внизу списка: человек должен знать, что переписка живёт на его
 * машине — и, как следствие, не переедет на другой компьютер и исчезнет
 * с очисткой данных сайта.
 */

const VISIBLE = 12

export function ThreadList() {
  const threads = useThreads((state) => state.threads)
  const currentId = useThreads((state) => state.currentId)
  const select = useThreads((state) => state.select)
  const remove = useThreads((state) => state.remove)

  return (
    <div>
      <div className="flex items-center justify-between gap-2 px-2.5 pb-1 pt-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Диалоги
        </p>
        <button
          type="button"
          onClick={() => select(null)}
          aria-label="новый диалог"
          title="Новый диалог"
          className="flex size-5 items-center justify-center rounded text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
        >
          <IconPlus className="size-3.5" />
        </button>
      </div>

      <div className="flex flex-col gap-0.5">
        {threads.slice(0, VISIBLE).map((thread) => (
          <div
            key={thread.id}
            className={cn(
              'group flex items-center gap-2 rounded-lg pr-1 transition',
              currentId === thread.id ? 'bg-sky-50' : 'hover:bg-slate-100',
            )}
          >
            <button
              type="button"
              onClick={() => select(thread.id)}
              className={cn(
                'flex min-w-0 flex-1 items-center gap-2.5 px-2.5 py-2 text-left text-sm',
                currentId === thread.id ? 'font-medium text-sky-700' : 'text-slate-600',
              )}
            >
              <IconChat className="size-4 shrink-0 text-slate-400" />
              <span className="min-w-0 flex-1 truncate">{thread.title}</span>
            </button>
            <button
              type="button"
              onClick={() => void remove(thread.id)}
              aria-label={`удалить диалог «${thread.title}»`}
              // Видна только при наведении и фокусе. Кнопка удаления,
              // висящая рядом с каждой строкой постоянно, приглашает по
              // себе попасть — а промах здесь невосстановим.
              className="flex size-6 shrink-0 items-center justify-center rounded text-slate-300 opacity-0 transition hover:bg-white hover:text-red-600 focus-visible:opacity-100 group-hover:opacity-100"
            >
              <IconTrash className="size-3.5" />
            </button>
          </div>
        ))}

        {threads.length === 0 ? (
          <p className="px-2.5 py-2 text-xs text-slate-400">пока пусто</p>
        ) : null}
        {threads.length > VISIBLE ? (
          <p className="px-2.5 py-1 text-xs text-slate-400">
            и ещё {threads.length - VISIBLE}
          </p>
        ) : null}
      </div>
    </div>
  )
}
