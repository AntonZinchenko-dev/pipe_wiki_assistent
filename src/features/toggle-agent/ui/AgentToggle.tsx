import { useAgentMode, useModels } from '@/entities/chat-model'
import { useViewer } from '@/entities/viewer'
import { IconSparkle } from '@/shared/ui/icons'
import { cn } from '@/shared/lib/cn'

/**
 * Переключатель агентского режима.
 *
 * Что он меняет. Обычно система ищет один раз и отвечает по найденному. С
 * включённым режимом модель смотрит на результат поиска и может поискать
 * ещё раз другими словами или дочитать соседний раздел документа.
 *
 * Почему это переключатель, а не всегдашнее поведение. Каждый шаг агента —
 * это лишний вызов модели: дольше и дороже. Выигрыш он даёт не всегда, и
 * пока не измерено на золотом наборе, включать его по умолчанию значит
 * поменять систему, не зная, в какую сторону.
 *
 * Если сервер режим не разрешил или его не разрешили этому человеку —
 * кнопки просто нет. Кнопка, которая всегда возвращает ошибку, хуже
 * отсутствующей: человек считает её сломанной и перестаёт верить
 * остальному интерфейсу.
 *
 * Проверка прав ЗДЕСЬ — удобство, а не защита. Настоящая проверка на
 * сервере: браузер открывается инструментами разработчика, и решать в нём,
 * что человеку положено, бессмысленно.
 */
export function AgentToggle() {
  const { data } = useModels()
  const { data: viewer } = useViewer()
  const on = useAgentMode((state) => state.on)
  const toggle = useAgentMode((state) => state.toggle)

  // Два независимых условия: режим разрешён НА СЕРВЕРЕ и он разрешён
  // ЭТОМУ человеку. Первое про установку, второе про права, и подменять
  // одно другим нельзя — иначе достаточно будет включить режим, чтобы он
  // стал доступен всем.
  if (!data?.agent?.available) return null
  if (viewer && !viewer.rights.agent) return null

  return (
    <button
      type="button"
      onClick={toggle}
      aria-pressed={on}
      title={
        on
          ? `Модель может дособрать контекст: до ${data.agent.max_steps} доп. шагов. Дольше и дороже.`
          : 'Один поиск, затем ответ'
      }
      className={cn(
        'flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-sm transition',
        on
          ? 'border-accent-line bg-accent-soft font-medium text-accent-ink'
          : 'border-line bg-surface text-ink-soft hover:bg-sunken',
      )}
    >
      <IconSparkle className={cn('size-4', on ? 'text-accent' : 'text-ink-faint')} />
      <span className="hidden sm:inline">Агент</span>
    </button>
  )
}
