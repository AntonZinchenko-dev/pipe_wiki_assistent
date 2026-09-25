import { useEffect, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { IconClose } from './icons'

/**
 * Выдвижная панель поверх приложения.
 *
 * Нужна там, где содержимое важное, но нужно редко. Разбор ответа — ровно
 * такой случай: проверить, откуда взялась цифра, человек хочет раз в
 * двадцать ответов, а место в ленте эти двести строк занимали всегда.
 * Свёрнутый блок задачу не решал: он всё равно оставлял в ленте заголовок,
 * рамку и отступы у КАЖДОГО ответа, и переписка из пяти вопросов
 * превращалась в простыню, по которой не докрутить до нужного места.
 *
 * ПОВЕРХ, а не рядом. Панель ассистента и так узкая; отдать её половину
 * разбору значит сделать нечитаемым и то и другое.
 *
 * Через портал в `body`, а не внутри карточки: иначе `overflow: auto` у
 * ленты обрежет панель по своей границе, и та превратится в блок внутри
 * прокрутки — то есть ровно в то, от чего мы уходим.
 */
export function Drawer({
  open,
  title,
  onClose,
  children,
}: {
  open: boolean
  title: ReactNode
  onClose: () => void
  children: ReactNode
}) {
  // Esc закрывает, и страница под панелью не прокручивается.
  //
  // Прокрутка фона — не придирка: человек крутит колесо, думая, что читает
  // разбор, а уезжает лента под ним. Вернувшись, он не находит того
  // ответа, с которого начал.
  useEffect(() => {
    if (!open) return
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose()
    }
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    document.addEventListener('keydown', onKey)
    return () => {
      document.body.style.overflow = previous
      document.removeEventListener('keydown', onKey)
    }
  }, [open, onClose])

  if (!open) return null

  return createPortal(
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* Затемнение кликабельно: закрыть, ткнув мимо, — то, что люди
          пробуют первым, раньше, чем ищут крестик. */}
      <button
        type="button"
        aria-label="закрыть разбор"
        onClick={onClose}
        className="absolute inset-0 bg-ink/25 backdrop-blur-[1px]"
      />

      <aside
        role="dialog"
        aria-modal="true"
        className="relative flex h-full w-full max-w-xl flex-col border-l border-line bg-chrome shadow-panel"
      >
        <header className="flex h-12 shrink-0 items-center gap-2 border-b border-line px-4">
          <span className="min-w-0 flex-1 truncate text-sm font-semibold text-ink">{title}</span>
          <button
            type="button"
            onClick={onClose}
            aria-label="закрыть"
            className="flex size-7 items-center justify-center rounded-md text-ink-faint transition hover:bg-sunken hover:text-ink"
          >
            <IconClose className="size-4" />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">{children}</div>
      </aside>
    </div>,
    document.body,
  )
}
