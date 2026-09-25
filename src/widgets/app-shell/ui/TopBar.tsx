import { useEffect, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useHealth } from '@/entities/document'
import { ModelPicker } from '@/features/select-model'
import { AgentToggle } from '@/features/toggle-agent'
import { Badge } from '@/shared/ui/primitives'
import { IconPanel, IconSearch } from '@/shared/ui/icons'
import { cn } from '@/shared/lib/cn'
import { AlertsBadge } from './AlertsBadge'

/**
 * Верхняя панель: поиск по вики, выбор модели, состояние провайдера.
 *
 * Поиск здесь обычный, по названиям — он не заменяет ассистента и не
 * притворяется им. Это важно: строка поиска, которая молча отправляет запрос
 * в модель, приучает людей думать, что поиск и ответ — одно и то же, и потом
 * каждый промах поиска выглядит как галлюцинация модели.
 *
 * Состояние провайдера вынесено сюда из логов по той же причине, что и
 * состояние индекса: «модель не отвечает» — вторая по частоте причина
 * «ассистент сломался», и человек должен видеть её, а не гадать.
 */
export function TopBar({
  assistantOpen,
  onToggleAssistant,
}: {
  assistantOpen: boolean
  onToggleAssistant: () => void
}) {
  const { data: health } = useHealth()
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const [query, setQuery] = useState(params.get('q') ?? '')
  const input = useRef<HTMLInputElement>(null)

  // Ctrl/Cmd+K — привычный способ попасть в поиск. Стоит одну строчку и
  // экономит движение мышью на каждом вопросе.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        input.current?.focus()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [])

  const providers = health?.providers ?? []

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-slate-200 bg-white px-4">
      <div className="flex w-52 shrink-0 items-center gap-2">
        <span className="flex size-7 items-center justify-center rounded-lg bg-slate-900 text-xs font-bold text-white">
          PW
        </span>
        <span className="text-[15px] font-semibold text-slate-900">PIPEWIKI</span>
      </div>

      <form
        className="relative mx-auto hidden w-full max-w-lg md:block"
        onSubmit={(event) => {
          event.preventDefault()
          navigate(query.trim() ? `/wiki?q=${encodeURIComponent(query.trim())}` : '/wiki')
        }}
      >
        <IconSearch className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-slate-400" />
        <input
          ref={input}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Поиск по документам…"
          className="w-full rounded-lg border border-slate-200 bg-slate-50 py-2 pl-9 pr-14 text-sm outline-none transition focus:border-slate-400 focus:bg-white"
        />
        <kbd className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 rounded border border-slate-200 bg-white px-1.5 py-0.5 text-[11px] text-slate-400">
          ⌘K
        </kbd>
      </form>

      <div className="ml-auto flex items-center gap-2">
        <div className="hidden items-center gap-1.5 xl:flex">
          {providers.map((provider, index) => {
            const ok = provider.ok === true
            return (
              <Badge key={index} tone={ok ? 'good' : 'bad'}>
                {String(provider.provider ?? 'провайдер')}: {ok ? 'на связи' : 'нет ответа'}
              </Badge>
            )
          })}
        </div>

        <AlertsBadge />
        <AgentToggle />
        <ModelPicker />

        {/* Кнопка показывает СОСТОЯНИЕ, а не просто существует. Раньше
            она выглядела одинаково при открытой и закрытой панели, и на
            узком экране, где панель лежит поверх содержимого, понять по
            ней ничего было нельзя. */}
        <button
          type="button"
          onClick={onToggleAssistant}
          aria-pressed={assistantOpen}
          aria-label={assistantOpen ? 'скрыть панель ассистента' : 'показать панель ассистента'}
          title={assistantOpen ? 'скрыть ассистента' : 'показать ассистента'}
          className={cn(
            'flex size-9 items-center justify-center rounded-lg border transition',
            assistantOpen
              ? 'border-sky-200 bg-sky-50 text-sky-700'
              : 'border-slate-200 text-slate-500 hover:bg-slate-50 hover:text-slate-800',
          )}
        >
          <IconPanel className="size-4" />
        </button>
      </div>
    </header>
  )
}
