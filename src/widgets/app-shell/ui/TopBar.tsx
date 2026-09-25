import { useEffect, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useHealth } from '@/entities/document'
import { ModelPicker } from '@/features/select-model'
import { ThemeSwitch } from '@/features/switch-theme'
import { AgentToggle } from '@/features/toggle-agent'
import { Dot } from '@/shared/ui/primitives'
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

/**
 * Провайдеры — ТОЧКАМИ, а не подписями «на связи».
 *
 * Раньше здесь висели две-три зелёные плашки со словами. Когда всё
 * работает — а это почти всегда — они сообщали ровно ничего и при этом
 * занимали середину шапки и перетягивали взгляд ярким цветом. Теперь
 * норма занимает четыре пикселя и молчит, а поломка становится единственным
 * красным пятном на экране. Подробности — во всплывающей подсказке: они
 * нужны раз в месяц, и ради них незачем держать строку постоянно.
 */
function Providers() {
  const { data: health } = useHealth()
  const providers = health?.providers ?? []
  if (!providers.length) return null

  const down = providers.filter((provider) => provider.ok !== true)
  const label = down.length
    ? `нет ответа: ${down.map((provider) => String(provider.provider ?? '?')).join(', ')}`
    : `на связи: ${providers.map((provider) => String(provider.provider ?? '?')).join(', ')}`

  return (
    <span
      title={label}
      aria-label={label}
      className={cn(
        'hidden items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs md:flex',
        down.length ? 'bg-bad-soft text-bad-ink' : 'text-ink-faint',
      )}
    >
      {providers.map((provider, index) => (
        <Dot key={index} tone={provider.ok === true ? 'good' : 'bad'} />
      ))}
      {down.length ? <span className="font-medium">нет ответа</span> : null}
    </span>
  )
}

export function TopBar({
  assistantOpen,
  onToggleAssistant,
}: {
  assistantOpen: boolean
  onToggleAssistant: () => void
}) {
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

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-line bg-chrome px-4">
      <div className="flex shrink-0 items-center gap-2.5 lg:w-60">
        <span className="flex size-8 items-center justify-center rounded-[10px] bg-accent text-[13px] font-bold tracking-tight text-white shadow-card">
          PW
        </span>
        <span className="hidden text-[15px] font-semibold tracking-tight text-ink sm:block">
          PIPEWIKI
        </span>
      </div>

      <form
        className="relative mx-auto hidden w-full max-w-xl md:block"
        onSubmit={(event) => {
          event.preventDefault()
          navigate(query.trim() ? `/wiki?q=${encodeURIComponent(query.trim())}` : '/wiki')
        }}
      >
        <IconSearch className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-ink-faint" />
        <input
          ref={input}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Поиск по документам…"
          className="w-full rounded-lg border border-line bg-sunken py-2 pl-9 pr-16 text-sm text-ink outline-none transition placeholder:text-ink-faint focus:border-accent-line focus:bg-surface focus:shadow-card"
        />
        <kbd className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 rounded border border-line bg-surface px-1.5 py-0.5 font-sans text-[11px] text-ink-faint">
          ⌘K
        </kbd>
      </form>

      <div className="ml-auto flex items-center gap-1.5">
        <Providers />
        <AlertsBadge />
        <AgentToggle />
        <ModelPicker />
        <ThemeSwitch />

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
            'flex size-8 items-center justify-center rounded-lg border transition',
            assistantOpen
              ? 'border-accent-line bg-accent-soft text-accent-ink'
              : 'border-line text-ink-soft hover:bg-sunken hover:text-ink',
          )}
        >
          <IconPanel className="size-4" />
        </button>
      </div>
    </header>
  )
}
