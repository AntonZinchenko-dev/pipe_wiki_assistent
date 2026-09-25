import { useEffect, useRef, useState } from 'react'
import { isLocal, shortLabel, useModelChoice, useModels } from '@/entities/chat-model'
import type { ModelChoice } from '@/shared/api/contracts'
import { IconCheck, IconChevronDown, IconCloud, IconCpu } from '@/shared/ui/icons'
import { cn } from '@/shared/lib/cn'

/**
 * Выбор модели в шапке.
 *
 * Зачем он вообще нужен. Локальная модель и облачная отвечают по-разному, и
 * увидеть разницу на своём вопросе полезнее, чем прочитать про неё в отчёте
 * о замерах. Раньше для этого надо было править файл настроек и
 * перезапускать сервер — то есть сравнение стоило пяти минут и потому не
 * делалось.
 *
 * Три вещи, которые здесь сделаны намеренно.
 *
 * ПЕРВОЕ. Список приходит с сервера и означает «что есть», а не «что мы
 * написали в настройках». Зашитый в код список устаревает молча: у нас в
 * настройках стояло имя модели, которой у сервиса больше нет, и отказ
 * выглядел как поломка облака.
 *
 * ВТОРОЕ. Пункт «как настроено» — не то же самое, что первая модель в
 * списке. Человек, ничего не выбиравший, должен ехать вместе с настройкой
 * сервера, а не остаться навсегда на той модели, которая была первой в день
 * открытия вкладки.
 *
 * ТРЕТЬЕ. У облачных моделей стоит пометка про деньги. Не предупреждение на
 * пол-экрана, а строчка рядом: локальная бесплатна, облачная считает токены
 * со счёта. Человек имеет право это знать до нажатия, а не из счёта в конце
 * месяца.
 */
export function ModelPicker() {
  const { data, isLoading, error } = useModels()
  const chosen = useModelChoice((state) => state.chosen)
  const choose = useModelChoice((state) => state.choose)
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)

  // Закрытие по клику мимо и по Escape. Меню, которое закрывается только
  // повторным нажатием на кнопку, — классическая ловушка: человек кликает в
  // сторону, меню остаётся висеть поверх содержимого.
  useEffect(() => {
    if (!open) return
    function onPointerDown(event: MouseEvent) {
      if (box.current && !box.current.contains(event.target as Node)) setOpen(false)
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onPointerDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onPointerDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const serverDefault = data?.default
  const current = chosen ?? serverDefault ?? null
  const label = current ? shortLabel(current.model) : isLoading ? 'загрузка…' : 'модель'
  const local = current ? isLocal(current.provider) : true

  const options: ModelChoice[] = data?.models ?? []
  const byProvider = new Map<string, ModelChoice[]>()
  for (const option of options) {
    const group = byProvider.get(option.provider) ?? []
    group.push(option)
    byProvider.set(option.provider, group)
  }

  function pick(option: ModelChoice | null): void {
    choose(option)
    setOpen(false)
  }

  return (
    <div className="relative" ref={box}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={cn(
          'flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-sm transition',
          'border-line bg-surface text-ink hover:border-line hover:bg-sunken',
        )}
        title={current ? `${current.provider} · ${current.model}` : 'выбор модели'}
      >
        {local ? (
          <IconCpu className="size-4 text-ink-faint" />
        ) : (
          <IconCloud className="size-4 text-accent" />
        )}
        <span className="max-w-44 truncate font-medium">{label}</span>
        {chosen ? null : (
          <span className="hidden text-xs text-ink-faint sm:inline">по умолчанию</span>
        )}
        <IconChevronDown className="size-4 text-ink-faint" />
      </button>

      {open ? (
        <div
          role="listbox"
          className="absolute right-0 z-30 mt-1.5 w-80 overflow-hidden rounded-xl border border-line bg-surface shadow-raised"
        >
          <button
            type="button"
            role="option"
            aria-selected={chosen === null}
            onClick={() => pick(null)}
            className="flex w-full items-start gap-2 px-3 py-2.5 text-left hover:bg-sunken"
          >
            <span className="mt-0.5 size-4 shrink-0">
              {chosen === null ? <IconCheck className="size-4 text-accent" /> : null}
            </span>
            <span className="min-w-0">
              <span className="block text-sm font-medium text-ink">Как настроено</span>
              <span className="block text-xs text-ink-soft">
                {serverDefault
                  ? `сейчас это ${shortLabel(serverDefault.model)}`
                  : 'первый провайдер из цепочки на сервере'}
              </span>
            </span>
          </button>

          {error ? (
            <p className="border-t border-line-soft px-3 py-2.5 text-xs text-bad-ink">
              Список моделей не пришёл: {(error as Error).message}
            </p>
          ) : null}

          {[...byProvider.entries()].map(([provider, group]) => (
            <div key={provider} className="border-t border-line-soft">
              <p className="flex items-center gap-1.5 px-3 pb-1 pt-2.5 text-xs font-semibold uppercase tracking-wide text-ink-faint">
                {isLocal(provider) ? (
                  <IconCpu className="size-3.5" />
                ) : (
                  <IconCloud className="size-3.5" />
                )}
                {provider}
                <span className="font-normal normal-case tracking-normal">
                  {isLocal(provider) ? '· бесплатно' : '· тратит токены со счёта'}
                </span>
              </p>
              {group.map((option) => {
                const active =
                  chosen?.provider === option.provider && chosen?.model === option.model
                return (
                  <button
                    key={`${option.provider}/${option.model}`}
                    type="button"
                    role="option"
                    aria-selected={active}
                    onClick={() => pick(option)}
                    className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-sunken"
                  >
                    <span className="size-4 shrink-0">
                      {active ? <IconCheck className="size-4 text-accent" /> : null}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-sm text-ink">
                      {option.model}
                    </span>
                    {option.is_default ? (
                      <span className="shrink-0 rounded bg-sunken px-1.5 py-0.5 text-xs text-ink-soft">
                        сервер
                      </span>
                    ) : null}
                  </button>
                )
              })}
            </div>
          ))}

          {!isLoading && options.length === 0 && !error ? (
            <p className="border-t border-line-soft px-3 py-2.5 text-xs text-ink-soft">
              Провайдеры не вернули ни одной модели. Проверьте, запущена ли Ollama.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
