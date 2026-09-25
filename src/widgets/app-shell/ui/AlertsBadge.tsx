import { useEffect, useRef, useState } from 'react'
import { useAlerts } from '@/entities/alert'
import type { AlertMetric } from '@/shared/api/contracts'
import { IconClock } from '@/shared/ui/icons'
import { cn } from '@/shared/lib/cn'

/**
 * Индикатор порогов оповещения в шапке.
 *
 * Зачем он на экране, а не только в журнале. Журнал читают, когда уже
 * пришли жаловаться. Пороги на виду означают, что человек, который сидит в
 * системе, увидит проблему раньше — и увидит её в тех же терминах, что и
 * дежурный по регламенту.
 *
 * В спокойном состоянии индикатор МОЛЧИТ: серая иконка без цифр. Значок,
 * который всё время что-то показывает, перестают замечать ровно так же,
 * как журнал, полный одинаковых строк.
 *
 * Прочерк вместо значения — это «не с чем сравнить», а не «ноль». Мало
 * наблюдений или не задан бюджет: в обоих случаях показать ноль значило бы
 * соврать успокаивающим образом.
 */

const LEVEL_LABEL: Record<string, string> = {
  ok: 'в норме',
  warn: 'внимание',
  crit: 'авария',
}

function value(metric: AlertMetric): string {
  if (metric.value === null) return '—'
  // Доли показываем процентами, секунды — секундами. Отличаем по порогу:
  // у долевых метрик он меньше единицы.
  return metric.crit_at <= 1.5 ? `${(metric.value * 100).toFixed(1)} %` : `${metric.value} с`
}

export function AlertsBadge() {
  const { data } = useAlerts()
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)

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

  if (!data) return null
  const level = data.level

  return (
    <div className="relative" ref={box}>
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        title={`Пороги оповещения: ${LEVEL_LABEL[level] ?? level}`}
        className={cn(
          'flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-sm transition',
          level === 'crit' && 'border-red-300 bg-red-50 font-medium text-red-700',
          level === 'warn' && 'border-amber-300 bg-amber-50 font-medium text-amber-800',
          level === 'ok' && 'border-slate-200 bg-white text-slate-500 hover:bg-slate-50',
        )}
      >
        <IconClock className="size-4" />
        {level === 'ok' ? null : <span>{LEVEL_LABEL[level]}</span>}
      </button>

      {open ? (
        <div className="absolute right-0 z-30 mt-1.5 w-96 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-lg shadow-slate-900/5">
          <p className="border-b border-slate-100 px-3 py-2 text-xs text-slate-500">
            Окно {data.window_minutes} мин, наблюдений {data.samples}
          </p>

          <ul>
            {data.metrics.map((metric) => (
              <li key={metric.metric} className="border-b border-slate-100 px-3 py-2 last:border-0">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-sm text-slate-800">{metric.title}</span>
                  <span
                    className={cn(
                      'shrink-0 text-sm font-medium',
                      metric.level === 'crit' && 'text-red-700',
                      metric.level === 'warn' && 'text-amber-700',
                      metric.level === 'ok' && 'text-slate-500',
                    )}
                  >
                    {value(metric)}
                  </span>
                </div>
                <p className="mt-0.5 text-xs text-slate-500">{metric.detail}</p>
                {/* Откуда порог. Дежурный работает по регламенту, и он
                    должен видеть, наш это порог или оттуда. */}
                <p className="text-xs text-slate-400">источник: {metric.source}</p>
              </li>
            ))}
          </ul>

          {data.events.length > 0 ? (
            <div className="border-t border-slate-200 bg-slate-50/70 px-3 py-2">
              <p className="pb-1 text-xs font-semibold text-slate-500">Последние переходы</p>
              <ul className="space-y-0.5">
                {[...data.events].reverse().slice(0, 5).map((event, index) => (
                  <li key={index} className="text-xs text-slate-600">
                    {new Date(event.at * 1000).toLocaleTimeString('ru-RU')} — {event.text}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
