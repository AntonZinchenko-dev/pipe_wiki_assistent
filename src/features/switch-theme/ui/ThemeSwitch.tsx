import { useEffect, useState } from 'react'
import { IconMonitor, IconMoon, IconSun } from '@/shared/ui/icons'
import { applyTheme, readTheme, saveTheme, watchSystemTheme, type Theme } from '@/shared/lib/theme'
import { cn } from '@/shared/lib/cn'

/**
 * Переключатель темы: три положения в одной дорожке.
 *
 * Дорожка, а не выпадающий список: положений три, они короткие, и текущее
 * должно быть видно без клика. Список здесь означал бы «нажми, чтобы
 * узнать, что стоит сейчас», — лишнее движение ради информации, которая
 * помещается в тридцать пикселей.
 *
 * Подписи нет, потому что три иконки — солнце, луна, монитор — читаются
 * без слов, а подписи в шапке заняли бы место поиска. Каждая кнопка при
 * этом называет себя для чтения с экрана.
 */

const OPTIONS: Array<{ value: Theme; label: string; Icon: typeof IconSun }> = [
  { value: 'light', label: 'светлая тема', Icon: IconSun },
  { value: 'dark', label: 'тёмная тема', Icon: IconMoon },
  { value: 'system', label: 'как в системе', Icon: IconMonitor },
]

export function ThemeSwitch() {
  const [theme, setTheme] = useState<Theme>(readTheme)

  useEffect(() => {
    applyTheme(theme)
    saveTheme(theme)
    // Слушаем систему только в режиме «как в системе»: в остальных
    // человек уже высказался, и менять интерфейс под ним — не услужливость,
    // а игнорирование его выбора.
    if (theme !== 'system') return
    return watchSystemTheme(() => applyTheme('system'))
  }, [theme])

  return (
    <div
      role="group"
      aria-label="тема оформления"
      className="flex items-center gap-0.5 rounded-lg border border-line bg-sunken p-0.5"
    >
      {OPTIONS.map(({ value, label, Icon }) => (
        <button
          key={value}
          type="button"
          onClick={() => setTheme(value)}
          aria-label={label}
          aria-pressed={theme === value}
          title={label}
          className={cn(
            'flex size-6 items-center justify-center rounded-md transition',
            theme === value
              ? 'bg-surface text-ink shadow-card'
              : 'text-ink-faint hover:text-ink-soft',
          )}
        >
          <Icon className="size-3.5" />
        </button>
      ))}
    </div>
  )
}
