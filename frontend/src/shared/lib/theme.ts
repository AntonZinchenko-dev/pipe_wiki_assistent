/**
 * Светлая тема, тёмная или «как в системе».
 *
 * Три состояния, а не переключатель. Переключатель на два положения
 * заставляет выбрать один раз и навсегда: человек, у которого система
 * сама темнеет к вечеру, оказывается с намертво светлым интерфейсом. А
 * «как в системе» без явных положений отнимает выбор у того, кому днём
 * удобнее тёмная.
 *
 * Хранится в браузере, а не на сервере: это настройка рабочего места, как
 * яркость монитора, а не свойство учётной записи.
 */

export type Theme = 'system' | 'light' | 'dark'

const KEY = 'pipewiki.theme'

export function readTheme(): Theme {
  try {
    const saved = localStorage.getItem(KEY)
    if (saved === 'light' || saved === 'dark' || saved === 'system') return saved
  } catch {
    // Приватное окно или запрет хранилища. Не повод падать.
  }
  return 'system'
}

/**
 * Проставляет атрибут на `<html>`, по которому работают все стили.
 *
 * Считаем ЗДЕСЬ, а не в CSS через `prefers-color-scheme`, потому что режим
 * «всегда светлая» обязан побеждать системную настройку, а медиазапрос
 * перебить нечем, кроме дублирования каждого правила.
 */
export function applyTheme(theme: Theme): void {
  const dark =
    theme === 'dark' ||
    (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)
  document.documentElement.dataset.theme = dark ? 'dark' : 'light'
  // Системные элементы — полосы прокрутки, поля ввода, выпадающие списки —
  // красит браузер, и без этой строчки они остаются светлыми на тёмном.
  document.documentElement.style.colorScheme = dark ? 'dark' : 'light'
}

export function saveTheme(theme: Theme): void {
  try {
    localStorage.setItem(KEY, theme)
  } catch {
    // см. readTheme
  }
}

/**
 * Подписка на смену системной темы.
 *
 * Нужна только в режиме «как в системе»: без неё интерфейс переключится
 * лишь после перезагрузки страницы, то есть вечером останется светлым до
 * следующего утра.
 */
export function watchSystemTheme(onChange: () => void): () => void {
  const media = window.matchMedia('(prefers-color-scheme: dark)')
  media.addEventListener('change', onChange)
  return () => media.removeEventListener('change', onChange)
}
