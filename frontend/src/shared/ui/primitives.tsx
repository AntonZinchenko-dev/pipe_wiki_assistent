import type { ReactNode } from 'react'
import { useState } from 'react'
import { cn } from '../lib/cn'

/**
 * Базовые элементы интерфейса.
 *
 * Все цвета здесь — РОЛИ (`surface`, `ink-soft`, `accent`), а не оттенки.
 * Компонент не знает, светлая тема сейчас или тёмная, и знать не должен:
 * иначе каждая новая тема — это обход всех файлов, и поэтому её не делают.
 */

export function Button({
  children,
  onClick,
  variant = 'primary',
  disabled,
  type = 'button',
  size = 'md',
}: {
  children: ReactNode
  onClick?: () => void
  variant?: 'primary' | 'ghost' | 'quiet' | 'danger'
  disabled?: boolean
  type?: 'button' | 'submit'
  size?: 'sm' | 'md'
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition',
        'disabled:cursor-not-allowed disabled:opacity-40',
        size === 'sm' ? 'px-2.5 py-1.5 text-xs' : 'px-3.5 py-2 text-sm',
        variant === 'primary' && 'bg-accent text-white shadow-card hover:bg-accent-hover',
        variant === 'ghost' &&
          'border border-line bg-surface text-ink shadow-card hover:border-ink-faint',
        variant === 'quiet' && 'text-ink-soft hover:bg-sunken hover:text-ink',
        variant === 'danger' && 'border border-bad/30 text-bad-ink hover:bg-bad-soft',
      )}
    >
      {children}
    </button>
  )
}

/**
 * Плашка состояния.
 *
 * Тон — про смысл, а не про красоту: `good` значит «работает», `bad` —
 * «сломано». Поэтому нейтральный тон здесь самый частый и самый бледный:
 * если раскрасить всё, цвет перестаёт что-либо сообщать, и настоящая
 * красная плашка теряется среди зелёных.
 */
export function Badge({
  children,
  tone = 'neutral',
}: {
  children: ReactNode
  tone?: 'neutral' | 'good' | 'warn' | 'bad' | 'info' | 'live'
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium',
        tone === 'neutral' && 'bg-sunken text-ink-soft',
        tone === 'good' && 'bg-good-soft text-good-ink',
        tone === 'warn' && 'bg-warn-soft text-warn-ink',
        tone === 'bad' && 'bg-bad-soft text-bad-ink',
        tone === 'info' && 'bg-accent-soft text-accent-ink',
        tone === 'live' && 'bg-live-soft text-live-ink',
      )}
    >
      {children}
    </span>
  )
}

/**
 * Точка состояния — плашка для случаев, когда важен только факт «жив или
 * нет», а слово «на связи» рядом с каждым провайдером превращает шапку в
 * гирлянду.
 */
export function Dot({ tone }: { tone: 'good' | 'warn' | 'bad' }) {
  return (
    <span
      aria-hidden
      className={cn(
        'size-1.5 shrink-0 rounded-full',
        tone === 'good' && 'bg-good',
        tone === 'warn' && 'bg-warn',
        tone === 'bad' && 'bg-bad',
      )}
    />
  )
}

/**
 * Свёрнутый блок «как получен ответ».
 *
 * Свёрнутый по умолчанию: это и прозрачность для пользователя, и готовый
 * инструмент отладки — что искали, что нашли, с какими баллами.
 */
export function Disclosure({
  summary,
  children,
  defaultOpen = false,
}: {
  summary: ReactNode
  children: ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="overflow-hidden rounded-lg border border-line bg-surface">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-medium text-ink-soft transition hover:bg-sunken hover:text-ink"
        aria-expanded={open}
      >
        {/* Стрелка, а не «плюс-минус»: поворот на 90° читается как
            «раскрылось», а смена знака — как «стало другим». */}
        <svg
          aria-hidden
          viewBox="0 0 16 16"
          className={cn('size-3 shrink-0 transition-transform', open && 'rotate-90')}
        >
          <path
            d="M6 4l4 4-4 4"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span>{summary}</span>
      </button>
      {open ? <div className="border-t border-line-soft px-3 py-2.5">{children}</div> : null}
    </div>
  )
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex gap-3 text-xs">
      <dt className="min-w-36 shrink-0 text-ink-faint">{label}</dt>
      <dd className="min-w-0 text-ink">{children}</dd>
    </div>
  )
}

export function Spinner({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-ink-soft">
      <span
        aria-hidden
        className="size-3 animate-spin rounded-full border-2 border-line border-t-accent"
      />
      {label}
    </span>
  )
}

/**
 * Заголовок группы в меню и в панелях.
 *
 * Отдельным компонентом, потому что таких заголовков четыре в разных
 * файлах, и они успели разъехаться по кеглю и отступам — разница мелкая,
 * но именно из неё складывается ощущение, что интерфейс собран наспех.
 */
export function GroupLabel({ children }: { children: ReactNode }) {
  return (
    <p className="px-2.5 pb-1.5 pt-4 text-[11px] font-semibold uppercase tracking-wider text-ink-faint">
      {children}
    </p>
  )
}
