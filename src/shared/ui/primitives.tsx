import type { ReactNode } from 'react'
import { useState } from 'react'
import { cn } from '../lib/cn'

export function Button({
  children,
  onClick,
  variant = 'primary',
  disabled,
  type = 'button',
}: {
  children: ReactNode
  onClick?: () => void
  variant?: 'primary' | 'ghost' | 'danger'
  disabled?: boolean
  type?: 'button' | 'submit'
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'rounded-md px-3 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40',
        variant === 'primary' && 'bg-slate-900 text-white hover:bg-slate-700',
        variant === 'ghost' && 'border border-slate-300 text-slate-700 hover:bg-slate-100',
        variant === 'danger' && 'border border-red-300 text-red-700 hover:bg-red-50',
      )}
    >
      {children}
    </button>
  )
}

export function Badge({
  children,
  tone = 'neutral',
}: {
  children: ReactNode
  tone?: 'neutral' | 'good' | 'warn' | 'bad' | 'info'
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium',
        tone === 'neutral' && 'bg-slate-100 text-slate-600',
        tone === 'good' && 'bg-emerald-50 text-emerald-700',
        tone === 'warn' && 'bg-amber-50 text-amber-700',
        tone === 'bad' && 'bg-red-50 text-red-700',
        tone === 'info' && 'bg-sky-50 text-sky-700',
      )}
    >
      {children}
    </span>
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
    <div className="rounded-md border border-slate-200 bg-slate-50/60">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between px-3 py-2 text-left text-xs font-medium text-slate-600 hover:text-slate-900"
        aria-expanded={open}
      >
        <span>{summary}</span>
        <span aria-hidden>{open ? '−' : '+'}</span>
      </button>
      {open ? <div className="border-t border-slate-200 px-3 py-2">{children}</div> : null}
    </div>
  )
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex gap-2 text-xs">
      <dt className="min-w-36 shrink-0 text-slate-500">{label}</dt>
      <dd className="text-slate-800">{children}</dd>
    </div>
  )
}

export function Spinner({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-slate-500">
      <span
        aria-hidden
        className="size-3 animate-spin rounded-full border-2 border-slate-300 border-t-slate-700"
      />
      {label}
    </span>
  )
}
