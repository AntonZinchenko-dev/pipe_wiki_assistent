import { useEffect, useRef, useState } from 'react'
import { AnswerCard, isBusy } from '@/entities/answer'
import { useAssistant, useThreadSync } from '@/features/ask-assistant'
import { Button } from '@/shared/ui/primitives'
import { IconSend, IconSparkle } from '@/shared/ui/icons'

/**
 * Лента диалога с ассистентом — правая панель приложения.
 *
 * Панель, а не отдельная страница, сознательно: вопрос почти всегда
 * возникает при чтении документа, и уход на другой экран заставляет
 * запоминать, о чём был вопрос. Рядом — значит можно сверить ответ с текстом
 * глазами, не переключаясь.
 *
 * Примеры вопросов на пустом экране — не украшение. Человек, впервые
 * увидевший строку ввода, не знает, что этой штуке можно задавать, и
 * набирает либо слишком общее, либо ничего. Три готовых вопроса за секунду
 * показывают жанр.
 */

const EXAMPLES = [
  'Как отключить кэш последних значений в шлюзе телеметрии?',
  'Что означает ошибка E-1042?',
  'Какой порог по усталости считается аварийным?',
]

export function ChatThread() {
  const [draft, setDraft] = useState('')
  const exchanges = useAssistant((state) => state.exchanges)
  const activeId = useAssistant((state) => state.activeId)
  const ask = useAssistant((state) => state.ask)
  const stop = useAssistant((state) => state.stop)
  const startNew = useAssistant((state) => state.startNew)
  const bottom = useRef<HTMLDivElement>(null)

  // Лента следует за выбранным диалогом. Подписка стоит здесь, в
  // виджете: он единственный, кто одновременно знает и про список
  // диалогов, и про ленту.
  useThreadSync()

  const active = exchanges.find((exchange) => exchange.id === activeId)
  const streaming = Boolean(active && isBusy(active.phase))

  // Прокрутка к последнему ответу при добавлении обмена, а НЕ на каждый
  // токен потока: иначе страницу утаскивает вниз прямо из-под читающего.
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [exchanges.length])

  function submit(question: string): void {
    const text = question.trim()
    if (!text) return
    setDraft('')
    void ask(text)
  }

  return (
    <section className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-4">
        {exchanges.length === 0 ? (
          <div className="space-y-5 pt-2">
            <div>
              <span className="mb-3 flex size-9 items-center justify-center rounded-xl bg-accent-soft text-accent">
                <IconSparkle className="size-5" />
              </span>
              <h2 className="text-[17px] font-semibold tracking-tight text-ink">
                Спросите о чём угодно из вики
              </h2>
              <p className="mt-1.5 text-[13.5px] leading-relaxed text-ink-soft">
                Ассистент отвечает только по документам вики и обязан ссылаться на фрагменты.
                Если ответа в документации нет, он скажет это прямо, а не придумает.
              </p>
            </div>
            {/* Примеры — в столбик, а не облаком плиток.
                Облаком они были разной ширины и переносились по-разному на
                каждой ширине панели; список читается сверху вниз и всегда
                выглядит одинаково. */}
            <div className="flex flex-col gap-1.5">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => submit(example)}
                  className="group flex items-center gap-2.5 rounded-lg border border-line bg-surface px-3 py-2.5 text-left text-[13px] text-ink-soft shadow-card transition hover:border-accent-line hover:text-ink"
                >
                  <IconSparkle className="size-3.5 shrink-0 text-ink-faint transition group-hover:text-accent" />
                  <span className="min-w-0 flex-1">{example}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          exchanges.map((exchange) => <AnswerCard key={exchange.id} exchange={exchange} />)
        )}
        <div ref={bottom} />
      </div>

      <form
        className="shrink-0 border-t border-line bg-surface p-3"
        onSubmit={(event) => {
          event.preventDefault()
          submit(draft)
        }}
      >
        <div className="rounded-xl border border-line bg-surface shadow-card transition focus-within:border-accent-line focus-within:shadow-raised">
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                submit(draft)
              }
            }}
            rows={2}
            placeholder="Вопрос по документации…"
            className="min-h-16 w-full resize-y rounded-xl bg-transparent px-3 py-2.5 text-sm text-ink outline-none placeholder:text-ink-faint"
          />
          <div className="flex items-center justify-between gap-2 px-2 pb-2">
            {/* Подсказка прячется на узкой панели.
                Целиком она не помещается рядом с кнопками и переносится на
                вторую строку, растаскивая низ формы. Человеку, у которого
                панель узкая, важнее видеть кнопку «Спросить», чем
                напоминание про Shift+Enter. */}
            <span className="hidden pl-1 text-[11px] text-ink-faint sm:block">
              <kbd className="font-sans">Enter</kbd> — отправить
            </span>
            <div className="flex items-center gap-2">
              {exchanges.length > 0 && !streaming ? (
                <Button variant="quiet" size="sm" onClick={startNew}>
                  Новый диалог
                </Button>
              ) : null}
              {/* Кнопка «Стоп» доступна ровно в фазе потока — это следствие
                  явного автомата состояний, а не отдельного флага isLoading. */}
              {streaming ? (
                <Button variant="danger" size="sm" onClick={stop}>
                  Стоп
                </Button>
              ) : (
                <Button type="submit" size="sm" disabled={!draft.trim()}>
                  Спросить
                  <IconSend className="size-3.5" />
                </Button>
              )}
            </div>
          </div>
        </div>
      </form>
    </section>
  )
}
