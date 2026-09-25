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
          <div className="space-y-4">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">
                Спросите о чём угодно из вики
              </h2>
              <p className="mt-1 text-sm leading-relaxed text-slate-600">
                Ассистент отвечает только по документам вики и обязан ссылаться на фрагменты.
                Если ответа в документации нет, он скажет это прямо, а не придумает.
              </p>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => submit(example)}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-left text-xs text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
                >
                  <IconSparkle className="size-3.5 shrink-0 text-sky-500" />
                  <span>{example}</span>
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
        className="shrink-0 border-t border-slate-200 bg-white p-3"
        onSubmit={(event) => {
          event.preventDefault()
          submit(draft)
        }}
      >
        <div className="rounded-xl border border-slate-200 bg-white focus-within:border-slate-400">
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
            className="min-h-16 w-full resize-y rounded-xl px-3 py-2.5 text-sm outline-none"
          />
          <div className="flex items-center justify-between gap-2 px-2 pb-2">
            <span className="pl-1 text-xs text-slate-400">Enter — отправить</span>
            <div className="flex items-center gap-2">
              {exchanges.length > 0 && !streaming ? (
                <Button variant="ghost" onClick={startNew}>
                  Новый диалог
                </Button>
              ) : null}
              {/* Кнопка «Стоп» доступна ровно в фазе потока — это следствие
                  явного автомата состояний, а не отдельного флага isLoading. */}
              {streaming ? (
                <Button variant="danger" onClick={stop}>
                  Стоп
                </Button>
              ) : (
                <Button type="submit" disabled={!draft.trim()}>
                  <span className="flex items-center gap-1.5">
                    Спросить
                    <IconSend className="size-4" />
                  </span>
                </Button>
              )}
            </div>
          </div>
        </div>
      </form>
    </section>
  )
}
