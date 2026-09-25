import type { ReactNode } from 'react'
import { useEffect, useState } from 'react'
import { IconClose, IconPanel, IconSparkle } from '@/shared/ui/icons'
import { cn } from '@/shared/lib/cn'
import { SideNav } from './SideNav'
import { TopBar } from './TopBar'

/**
 * Оболочка приложения: шапка, меню слева, содержимое, ассистент справа.
 *
 * У АССИСТЕНТА ОДНО МЕСТО, А НЕ ДВА.
 *
 * Раньше их было два: правая панель и отдельный раздел `/assistant`. На
 * широком экране это давало буквально два окна в одну переписку — оба
 * читают одно состояние, поэтому вопрос, заданный в одном, появлялся в
 * обоих. Разделу было оправдание: на узком экране панель не помещалась и
 * пряталась, а остаться без ассистента там нельзя.
 *
 * Оправдание было настоящим, а решение неправильным. Узкому экрану нужна
 * не вторая копия, а другое поведение той же панели: там она накрывает
 * содержимое, а не отбирает у него колонку. Это одна строчка в классах и
 * ни одного лишнего маршрута.
 *
 * Отсюда три состояния вместо булева «открыто / закрыто»:
 *
 *   closed — панели нет, всё место у документа;
 *   side   — колонка справа (на узком экране — поверх содержимого);
 *   full   — ассистент занимает всю рабочую область.
 *
 * `full` заменил собой тот самый раздел `/assistant` и делает это лучше:
 * маршрут не меняется, значит открытый документ остаётся открытым, и
 * возврат к нему — это не «найти его заново», а один клик.
 *
 * Содержимое при этом НЕ размонтируется, а прячется классом. Разница
 * заметна сразу: иначе разворот ассистента на весь экран сбрасывал бы
 * прокрутку документа, и после сворачивания человек оказывался бы в его
 * начале.
 *
 * Ассистент приходит сюда СНАРУЖИ, отдельным свойством, а не импортируется
 * внутри. Оболочка и лента диалога — два виджета одного слоя, и прямой
 * импорт между ними связал бы их насмерть. Кто кого складывает вместе,
 * решает слой выше — приложение. Ровно поэтому и страницы приходят сюда
 * как `children`.
 */

type Panel = 'closed' | 'side' | 'full'

/**
 * Состояние панели переживает перезагрузку.
 *
 * Мелочь, которая заметна каждый день: человек, работающий с документами,
 * закрывает панель один раз, а не каждое утро. Хранится в localStorage, а
 * не на сервере, — это настройка рабочего места, а не пользователя.
 */
const PANEL_KEY = 'pipewiki.panel'

function readPanel(): Panel {
  try {
    const saved = localStorage.getItem(PANEL_KEY)
    if (saved === 'closed' || saved === 'side' || saved === 'full') return saved
  } catch {
    // Приватное окно или запрет хранилища. Не повод падать: берём
    // значение по умолчанию и работаем дальше.
  }
  return 'side'
}

export function AppShell({ children, aside }: { children: ReactNode; aside: ReactNode }) {
  const [panel, setPanel] = useState<Panel>(readPanel)

  useEffect(() => {
    try {
      localStorage.setItem(PANEL_KEY, panel)
    } catch {
      // см. readPanel
    }
  }, [panel])

  return (
    <div className="flex h-dvh flex-col bg-canvas text-ink">
      <TopBar
        assistantOpen={panel !== 'closed'}
        onToggleAssistant={() => setPanel((value) => (value === 'closed' ? 'side' : 'closed'))}
      />

      {/* relative нужен узкому экрану: там панель ложится поверх
          содержимого, и ей нужна точка отсчёта. */}
      <div className="relative flex min-h-0 flex-1">
        <SideNav onAskAssistant={() => setPanel((value) => (value === 'closed' ? 'side' : value))} />

        <main
          className={cn(
            'min-w-0 flex-1 overflow-y-auto',
            // Прячем, а не размонтируем: прокрутка документа должна
            // пережить разворот ассистента.
            panel === 'full' && 'hidden',
          )}
        >
          <div className="mx-auto w-full max-w-4xl px-6 py-7">{children}</div>
        </main>

        {panel !== 'closed' ? (
          <section
            className={cn(
              'flex min-h-0 flex-col border-l border-line bg-chrome',
              panel === 'full'
                ? 'flex-1'
                : // Узкий экран: поверх содержимого. Широкий: своя колонка.
                  'absolute inset-y-0 right-0 z-20 w-full max-w-[440px] shadow-panel xl:static xl:w-[440px] xl:shrink-0 xl:shadow-none',
            )}
          >
            <div className="flex h-12 shrink-0 items-center gap-2 border-b border-line bg-chrome px-4">
              <IconSparkle className="size-4 text-accent" />
              <span className="text-sm font-semibold text-ink">Ассистент</span>
              {/* Пояснение — служебным тоном, а не зелёной плашкой.
                  Зелёный означает «работает», а тут просто сказано, по
                  чему ассистент отвечает. Плашка того же цвета, что и
                  «провайдер на связи», заставляла читать её как состояние. */}
              <span className="hidden text-xs text-ink-faint sm:inline">по документам вики</span>

              <div className="ml-auto flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => setPanel(panel === 'full' ? 'side' : 'full')}
                  aria-label={panel === 'full' ? 'свернуть в панель' : 'развернуть на весь экран'}
                  title={panel === 'full' ? 'свернуть в панель' : 'развернуть на весь экран'}
                  className="flex size-7 items-center justify-center rounded-md text-ink-faint transition hover:bg-sunken hover:text-ink"
                >
                  <IconPanel className="size-4" />
                </button>
                <button
                  type="button"
                  onClick={() => setPanel('closed')}
                  aria-label="закрыть панель ассистента"
                  className="flex size-7 items-center justify-center rounded-md text-ink-faint transition hover:bg-sunken hover:text-ink"
                >
                  <IconClose className="size-4" />
                </button>
              </div>
            </div>
            {aside}
          </section>
        ) : null}
      </div>
    </div>
  )
}
