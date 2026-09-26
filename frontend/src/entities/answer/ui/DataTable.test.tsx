import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { Dataset } from '@/shared/api/contracts'
import { DataTable } from './DataTable'

function table(over: Partial<Dataset> = {}): Dataset {
  return {
    handle: 'TAB01',
    kind: 'pipes.fleet',
    title: 'Топ-5 труб по выработке ресурса',
    columns: [
      { key: 'pipe_id', title: 'Труба', unit: '', kind: 'text' },
      { key: 'damage_percent', title: 'Выработка', unit: '%', kind: 'number' },
    ],
    rows: [
      { pipe_id: 'PP-0035', damage_percent: 97.7 },
      { pipe_id: 'PP-0028', damage_percent: 87 },
    ],
    total_found: 40,
    scanned: 40,
    offset: 0,
    truncated: true,
    hint: '',
    taken_at: '25.09.2026 12:00',
    skipped: [],
    next_cursor: '',
    error: null,
    ...over,
  }
}

describe('таблица из инструмента', () => {
  it('показывает значения дословно, как их отдал сервис', () => {
    render(<DataTable data={table()} />)

    expect(screen.getByText('PP-0035')).toBeInTheDocument()
    // Разряды разделяем по-русски, само значение не трогаем.
    expect(screen.getByText('97,7')).toBeInTheDocument()
  })

  it('говорит, сколько нашлось, а не только сколько показано', () => {
    // Это первым терялось в пересказе: «топ-5» читалось как «в парке
    // пять труб», и человек делал вывод по выборке, приняв её за парк.
    render(<DataTable data={table()} />)

    expect(screen.getByText(/строк: 2 из 40/)).toBeInTheDocument()
  })

  it('называет поимённо строки, по которым данных нет', () => {
    // «Две трубы за порогом» и «две за порогом, по одной расчёт устарел»
    // — разные ответы, и второй честный.
    render(<DataTable data={table({ skipped: ['PP-0007 (E-1042)'] })} />)

    expect(screen.getByText(/PP-0007/)).toBeInTheDocument()
  })

  it('различает временный отказ и постоянный', () => {
    // До этого оба выглядели как «произошла ошибка», и человек шёл
    // пробовать снова там, где пробовать бессмысленно.
    const { unmount } = render(
      <DataTable
        data={table({
          rows: [],
          error: {
            code: 'E-1042', message: 'Расчёт устарел.',
            retriable: true, retry_after_s: 60, hint: '',
          },
        })}
      />,
    )
    expect(screen.getByText(/через 60 с/)).toBeInTheDocument()
    unmount()

    render(
      <DataTable
        data={table({
          rows: [],
          error: {
            code: 'E-1108', message: 'Трубы нет в парке.',
            retriable: false, retry_after_s: null, hint: '',
          },
        })}
      />,
    )
    expect(screen.getByText(/повтор того же запроса/)).toBeInTheDocument()
  })

  it('пустой ответ — это не отказ', () => {
    // Пустая таблица говорит «таких труб нет», отказ говорит «мы не
    // смогли посмотреть». Выдавать второе за первое нельзя.
    render(<DataTable data={table({ rows: [], total_found: 0, hint: 'Ни одной трубы не нашлось.' })} />)

    expect(screen.getByText('Ни одной трубы не нашлось.')).toBeInTheDocument()
  })
})

describe('вторая страница', () => {
  it('называет номера строк, а не только их количество', () => {
    // «5 из 39» на второй странице не говорит, КАКИЕ пять, и читается
    // как начало списка.
    render(<DataTable data={table({ offset: 5, total_found: 39 })} />)

    expect(screen.getByText(/строки 6–7 из 39/)).toBeInTheDocument()
  })
})
