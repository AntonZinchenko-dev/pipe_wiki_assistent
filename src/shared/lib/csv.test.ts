import { describe, expect, it } from 'vitest'
import type { Dataset } from '@/shared/api/contracts'
import { datasetFileName, datasetToCsv } from './csv'

const data: Dataset = {
  handle: 'T98D4',
  kind: 'pipes.fleet',
  title: 'Топ-5 труб по выработке',
  columns: [
    { key: 'pipe_id', title: 'Труба', unit: '', kind: 'text' },
    { key: 'damage_percent', title: 'Выработка', unit: '%', kind: 'number' },
    { key: 'well', title: 'Скважина', unit: '', kind: 'text' },
  ],
  rows: [
    { pipe_id: 'PP-0035', damage_percent: 97.7, well: 'W-122' },
    { pipe_id: 'PP-0028', damage_percent: 87, well: null },
  ],
  total_found: 39, scanned: 40, offset: 0, truncated: true, hint: '',
  taken_at: '25.09.2026 13:28', skipped: [], next_cursor: '', error: null,
}

describe('datasetToCsv', () => {
  it('разделяет точкой с запятой, а дробные пишет через запятую', () => {
    // В русской локали запятая — десятичный знак. Запятая же разделителем
    // рвёт «97,7» на два столбца, и файл открывается кашей.
    const lines = datasetToCsv(data).split('\r\n')

    expect(lines[1]).toBe('PP-0035;97,7;W-122')
  })

  it('ставит BOM, иначе Excel показывает кракозябры вместо русского', () => {
    expect(datasetToCsv(data).startsWith('﻿')).toBe(true)
  })

  it('уносит единицу измерения в шапку, а не в каждую ячейку', () => {
    // «Выработка, %» — столбец чисел. «97,7 %» — столбец текста, по
    // которому не отсортировать и не посчитать среднее.
    expect(datasetToCsv(data).split('\r\n')[0]).toBe('﻿Труба;Выработка, %;Скважина')
  })

  it('пустое значение остаётся пустым, а не превращается в null', () => {
    expect(datasetToCsv(data).split('\r\n')[2]).toBe('PP-0028;87;')
  })

  it('экранирует значение, в котором есть разделитель или кавычка', () => {
    const tricky = { ...data, rows: [{ pipe_id: 'A;B', damage_percent: 1, well: 'он сказал "да"' }] }

    expect(datasetToCsv(tricky).split('\r\n')[1]).toBe('"A;B";1;"он сказал ""да"""')
  })

  it('выгружает ровно видимые колонки, а не всё, что прислал сервис', () => {
    // Человек не видел скрытых полей. Файл, в котором столбцов вдвое
    // больше таблицы, читается как чужой.
    const extra = { ...data, rows: [{ ...data.rows[0], steel: 'G-105', survey: 'SV-2026-08' }] }

    expect(datasetToCsv(extra)).not.toContain('G-105')
  })
})

describe('datasetFileName', () => {
  it('кладёт в имя метку таблицы', () => {
    // «топ-5 труб» человек выгрузит трижды за утро. Три одинаковых имени в
    // «Загрузках» — три файла, про которые непонятно, какой свежий.
    expect(datasetFileName(data)).toBe('Топ-5 труб по выработке T98D4.csv')
  })

  it('убирает символы, запрещённые в именах файлов', () => {
    const named = { ...data, title: 'Трубы: W-122/W-121 *срочно*' }

    expect(datasetFileName(named)).toBe('Трубы W-122 W-121 срочно T98D4.csv')
  })
})
