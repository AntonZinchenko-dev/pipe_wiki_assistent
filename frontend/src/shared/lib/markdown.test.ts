import { describe, expect, it } from 'vitest'
import { parseAnswer, parseInline, type Inline } from './markdown'

/**
 * Разбор ответа модели — это граница безопасности, а не украшательство.
 *
 * Вывод модели собран из документов, которые пишут люди с доступом к вики,
 * и попадает прямиком на экран. Проверяется здесь поэтому не «красиво ли
 * получилось», а одно свойство: НА ВЫХОДЕ НЕ БЫВАЕТ НИЧЕГО, КРОМЕ ПЯТИ
 * ИЗВЕСТНЫХ ВИДОВ УЗЛОВ. Ни разметки, ни ссылок, ни атрибутов.
 *
 * Это и есть разница между белым списком по построению и санитайзером:
 * санитайзер надо пробить, найдя дырку в правилах, а это — дописав сюда
 * новый вид узла.
 */

const KINDS = new Set(['text', 'bold', 'italic', 'code', 'citation'])

function flatten(text: string): Inline[] {
  return parseAnswer(text).flatMap((block) =>
    block.kind === 'paragraph' ? block.content : block.items.flat(),
  )
}

describe('чего на выходе быть не может', () => {
  it('размётка из ответа остаётся текстом', () => {
    const nasty = '<script>alert(1)</script> и <img src=x onerror=alert(1)>'
    const nodes = flatten(nasty)

    expect(nodes.every((node) => KINDS.has(node.kind))).toBe(true)
    // Текст сохранён дословно — мы его не вырезаем, а просто никогда не
    // превращаем в разметку. Вырезание было бы хуже: человек не увидел бы,
    // что в документе лежит такое.
    const joined = nodes.map((node) => ('text' in node ? node.text : '')).join('')
    expect(joined).toContain('<script>')
  })

  it('ссылка markdown не становится ссылкой', () => {
    // Самый опасный случай: адрес в ответе модели взят из документа вики.
    // Кликабельная ссылка оттуда уводит пользователя куда угодно с видом
    // «так написано в регламенте».
    const nodes = flatten('смотри [тут](javascript:alert(1)) подробнее')

    expect(nodes.every((node) => KINDS.has(node.kind))).toBe(true)
    const joined = nodes.map((node) => ('text' in node ? node.text : '')).join('')
    expect(joined).toContain('javascript:alert(1)')
    // Ровно потому, что вида узла «ссылка» не существует, — а не потому,
    // что мы отфильтровали именно эту схему.
    expect(nodes.some((node) => node.kind === 'citation')).toBe(false)
  })

  it('внутри кода ничего не разбирается', () => {
    const nodes = parseInline('значение `**не жирное** [4]` дальше')

    expect(nodes[1]).toEqual({ kind: 'code', text: '**не жирное** [4]' })
    expect(nodes.some((node) => node.kind === 'citation')).toBe(false)
  })
})

describe('то, ради чего это затевалось', () => {
  it('маркированный список становится списком', () => {
    const blocks = parseAnswer('Порядок такой:\n- сначала одно\n- потом другое')

    expect(blocks[0].kind).toBe('paragraph')
    expect(blocks[1]).toMatchObject({ kind: 'list', ordered: false })
    expect(blocks[1].kind === 'list' && blocks[1].items).toHaveLength(2)
  })

  it('нумерованный список отличается от маркированного', () => {
    const blocks = parseAnswer('1. первое\n2. второе')
    expect(blocks[0]).toMatchObject({ kind: 'list', ordered: true })
  })

  it('два разных списка подряд не слипаются', () => {
    const blocks = parseAnswer('- один\n1. два')
    expect(blocks).toHaveLength(2)
    expect(blocks[0]).toMatchObject({ ordered: false })
    expect(blocks[1]).toMatchObject({ ordered: true })
  })

  it('ссылка на фрагмент остаётся отдельным узлом', () => {
    // По этому узлу интерфейс строит переход в документ — адрес он берёт
    // из метаданных поиска, а отсюда только число.
    const nodes = parseInline('порог 80 % [4] по регламенту')
    expect(nodes).toContainEqual({ kind: 'citation', number: 4 })
  })

  it('жирный не путается с курсивом', () => {
    expect(parseInline('**важно** и *слегка*')).toEqual([
      { kind: 'bold', text: 'важно' },
      { kind: 'text', text: ' и ' },
      { kind: 'italic', text: 'слегка' },
    ])
  })

  it('незакрытая звёздочка остаётся звёздочкой', () => {
    // Модель обрывается на лимите токенов регулярно. Ответ, обрезанный на
    // полуслове, не должен ломать разбор.
    const nodes = parseInline('начал **жирный и не закр')
    expect(nodes).toEqual([{ kind: 'text', text: 'начал **жирный и не закр' }])
  })

  it('пустой ответ не роняет разбор', () => {
    expect(parseAnswer('')).toEqual([])
    expect(parseAnswer('\n\n')).toEqual([])
  })
})

describe('граница держится кодом, а не обещанием в комментарии', () => {
  it('вставки разметки нет ни в одном файле интерфейса', async () => {
    // Утверждение «мы не вставляем HTML» стоит ровно столько, сколько
    // стоит привычка его соблюдать. Через полгода кто-нибудь добавит
    // «временно, только для одного блока» — и белый список по построению
    // перестанет быть по построению, причём молча.
    //
    // Ищем именно ИСПОЛЬЗОВАНИЕ (атрибут со знаком равенства), а не
    // упоминание: слово встречается в комментариях, объясняющих, почему
    // его здесь нет.
    const fs = await import('node:fs/promises')
    const path = await import('node:path')

    async function walk(dir: string): Promise<string[]> {
      const entries = await fs.readdir(dir, { withFileTypes: true })
      const found: string[] = []
      for (const entry of entries) {
        const full = path.join(dir, entry.name)
        if (entry.isDirectory()) found.push(...(await walk(full)))
        else if (/\.tsx?$/.test(entry.name)) found.push(full)
      }
      return found
    }

    const files = await walk(path.resolve(__dirname, '..', '..'))
    const guilty: string[] = []
    for (const file of files) {
      const source = await fs.readFile(file, 'utf-8')
      if (/dangerouslySetInnerHTML\s*=/.test(source)) guilty.push(file)
    }

    expect(guilty).toEqual([])
    // Заодно убеждаемся, что обход вообще что-то нашёл: тест, который
    // ничего не просмотрел, проходит всегда.
    expect(files.length).toBeGreaterThan(10)
  })
})

describe('несколько номеров одной скобкой', () => {
  it('«[1, 2]» — это две ссылки, а не текст', () => {
    // Из живого прогона: модель свела два источника в одну скобку, и весь
    // ответ проехал мимо — скобка осталась обычным текстом, ссылки не
    // кликались, а проверка на сервере сочла ответ вовсе без источников и
    // переписала статус на «в документации нет ответа».
    const nodes = parseInline('независимо от замеров [1, 2].')

    expect(nodes.filter((node) => node.kind === 'citation')).toEqual([
      { kind: 'citation', number: 1 },
      { kind: 'citation', number: 2 },
    ])
  })

  it('точка с запятой считается так же', () => {
    const nodes = parseInline('по регламенту [3;4]')

    expect(nodes.filter((node) => node.kind === 'citation')).toHaveLength(2)
  })

  it('одиночная ссылка работает как работала', () => {
    expect(parseInline('порог 80 % [2].')).toContainEqual({ kind: 'citation', number: 2 })
  })

  it('перечисление без скобок ссылкой не становится', () => {
    // «1, 2» в тексте — это просто числа. Ссылкой делает скобка.
    expect(parseInline('строки 1, 2 и 3').some((node) => node.kind === 'citation')).toBe(false)
  })
})
