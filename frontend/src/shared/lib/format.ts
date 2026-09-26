/** Форматирование величин, которые показываем человеку. */

export function ms(value: number | null | undefined): string {
  if (value == null) return '—'
  return value < 1000 ? `${Math.round(value)} мс` : `${(value / 1000).toFixed(1)} с`
}

export function tokens(prompt: number, completion: number): string {
  return `${prompt} + ${completion}`
}

/**
 * Близость показываем как есть, двумя знаками, и рядом всегда порог.
 *
 * Числового «процента уверенности модели» в интерфейсе нет сознательно:
 * модели систематически переоценивают себя, и «уверен на 87 %» — это ложная
 * точность, которая хуже её отсутствия. Косинус — другое дело: это измеренная
 * величина, у неё есть порог, и она объясняет решение системы.
 */
export function similarity(value: number | null | undefined): string {
  return value == null ? '—' : value.toFixed(2)
}

export function pages(from: number, to: number): string {
  return from === to ? `с. ${from}` : `с. ${from}–${to}`
}

/**
 * Стоимость запроса.
 *
 * Ноль у локальной модели — честный ноль: она бесплатная. Ноль у облака
 * означает другое — «тариф не задан, посчитать нечем», и показывать его той
 * же цифрой нельзя: аккуратная «0 ₽» рядом с реально потраченными токенами
 * читается как «бесплатно» и врёт про деньги.
 *
 * Поэтому решение принимается не по сумме, а по тому, локальная модель или
 * нет: у локальной — «бесплатно», у облачной без тарифа — прочерк.
 */
export function cost(value: number, isLocal: boolean): string {
  if (isLocal) return 'бесплатно (локальная модель)'
  if (!value) return '— (тариф не задан в PW_GIGACHAT_RUB_PER_1K)'
  return `${value.toFixed(2)} ₽`
}
