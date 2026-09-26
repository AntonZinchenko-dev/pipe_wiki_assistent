import { create } from 'zustand'
import type { ModelChoice } from '@/shared/api/contracts'

/**
 * Выбранная человеком модель.
 *
 * Почему это состояние живёт в сущности, а не в фиче выбора. Читает его
 * фича вопроса (ей надо знать, кого спрашивать), а пишет фича выбора. Две
 * фичи не имеют права ходить друг к другу — иначе через полгода получится
 * клубок, где нельзя удалить ни одну. Общее у них — понятие «модель», и
 * место общего понятия здесь.
 *
 * `null` означает «как настроено на сервере», и это НЕ то же самое, что
 * «первая в списке»: настройка сервера может смениться, и человек, ничего
 * не выбиравший, должен поехать вместе с ней, а не остаться на модели,
 * которая когда-то была первой.
 */
type ModelChoiceState = {
  chosen: { provider: string; model: string } | null
  choose: (choice: ModelChoice | null) => void
}

export const useModelChoice = create<ModelChoiceState>((set) => ({
  chosen: null,
  choose: (choice) =>
    set({ chosen: choice ? { provider: choice.provider, model: choice.model } : null }),
}))

/** Локальная ли модель. От этого зависит показ стоимости и предупреждений. */
export function isLocal(provider: string): boolean {
  return provider === 'ollama'
}

/** Короткая подпись для шапки: без длинных суффиксов вроде `:latest`. */
export function shortLabel(model: string): string {
  return model.replace(/:latest$/, '')
}

/**
 * Агентский режим: разрешаем ли модели дособирать контекст инструментами.
 *
 * Живёт рядом с выбором модели по той же причине, по которой там живёт сам
 * выбор: пишет этот флаг переключатель в шапке, а читает отправка вопроса —
 * две разные фичи. Ходить друг к другу они не имеют права, и общее у них —
 * понятие «чем и как отвечаем».
 *
 * Выключен по умолчанию сознательно. Агентский режим — это лишние вызовы
 * модели и лишние секунды на каждый вопрос; включать его молча значит
 * менять систему за спиной у человека.
 */
type AgentModeState = {
  on: boolean
  toggle: () => void
}

export const useAgentMode = create<AgentModeState>((set) => ({
  on: false,
  toggle: () => set((state) => ({ on: !state.on })),
}))
