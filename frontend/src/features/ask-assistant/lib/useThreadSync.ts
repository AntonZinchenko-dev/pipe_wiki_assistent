import { useEffect } from 'react'
import { useThreads } from '@/entities/thread'
import { useAssistant } from '../model'

/**
 * Связывает выбор диалога в меню с лентой на экране.
 *
 * Связь однонаправленная и живёт в ОДНОМ месте. Соблазн был сделать иначе:
 * пусть список диалогов сам просит ленту перезагрузиться. Но список — это
 * сущность, лента — возможность, и сущность, дёргающая возможность,
 * переворачивает зависимости вверх ногами: дальше список нельзя ни
 * переиспользовать, ни протестировать без всей машинерии запроса.
 *
 * Здесь же делается первое чтение базы. Не в модуле при импорте: чтение
 * асинхронное, а состояние «ещё не читали» и «прочитали, пусто» — разные,
 * и путать их нельзя, иначе на долю секунды после загрузки страницы
 * человек видит «диалогов нет» вместо своих диалогов.
 */
export function useThreadSync(): void {
  const ready = useThreads((state) => state.ready)
  const currentId = useThreads((state) => state.currentId)
  const load = useThreads((state) => state.load)
  const openThread = useAssistant((state) => state.openThread)

  useEffect(() => {
    if (!ready) void load()
  }, [ready, load])

  useEffect(() => {
    // Пока список не прочитан, `currentId` равен null просто потому, что
    // мы ещё не знаем ответа. Открывать по этому пустую ленту рано.
    if (!ready) return
    void openThread(currentId)
  }, [ready, currentId, openThread])
}
