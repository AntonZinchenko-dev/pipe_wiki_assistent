import { useQuery } from '@tanstack/react-query'
import { fetchModels } from '@/shared/api/client'

/**
 * Список моделей, доступных прямо сейчас.
 *
 * Обновляем нечасто: набор установленных моделей меняется руками, а не сам
 * по себе. Но обновляем: человек может поставить новую модель в Ollama, не
 * перезагружая вкладку, и не найти её в списке — это выглядело бы как
 * «интерфейс не видит модель», хотя дело в устаревшем кэше.
 */
export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: ({ signal }) => fetchModels(signal),
    staleTime: 60_000,
    refetchInterval: 120_000,
  })
}
