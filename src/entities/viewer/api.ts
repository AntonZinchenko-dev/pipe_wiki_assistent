import { useQuery } from '@tanstack/react-query'
import { fetchViewer } from '@/shared/api/client'

/**
 * Кто сейчас пользуется системой и что ему можно.
 *
 * Права спрашиваем у сервера и не выводим на клиенте: браузер — не место,
 * где решают, что человеку положено. Здесь они нужны ровно для одного —
 * не показывать кнопку, которая всё равно вернёт отказ. Само ограничение
 * держит сервер.
 */
export function useViewer() {
  return useQuery({
    queryKey: ['viewer'],
    queryFn: ({ signal }) => fetchViewer(signal),
    staleTime: 5 * 60_000,
  })
}
