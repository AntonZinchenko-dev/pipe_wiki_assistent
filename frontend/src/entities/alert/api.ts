import { useQuery } from '@tanstack/react-query'
import { fetchAlerts } from '@/shared/api/client'

/**
 * Состояние порогов оповещения.
 *
 * Обновляем чаще, чем список моделей: смысл алерта в том, чтобы человек
 * увидел проблему раньше жалобы. Индикатор, отстающий на пять минут, —
 * это индикатор, по которому узнаёшь об аварии последним.
 */
export function useAlerts() {
  return useQuery({
    queryKey: ['alerts'],
    queryFn: ({ signal }) => fetchAlerts(signal),
    refetchInterval: 20_000,
    refetchOnWindowFocus: true,
  })
}
