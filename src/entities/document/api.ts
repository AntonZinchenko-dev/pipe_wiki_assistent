import { useQuery } from '@tanstack/react-query'
import { fetchDocument, fetchDocuments, fetchHealth } from '@/shared/api/client'

export function useDocuments() {
  return useQuery({ queryKey: ['documents'], queryFn: ({ signal }) => fetchDocuments(signal) })
}

export function useDocument(docId: string | undefined) {
  return useQuery({
    queryKey: ['document', docId],
    queryFn: ({ signal }) => fetchDocument(docId as string, signal),
    enabled: Boolean(docId),
  })
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: ({ signal }) => fetchHealth(signal),
    // Здоровье индекса и провайдера меняется само по себе, поэтому обновляем
    // фоном: пустой индекс или упавшая Ollama должны быть видны сразу.
    refetchInterval: 30_000,
  })
}
