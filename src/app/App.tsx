import { AppProviders } from './providers/AppProviders'
import { AppRouter } from './router/AppRouter'
import { AppShell } from '@/widgets/app-shell'
import { ChatThread } from '@/widgets/chat-thread'

/**
 * Здесь два виджета складываются в один экран.
 *
 * Оболочка не импортирует ленту диалога, а получает её отсюда. Складывать
 * виджеты друг с другом — работа слоя приложения; если бы это делала сама
 * оболочка, вынуть ассистента или показать другой правый блок стало бы
 * невозможно без правки оболочки.
 */
export function App() {
  return (
    <AppProviders>
      <AppShell aside={<ChatThread />}>
        <AppRouter />
      </AppShell>
    </AppProviders>
  )
}
