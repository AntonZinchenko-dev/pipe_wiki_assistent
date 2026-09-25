import { Navigate, Route, Routes } from 'react-router-dom'
import { DocumentPage } from '@/pages/document'
import { WikiPage } from '@/pages/wiki'

export function AppRouter() {
  return (
    <Routes>
      {/* Маршрутов ровно столько, сколько РАЗНЫХ экранов. Ассистент своего
          маршрута не имеет: он живёт в панели, доступен отовсюду и
          разворачивается на весь экран, не меняя адреса. Это не мелочь —
          именно поэтому открытый документ остаётся открытым, пока ты
          споришь с моделью, и возврат к нему стоит один клик.

          `/assistant` оставлен перенаправлением: адрес мог попасть в
          закладки, и 404 вместо приложения — плохая награда за это. */}
      <Route path="/" element={<Navigate to="/wiki" replace />} />
      <Route path="/wiki" element={<WikiPage />} />
      <Route path="/wiki/:docId" element={<DocumentPage />} />
      <Route path="/assistant" element={<Navigate to="/wiki" replace />} />
      <Route path="*" element={<Navigate to="/wiki" replace />} />
    </Routes>
  )
}