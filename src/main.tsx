import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './app/App'
import { applyTheme, readTheme } from './shared/lib/theme'
import './app/styles/index.css'

// Тему ставим ДО первой отрисовки, а не в эффекте компонента.
//
// Иначе тёмный интерфейс на долю секунды показывается светлым — та самая
// белая вспышка, из-за которой тёмную тему в половине приложений неприятно
// включать. Стоит одна строчка, а ощущается как качество сборки.
applyTheme(readTheme())

const root = document.getElementById('root')
if (!root) throw new Error('нет элемента #root в index.html')

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
