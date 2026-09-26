import { defineConfig } from 'steiger'
import fsd from '@feature-sliced/steiger-plugin'

export default defineConfig([
  ...fsd.configs.recommended,
  {
    files: ['./src/shared/**'],
    rules: { 'fsd/public-api': 'off', 'fsd/no-public-api-sidestep': 'off' },
  },
  {
    // Виджеты и страницы здесь — композиция; единичные слайсы на старте
    // нормальны и дробить их ради линтера смысла нет.
    rules: { 'fsd/insignificant-slice': 'off', 'fsd/repetitive-naming': 'off' },
  },
])
