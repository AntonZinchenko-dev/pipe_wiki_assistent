/**
 * Подключает матчеры jest-dom (`toBeInTheDocument` и родню).
 *
 * Отдельным файлом, а не импортом в каждом тесте: забытый импорт даёт
 * ошибку вида «expect(...).toBeInTheDocument is not a function» в одном
 * случайном тесте, и искать её приходится не там, где она есть.
 */
import '@testing-library/jest-dom/vitest'
