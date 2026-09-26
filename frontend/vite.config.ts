import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(__dirname, 'src') } },
  server: {
    port: 5174,
    // Прокси на бэкенд, а не CORS-запросы из браузера напрямую: ключ модели
    // живёт на сервере, и фронт не должен знать ни его, ни адрес провайдера.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // Для потока обязательно: иначе прокси буферизует ответ и стриминг
        // «работает локально, ломается в проде» ровно наоборот — ломается
        // локально, потому что буферизует dev-прокси.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['cache-control'] = 'no-cache, no-transform'
          })
        },
      },
    },
  },
})
