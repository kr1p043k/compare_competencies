import { defineConfig, loadEnv } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  
  return {
    plugins: [
      react(),
      tailwindcss(),
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    
    base: '/',
    
    server: {
      port: 3000,
      host: true,
      proxy: {
        '/api': {
          target: env.VITE_API_URL || 'http://localhost:8000',
          changeOrigin: true,
          secure: false,
          ws: true,
        },
        '/health': {
          target: env.VITE_API_URL || 'http://localhost:8000',
          changeOrigin: true,
          secure: false,
        },
      }
    },

    build: {
      outDir: 'dist',
      emptyOutDir: true,
      assetsDir: 'assets',
      sourcemap: false,
      // vendor-react отдельно: иначе shell чанк >500kB и варнинг не уходит
      // даже после lazy-сплита страниц (замер: shell+vendor 505kB raw).
      // motion и recharts весят больше всего в приложении, их выносим в
      // отдельные чанки: shell грузится быстрее, кэш живёт между релизами.
      // Хвостовой catch-all: прочие node_modules (clsx, tailwind-merge,
      // class-variance-authority, lucide-react, cmdk) собираются в `app`,
      // чтобы Rollup не «присваивал» общие утилиты vendor-чанкам:
      // раньше clsx (общая для index и recharts) попадала в vendor-charts,
      // и вся графика ехала в первый экран через modulepreload.
      rollupOptions: {
        output: {
          manualChunks: (id) => {
            if (id.includes('node_modules/react-dom') || id.includes('node_modules/react/') || id.includes('node_modules/scheduler')) {
              return 'vendor-react';
            }
            if (id.includes('node_modules/framer-motion') || id.includes('node_modules/motion')) {
              return 'vendor-motion';
            }
            if (id.includes('node_modules/recharts') || id.includes('node_modules/d3-') || id.includes('node_modules/victory-vendor')) {
              return 'vendor-charts';
            }
            // Рантайм recharts (redux-стек, es-toolkit, decimal.js-light и пр.)
            // нужен ТОЛЬКО графикам. Без этого правила он «проваливается»
            // в eager-чанк app (из-за общего clsx) и едет в первый экран.
            if (
              id.includes('node_modules/react-redux') ||
              id.includes('node_modules/@reduxjs') ||
              id.includes('node_modules/redux') ||
              id.includes('node_modules/reselect') ||
              id.includes('node_modules/immer') ||
              id.includes('node_modules/es-toolkit') ||
              id.includes('node_modules/decimal.js-light') ||
              id.includes('node_modules/eventemitter3') ||
              id.includes('node_modules/tiny-invariant') ||
              id.includes('node_modules/use-sync-external-store')
            ) {
              return 'vendor-charts';
            }
            if (id.includes('node_modules/@radix-ui')) {
              return 'vendor-radix';
            }
            if (id.includes('node_modules')) {
              return 'app';
            }
          }
        }
      },
    },

    assetsInclude: ['**/*.svg', '**/*.csv'],
  }
})