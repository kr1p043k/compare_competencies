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
            if (id.includes('node_modules/@radix-ui')) {
              return 'vendor-radix';
            }
          }
        }
      },
    },

    assetsInclude: ['**/*.svg', '**/*.csv'],
  }
})