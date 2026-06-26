import { defineConfig, loadEnv } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

function figmaAssetResolver() {
  return {
    name: 'figma-asset-resolver',
    resolveId(id) {
      if (id.startsWith('figma:asset/')) {
        const filename = id.replace('figma:asset/', '')
        return path.resolve(__dirname, 'src/assets', filename)
      }
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  
  return {
    plugins: [
      figmaAssetResolver(),
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
      // Убираем manualChunks - все в один файл
      rollupOptions: {
        output: {
          manualChunks: undefined
        }
      },
    },

    assetsInclude: ['**/*.svg', '**/*.csv'],
  }
})