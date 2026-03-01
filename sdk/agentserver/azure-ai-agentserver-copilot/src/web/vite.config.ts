import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: './dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/responses': 'http://localhost:8088',
      '/liveness': 'http://localhost:8088',
      '/readiness': 'http://localhost:8088',
    },
  },
})
