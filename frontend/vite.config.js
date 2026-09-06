import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// During development, proxy API + viewer calls to the Modal backend so the
// browser talks to a same-origin URL (no CORS, cookies/headers just work).
// Override the target with VITE_BACKEND_URL when pointing at a deployed URL:
//   VITE_BACKEND_URL=https://your-modal-url.modal.run npm run dev
const backend = process.env.VITE_BACKEND_URL || 'http://localhost:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/generate-3d': { target: backend, changeOrigin: true },
      '/api': { target: backend, changeOrigin: true },
    },
  },
})
