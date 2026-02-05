import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: 'localhost',
    port: 6010,
    open: true,
    proxy: {
      '/api': {
        target: 'http://localhost:9010',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:9010',
        ws: true,
      },
    },
  },
  resolve: {
    alias: {
      '@': '/src',
    },
  },
});