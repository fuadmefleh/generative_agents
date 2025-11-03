import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 6010,
    open: true,
    proxy: {
      '/api': {
        target: 'http://192.168.18.145:9010',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://192.168.18.145:9010',
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