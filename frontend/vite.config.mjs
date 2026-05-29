import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return undefined;
          }
          if (id.includes('d3-')) {
            return 'd3';
          }
          if (id.includes('recharts')) {
            return 'recharts';
          }
          if (id.includes('@react-pdf')) {
            return 'pdf';
          }
          if (
            id.includes('react') ||
            id.includes('scheduler') ||
            id.includes('react-router')
          ) {
            return 'react-vendor';
          }
          return undefined;
        },
      },
    },
  },
  server: {
    port: 3000,
    strictPort: true,
    proxy: {
      // Dev convenience: avoid CORS by proxying API requests to the backend.
      // Frontend can call `/api/...` and Vite will forward to Django on :8000.
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/setupTests.js',
    globals: true,
  },
});
