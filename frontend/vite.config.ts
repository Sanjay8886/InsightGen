// frontend/vite.config.ts

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // ------------------------------------
  // ADD THIS SECTION
  server: {
    // Railway containers often require listening on all interfaces
    host: '0.0.0.0', 
    // Add the specific Railway domain to allowed hosts
    allowedHosts: [
      'insightgen-production.up.railway.app'
      // You can also use a wildcard for all Railway domains (if supported by your Vite version)
      // '.up.railway.app' 
    ],
  },
  // ------------------------------------
});