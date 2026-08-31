import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// Where this dev server listens, and where it forwards /api. `make ui` sets all three
// from the same variables the API is launched with (UI_HOST, UI_PORT, API_URL), so moving
// the backend does not leave the dev server proxying to a port nothing is on.
const UI_HOST = process.env.TRAINLAB_UI_HOST || "127.0.0.1";
const UI_PORT = Number(process.env.TRAINLAB_UI_PORT) || 5173;
const API_URL = process.env.TRAINLAB_API_URL || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: {
    host: UI_HOST,
    port: UI_PORT,
    // The API is the single origin for everything under /api, including the SSE stream.
    proxy: {
      "/api": {
        target: API_URL,
        changeOrigin: true,
        // Server-sent events must not be buffered by the dev proxy.
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            if (proxyRes.headers["content-type"]?.includes("text/event-stream")) {
              proxyRes.headers["cache-control"] = "no-cache";
            }
          });
        },
      },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
