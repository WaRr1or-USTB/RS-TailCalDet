import {defineConfig, loadEnv} from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const apiTargets = ["/health", "/system", "/methods", "/models", "/classes", "/runs"];

export default defineConfig(({mode}) => {
  const apiTarget = loadEnv(mode, ".", "").RS_CALVISION_API_TARGET || "http://127.0.0.1:8000";
  return {
  base: "./",
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      apiTargets.map((path) => [path, {target: apiTarget, changeOrigin: true}]),
    ),
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          viewer: ["openseadragon"],
          motion: ["motion"],
        },
      },
    },
  },
  };
});
