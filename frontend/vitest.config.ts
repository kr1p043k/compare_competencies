import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    css: false,
    restoreMocks: true,
    // Пулы worker-потоков на этой машине падают с
    // "Zone Allocation failed - process out of memory" даже на одном файле.
    // Один форк стабилен и для 9 тестов, и для всего набора.
    pool: "forks",
    poolOptions: {
      forks: { singleFork: true },
    },
    maxWorkers: 1,
  },
});