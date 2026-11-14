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
    coverage: {
      provider: "v8",
      reporter: ["text-summary", "json-summary"],
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/**/*.test.{ts,tsx}", "src/**/*.a11y.test.tsx", "src/test/**", "src/**/*.d.ts"],
      // Базовый порог зафиксирован на фактическом значении, чтобы регресс
      // ловился, а не «позеленить» сборку. Поднимать по мере роста тестов (6.5.1).
      thresholds: {
        statements: 44,
        branches: 72,
        functions: 52,
        lines: 44,
      },
    },
  },
});