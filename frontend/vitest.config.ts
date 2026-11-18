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
    // Гейт для core-скоупа (lib + design): 80% по всем метрикам (бэклог 7.5.4).
    // Замер 2026-11-19 (vitest --coverage, v8):
    //   stmts 98.15 | branch 93.57 | funcs 97.29 | lines 98.15.
    // Поднимать числа можно только вместе с ростом реального покрытия.
    coverage: {
      provider: "v8",
      reporter: ["text", "html"],
      // Core = lib + design (see backlog 7.5.4). App components and shadcn ui
      // are out of scope: gating them at 80% would demand brittle DOM tests.
      include: ["src/lib/**/*.{ts,tsx}", "src/design/**/*.{ts,tsx}"],
      thresholds: {
        statements: 80,
        branches: 80,
        functions: 80,
        lines: 80,
      },
    },
  },
});
