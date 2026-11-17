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
    // Пороги равны текущему измеренному покрытию (округление вниз до целого),
    // чтобы гейт ловил деградацию, но не падал на зелёной базе.
    // Замер 2026-11-16 (vitest --coverage, v8):
    //   stmts 44.98 | branch 73.07 | funcs 52.84 | lines 44.98.
    // Поднимать числа можно только вместе с ростом реального покрытия.
    coverage: {
      provider: "v8",
      reporter: ["text", "html"],
      thresholds: {
        statements: 44,
        branches: 73,
        functions: 52,
        lines: 44,
      },
    },
  },
});
