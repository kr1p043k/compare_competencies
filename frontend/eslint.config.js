import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";

/**
 * Flat-конфиг ESLint: базовые правила + typescript-eslint + react-hooks.
 *
 * Подключён после ручных правок гонок запросов (c792d4c): правила хуков
 * ловят стейл-замыкания и пропущенные зависимости эффектов — тот класс
 * багов, который мы только что чинили руками.
 *
 * no-explicit-any пока выключен: в коде ~142 any — типизация идёт
 * отдельным этапом,lint не должен блокировать CI на текущем состоянии.
 */
export default tseslint.config(
  { ignores: ["dist", "node_modules", "coverage"] },
  { languageOptions: { globals: { ...globals.browser } } },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["**/*.{ts,tsx}"],
    plugins: { "react-hooks": reactHooks },
    rules: {
      // Только проверенные временем правила хуков. Новые компиляторные
      // (set-state-in-effect, immutability, purity) на текущем коде дают
      // десятки срабатываний — подключать их будем по мере рефакторинга.
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
    },
  },
  {
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      // _префикс — осознанные заглушки позиционных аргументов/деструктуризации
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_", caughtErrors: "none" },
      ],
    },
  },
);
