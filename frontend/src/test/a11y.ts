/**
 * a11y.ts - хелпер для автоматических проверок доступности (axe) в vitest.
 *
 * Зависимость: jest-axe. В frontend/package.json её пока нет; установка:
 *
 *   npm i -D jest-axe
 *
 * Почему jest-axe, а не vitest-axe: последний стабильный релиз vitest-axe
 * 0.1.0 вышел в октябре 2022 под peer "vitest ^0.16", дальше года идут
 * одни prerelease. jest-axe поддерживается активно, а его матчер берёт
 * форматирование из собственного jest-matcher-utils (не из this.utils
 * хост-раннера) и поэтому корректно регистрируется через expect.extend
 * в vitest.
 *
 * Пакет грузится лениво: до установки jest-axe (`npm i -D jest-axe`)
 * npx tsc --noEmit остаётся зелёным, а тесты падают с понятной
 * подсказкой, а не пропускаются молча. Это осознанный выбор: тихий skip
 * проверок доступности - это «зелёный» отчёт без настоящей проверки.
 */
import { expect } from "vitest";

/**
 * Спецификатор лежит в переменной типа string: иначе tsc резолвит пакет
 * статически и падает до установки зависимости (см. loadJestAxe).
 */
const JEST_AXE_MODULE: string = "jest-axe";

export interface A11yViolationNode {
  /** HTML-фрагмент узла с нарушением (попадает в вывод теста). */
  html: string;
  /** CSS-селекторы до узла. */
  target: string[];
  /** Человекочитаемое резюме нарушений узла. */
  failureSummary?: string;
}

export interface A11yViolation {
  /** id правила axe, например "image-alt". */
  id: string;
  /** "minor" | "moderate" | "serious" | "critical". */
  impact?: string | null;
  /** Краткое описание правила. */
  help: string;
  /** Ссылка на документацию правила. */
  helpUrl?: string;
  nodes: A11yViolationNode[];
}

/** Минимальная форма результатов axe, нужная матчеру toHaveNoViolations. */
export interface A11yAxeResults {
  violations: A11yViolation[];
}

type AxeRunner = (
  element: HTMLElement,
  options?: Record<string, unknown>,
) => Promise<A11yAxeResults>;

/** Контракт матчера expect.extend: { pass, message }. */
interface A11yMatcherResult {
  pass: boolean;
  message: () => string;
}

interface JestAxeModule {
  axe: AxeRunner;
  /** Объект матчера для expect.extend. */
  toHaveNoViolations: {
    toHaveNoViolations: (results: A11yAxeResults) => A11yMatcherResult;
  };
}

/**
 * Правила, бессмысленные при проверке изолированного компонента в jsdom.
 * Каждое выключение объясняется отдельно:
 *
 * - color-contrast: jsdom не загружает CSS-файлы и не строит layout,
 *   getComputedStyle пуст - реальный контраст посчитать нечем, правило
 *   даёт ложные срабатывания. Сам jest-axe тоже выключает категорию
 *   cat.color (README, "Checks that do not work in jest-axe"); здесь
 *   дублируем явно, чтобы набор правил читался в одном месте и не
 *   зависел от настроек версии jest-axe.
 *
 * - region: «весь контент страницы должен лежать в лендмарках» - правило
 *   уровня страницы, а мы проверяем изолированный фрагмент без main/nav;
 *   лендмарок у него и не должно быть. README jest-axe (раздел Testing
 *   isolated components) рекомендует выключать region ровно для этого
 *   случая.
 *
 * Всё остальное остаётся включённым: проверка должна оставаться честной.
 */
const JSDOM_AXE_RULES: Record<string, { enabled: boolean }> = {
  "color-contrast": { enabled: false },
  region: { enabled: false },
};

/**
 * Достаёт { axe, toHaveNoViolations } из загруженного модуля, допуская
 * интероп CJS/ESM: настоящий экспорт может лежать и в .default.
 * Форма проверяется в рантайме от unknown, не от типа пакета (его нет).
 * Возвращает null, если форма модуля неожиданная.
 */
function asJestAxeModule(raw: unknown): JestAxeModule | null {
  const sources: unknown[] = [
    raw,
    (raw as { default?: unknown } | null | undefined)?.default,
  ];
  for (const source of sources) {
    if (typeof source !== "object" || source === null) continue;
    const record = source as Record<string, unknown>;
    if (typeof record["axe"] !== "function") continue;

    const matcher = record["toHaveNoViolations"];
    if (typeof matcher !== "object" || matcher === null) continue;
    const matcherRecord = matcher as Record<string, unknown>;
    if (typeof matcherRecord["toHaveNoViolations"] !== "function") continue;

    return {
      axe: record["axe"] as AxeRunner,
      toHaveNoViolations: {
        toHaveNoViolations:
          matcherRecord["toHaveNoViolations"] as JestAxeModule["toHaveNoViolations"]["toHaveNoViolations"],
      },
    };
  }
  return null;
}

let cachedModule: JestAxeModule | null = null;

async function loadJestAxe(): Promise<JestAxeModule> {
  if (cachedModule) return cachedModule;

  let raw: unknown;
  try {
    // Спецификатор - переменная (см. JEST_AXE_MODULE): пакет резолвится
    // только в рантайме vitest.
    raw = await import(JEST_AXE_MODULE);
  } catch {
    throw new Error(
      "a11y: пакет jest-axe не установлен, проверку доступности выполнить " +
        "нельзя. Установите: npm i -D jest-axe. Тест падает, а не " +
        "пропускается: тихий skip a11y-проверок запрещён.",
    );
  }

  const mod = asJestAxeModule(raw);
  if (!mod) {
    throw new Error(
      "a11y: jest-axe загрузился, но экспорта axe/toHaveNoViolations не " +
        "найдено - проверьте целостность node_modules (npm ci).",
    );
  }

  // Матчер toHaveNoViolations регистрируется один раз на процесс.
  expect.extend(mod.toHaveNoViolations);
  cachedModule = mod;
  return mod;
}

/**
 * Запускает axe по отрендеренному фрагменту и падает при нарушениях.
 * Вывод каждой проблемы содержит impact, help и helpUrl правила,
 * HTML узла и CSS-селектор - этого хватает для локальной починки.
 *
 * @param container результат render(...) из @testing-library/react;
 *   для порталов передавайте baseElement.
 * @param axeOptions доп. опции axe, мержатся поверх дефолтов,
 *   например { rules: { "heading-order": { enabled: true } } }.
 */
export async function expectNoA11yViolations(
  container: HTMLElement,
  axeOptions: Record<string, unknown> = {},
): Promise<void> {
  const { axe } = await loadJestAxe();

  const options: Record<string, unknown> = {
    ...axeOptions,
    rules: {
      ...JSDOM_AXE_RULES,
      ...((axeOptions.rules as Record<string, unknown>) ?? {}),
    },
  };

  const results = await axe(container, options);
  expect(results).toHaveNoViolations();
}

// Типы для expect(...).toHaveNoViolations(), пока @types/jest-axe не
// установлен (он нужен только для статических импортов; хелпер
// объявляет минимальные типы сам).
declare module "vitest" {
  interface Assertion<T = any> {
    toHaveNoViolations(): Assertion<T>;
  }
  interface AsymmetricMatchersContaining {
    toHaveNoViolations(): void;
  }
}
