import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { KNOWN_TABS } from "./router";

const appPath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../app/App.tsx");

/** Значения вкладок из навигации App.tsx: `{ value: "publications", label: ... }`. */
function navTabValues(): string[] {
  const source = readFileSync(appPath, "utf8");
  return [...source.matchAll(/\{\s*value:\s*"([a-z-]+)"/g)].map((match) => match[1]);
}

describe("KNOWN_TABS", () => {
  it("содержит все вкладки навигации из App.tsx", () => {
    const nav = navTabValues();
    expect(nav.length).toBeGreaterThan(5);
    const missing = nav.filter((tab) => !(KNOWN_TABS as readonly string[]).includes(tab));
    expect(missing).toEqual([]);
  });

  it("содержит вкладку публикаций (роут #/publications)", () => {
    expect(KNOWN_TABS).toContain("publications");
  });

  it("не имеет дублей", () => {
    expect(new Set(KNOWN_TABS).size).toBe(KNOWN_TABS.length);
  });
});
