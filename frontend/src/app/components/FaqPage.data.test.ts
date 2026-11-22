import { describe, expect, it } from "vitest";

import { faqData } from "./FaqPage.data";

describe("faqData", () => {
  it("is not empty and every entry has question, answer and category", () => {
    expect(faqData.length).toBeGreaterThan(20);
    for (const item of faqData) {
      expect(item.q.trim().length).toBeGreaterThan(5);
      expect(item.a.trim().length).toBeGreaterThan(10);
      expect(item.category.trim().length).toBeGreaterThan(0);
    }
  });

  it("has no duplicate questions", () => {
    const seen = new Set<string>();
    for (const item of faqData) {
      const key = item.q.trim().toLowerCase();
      expect(seen.has(key)).toBe(false);
      seen.add(key);
    }
  });

  it("covers a handful of distinct categories", () => {
    const categories = new Set(faqData.map((i) => i.category));
    expect(categories.size).toBeGreaterThanOrEqual(3);
  });
});