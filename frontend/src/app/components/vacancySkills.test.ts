import { describe, expect, it } from "vitest";

import { parseSkillsFromHtml, sanitizeHtml } from "./VacancyCard";

describe("sanitizeHtml", () => {
  it("strips tags and keeps line breaks", () => {
    expect(sanitizeHtml("<p>Первая</p><br/><p>Вторая</p>")).toBe("Первая\nВторая");
  });

  it("decodes basic entities", () => {
    expect(sanitizeHtml("A &amp; B &lt;C&gt; &quot;D&quot; &#39;E&#39;")).toBe(
      'A & B <C> "D" \'E\'',
    );
  });
});

describe("parseSkillsFromHtml", () => {
  it("finds technology keywords", () => {
    const html = "<p>Нужны Python, Docker и PostgreSQL</p>";
    const skills = parseSkillsFromHtml(html);
    expect(skills).toEqual(expect.arrayContaining(["Python", "Docker", "PostgreSQL"]));
  });

  it("matches by word boundary, not substring", () => {
    // «django» не должен давать навык «go»
    const skills = parseSkillsFromHtml("<p>Опыт работы с Django</p>");
    expect(skills).not.toContain("Go");
  });

  it("rejects russian prose", () => {
    const skills = parseSkillsFromHtml("<p>умение работать в команде и отвечать клиентам</p>");
    expect(skills).toEqual([]);
  });

  it("matches cyrillic skills with unicode boundaries", () => {
    // \b в JS не считает кириллицу буквой, поэтому парсер использует
    // (?<!\p{L}\p{N})..(?!\p{L}\p{N}). Проверяем именно этот путь.
    expect(parseSkillsFromHtml("<p>Опыт: 1С и Excel</p>")).toEqual(
      expect.arrayContaining(["1С", "Excel"]),
    );
  });

  it("rejects names with punctuation such as colons", () => {
    // «1С:Предприятие» целиком отбрасывается из-за ':', но базовый «1С» ловится.
    expect(parseSkillsFromHtml("<p>1С:Предприятие</p>")).toEqual(["1С"]);
  });

  it("ignores empty input", () => {
    expect(parseSkillsFromHtml("")).toEqual([]);
  });
});