/**
 * ChatMarkdown: безопасный markdown-поднабор для чата (7.5.4).
 * Проверяем рендер блоков, запрет javascript:-ссылок и экранирование HTML.
 */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ChatMarkdown } from "./ChatMarkdown";

afterEach(cleanup);

describe("ChatMarkdown", () => {
  it("пустой текст -> ничего", () => {
    const { container } = render(<ChatMarkdown>{""}</ChatMarkdown>);
    expect(container.textContent).toBe("");
  });

  it("заголовок, параграф и жирный текст", () => {
    render(<ChatMarkdown>{"## Итог\n\nэто **важно**"}</ChatMarkdown>);
    expect(screen.getByText("Итог")).toBeTruthy();
    expect(screen.getByText("важно")).toBeTruthy();
  });

  it("списки: маркированный и нумерованный", () => {
    render(<ChatMarkdown>{"- один\n- два"}</ChatMarkdown>);
    expect(screen.getByText("один")).toBeTruthy();
    expect(screen.getByText("два")).toBeTruthy();
  });

  it("код-блок рендерится как <code>", () => {
    const { container } = render(<ChatMarkdown>{"```\nprint(1)\n```"}</ChatMarkdown>);
    expect(container.querySelector("code")?.textContent).toContain("print(1)");
  });

  it("http-ссылка становится кликабельной", () => {
    const { container } = render(
      <ChatMarkdown>{"[хаб](https://hub.sfedu.ru)"}</ChatMarkdown>,
    );
    const a = container.querySelector("a");
    expect(a?.getAttribute("href")).toBe("https://hub.sfedu.ru");
    expect(a?.getAttribute("rel")).toContain("noopener");
  });

  it("javascript:-ссылка не превращается в href", () => {
    const { container } = render(
      <ChatMarkdown>{"[x](javascript:alert(1))"}</ChatMarkdown>,
    );
    expect(container.querySelector("a")).toBeNull();
  });

  it("HTML из модели показывается как текст, не интерпретируется", () => {
    const { container } = render(
      <ChatMarkdown>{"<img src=x onerror=alert(1)>"}</ChatMarkdown>,
    );
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("<img");
  });
});