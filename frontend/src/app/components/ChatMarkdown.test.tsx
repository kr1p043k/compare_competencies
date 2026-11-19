import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { ChatMarkdown } from "./ChatMarkdown";

describe("ChatMarkdown", () => {
  it("renders headings and paragraphs", () => {
    render(<ChatMarkdown>{"# Заголовок\n\nОбычный абзац"}</ChatMarkdown>);
    expect(screen.getByText("Заголовок")).toBeTruthy();
    expect(screen.getByText("Обычный абзац")).toBeTruthy();
  });

  it("renders bullet lists", () => {
    render(<ChatMarkdown>{"- первый\n- второй"}</ChatMarkdown>);
    expect(screen.getByText("первый")).toBeTruthy();
    expect(screen.getByText("второй")).toBeTruthy();
    expect(document.querySelectorAll("li").length).toBe(2);
  });

  it("renders inline code and fenced code blocks", () => {
    render(<ChatMarkdown>{"до `x = 1` после\n\n```\nprint(1)\n```"}</ChatMarkdown>);
    expect(screen.getByText("x = 1")).toBeTruthy();
    expect(screen.getByText(/print\(1\)/)).toBeTruthy();
  });

  it("makes https links external and safe", () => {
    render(<ChatMarkdown>{"[ссылка](https://example.com/x)"}</ChatMarkdown>);
    const a = screen.getByRole("link") as HTMLAnchorElement;
    expect(a.getAttribute("href")).toBe("https://example.com/x");
    expect(a.getAttribute("target")).toBe("_blank");
    expect(a.getAttribute("rel")).toContain("noopener");
  });

  it("does not emit a javascript: href", () => {
    const { container } = render(<ChatMarkdown>{"[bad](javascript:alert(1))"}</ChatMarkdown>);
    const hrefs = Array.from(container.querySelectorAll("a")).map((a) => a.getAttribute("href"));
    expect(hrefs.every((h) => !h || !h.toLowerCase().startsWith("javascript:"))).toBe(true);
  });

  it("renders nothing for empty or whitespace input", () => {
    const { container } = render(<ChatMarkdown>{"   "}</ChatMarkdown>);
    expect(container.firstChild).toBeNull();
  });

  it("keeps plain text intact when there is no markdown", () => {
    render(<ChatMarkdown>{"просто текст без разметки"}</ChatMarkdown>);
    expect(screen.getByText("просто текст без разметки")).toBeTruthy();
  });
});
