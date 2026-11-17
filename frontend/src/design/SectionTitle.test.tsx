/**
 * SectionTitle (primitives.tsx): заголовок h2, опциональный hint, опциональные
 * actions.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SectionTitle } from "./primitives";

describe("SectionTitle", () => {
  it("рендерит заголовок как h2", () => {
    render(<SectionTitle title="Навыки рынка" />);
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent("Навыки рынка");
  });

  it("показывает hint под заголовком, когда он передан", () => {
    render(<SectionTitle title="Навыки рынка" hint="обновлено из hh.ru" />);
    expect(screen.getByText("обновлено из hh.ru")).toBeInTheDocument();
  });

  it("рендерит actions справа", () => {
    render(
      <SectionTitle
        title="Навыки рынка"
        actions={<button type="button">Обновить</button>}
      />,
    );
    expect(screen.getByRole("button", { name: "Обновить" })).toBeInTheDocument();
  });

  it("без hint и actions дополнительных блоков нет", () => {
    const { container } = render(<SectionTitle title="Навыки рынка" />);
    expect(container.querySelector("p")).toBeNull();
    expect(container.querySelector("button")).toBeNull();
  });
});