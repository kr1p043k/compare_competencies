import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ShowMore } from "./show-more";

describe("ShowMore", () => {
  it("ничего не рисует, когда всё уже показано", () => {
    const { container } = render(
      <ShowMore total={5} shown={5} expanded={false} onToggle={() => {}} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("показывает, сколько ещё скрыто, и кнопку", () => {
    render(<ShowMore total={10} shown={4} expanded={false} onToggle={() => {}} />);
    const button = screen.getByRole("button");
    expect(button).toHaveTextContent("6 из 10");
    expect(button).toHaveAttribute("aria-expanded", "false");
  });

  it("в раскрытом состоянии - подпись «Свернуть» и aria-expanded=true", () => {
    render(<ShowMore total={10} shown={4} expanded onToggle={() => {}} />);
    const button = screen.getByRole("button");
    expect(button).toHaveTextContent("Свернуть");
    expect(button).toHaveAttribute("aria-expanded", "true");
  });

  it("клик вызывает onToggle", () => {
    const onToggle = vi.fn();
    render(<ShowMore total={3} shown={1} expanded={false} onToggle={onToggle} />);
    fireEvent.click(screen.getByRole("button"));
    expect(onToggle).toHaveBeenCalledTimes(1);
  });
});
