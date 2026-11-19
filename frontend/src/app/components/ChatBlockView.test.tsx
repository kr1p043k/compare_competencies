import { afterEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import {
  ChatBlockView,
  registerBlockRenderer,
  registeredBlockTypes,
  unregisterBlockRenderer,
} from "./ChatBlockView";

describe("ChatBlockView: реестр типов блоков (3.5)", () => {
  afterEach(() => {
    unregisterBlockRenderer("ping");
    unregisterBlockRenderer("bad");
  });

  it("встроенные типы зарегистрированы через реестр", () => {
    expect(registeredBlockTypes()).toEqual(
      expect.arrayContaining(["text", "metric", "citation", "chart", "action_list"]),
    );
  });

  it("рендерит встроенный text-блок", () => {
    render(<ChatBlockView block={{ type: "text", content: "привет" }} />);
    expect(screen.getByText("привет")).toBeTruthy();
  });

  it("неизвестный тип не падает, а уходит в фолбэк", () => {
    render(<ChatBlockView block={{ type: "video", url: "x" }} />);
    expect(screen.getByText(/без рендера/)).toBeTruthy();
  });

  it("плагин добавляет новый тип, отписка возвращает фолбэк", () => {
    const off = registerBlockRenderer("ping", (b) => (
      <span>{`PONG:${String(b.value)}`}</span>
    ));
    const view = render(<ChatBlockView block={{ type: "ping", value: 42 }} />);
    expect(screen.getByText("PONG:42")).toBeTruthy();
    view.unmount();
    off();
    render(<ChatBlockView block={{ type: "ping", value: 42 }} />);
    expect(screen.getByText(/без рендера/)).toBeTruthy();
  });

  it("пустой тип регистрировать нельзя", () => {
    expect(() => registerBlockRenderer("", () => null)).toThrow();
  });
});
