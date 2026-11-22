/**
 * MyTechnologiesCard (7.5.4): скелетон, пустой список, бейджи с удалением,
 * ввод технологии, подсказки из пула и из suggestPool (datalist), ошибка.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MyTechnologiesCard } from "./MyTechnologiesCard";

afterEach(cleanup);

type Props = ComponentProps<typeof MyTechnologiesCard>;

const SELF = {
  profile: "backend",
  target_level: "middle",
  skills: [],
  user_added: [],
  technologies: ["Docker"],
};

function base(overrides: Partial<Props> = {}): Props {
  return {
    self: SELF,
    selfLoading: false,
    techList: ["Docker"],
    techSaving: false,
    techInput: "",
    onTechInputChange: vi.fn(),
    onAddTech: vi.fn(),
    onRemoveTech: vi.fn(),
    techSuggestions: [],
    suggestPool: [],
    techError: null,
    ...overrides,
  };
}

describe("MyTechnologiesCard", () => {
  it("загрузка -> скелетоны, без данных", () => {
    render(<MyTechnologiesCard {...base({ self: null, selfLoading: true })} />);
    expect(screen.getByLabelText("Загрузка технологий")).toBeTruthy();
    expect(screen.queryByText("Docker")).toBeNull();
  });

  it("пустой список -> EmptyState", () => {
    render(<MyTechnologiesCard {...base({ techList: [] })} />);
    expect(screen.getByText("Пока пусто")).toBeTruthy();
  });

  it("счётчик в заголовке и удаление технологии", () => {
    const onRemoveTech = vi.fn();
    render(<MyTechnologiesCard {...base({ onRemoveTech })} />);
    expect(screen.getByText("Мои технологии (1)")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Убрать Docker" }));
    expect(onRemoveTech).toHaveBeenCalledWith("Docker");
  });

  it("ввод и добавление технологии (кнопка и Enter)", () => {
    const onTechInputChange = vi.fn();
    const onAddTech = vi.fn();
    const { rerender } = render(
      <MyTechnologiesCard {...base({ onTechInputChange, onAddTech })} />,
    );
    expect((screen.getByRole("button", { name: "Добавить технологию" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Новая технология"), { target: { value: "Kafka" } });
    expect(onTechInputChange).toHaveBeenCalledWith("Kafka");
    rerender(<MyTechnologiesCard {...base({ onTechInputChange, onAddTech, techInput: "Kafka" })} />);
    fireEvent.click(screen.getByRole("button", { name: "Добавить технологию" }));
    expect(onAddTech).toHaveBeenCalled();
    fireEvent.keyDown(screen.getByLabelText("Новая технология"), { key: "Enter" });
    expect(onAddTech).toHaveBeenCalledTimes(2);
  });

  it("подсказки: клик добавляет технологию, datalist заполнен из пула", () => {
    const onAddTech = vi.fn();
    render(
      <MyTechnologiesCard
        {...base({ onAddTech, techSuggestions: ["Kubernetes"], suggestPool: ["Kafka", "Redis"] })}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Добавить Kubernetes" }));
    expect(onAddTech).toHaveBeenCalledWith("Kubernetes");
    const options = Array.from(document.querySelectorAll("#my-tech-suggest option")).map((o) =>
      o.getAttribute("value"),
    );
    expect(options).toEqual(["Kafka", "Redis"]);
  });

  it("ошибка и состояние сохранения", () => {
    render(
      <MyTechnologiesCard
        {...base({ techSaving: true, techInput: "Kafka", techError: "Не сохранилось", techSuggestions: ["Redis"] })}
      />,
    );
    expect((screen.getByRole("button", { name: "Добавить технологию" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Убрать Docker" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Добавить Redis" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("Не сохранилось")).toBeTruthy();
  });
});