/**
 * MySkillsCard (7.5.4): скелетон загрузки, ошибка чтения профиля, пустой список,
 * бейджи с удалением только «своих» навыков, ввод нового навыка.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MySkillsCard } from "./MySkillsCard";

afterEach(cleanup);

type Props = ComponentProps<typeof MySkillsCard>;

const SELF = {
  profile: "backend",
  target_level: "middle",
  skills: ["Python", "SQL"],
  user_added: ["Python"],
};

function base(overrides: Partial<Props> = {}): Props {
  return {
    self: SELF,
    selfLoading: false,
    selfError: null,
    skillsCount: 2,
    owned: new Set(["python"]),
    saving: false,
    newSkill: "",
    onNewSkillChange: vi.fn(),
    onAddSkill: vi.fn(),
    onRemoveSkill: vi.fn(),
    opError: null,
    ...overrides,
  };
}

describe("MySkillsCard", () => {
  it("загрузка -> скелетоны, без данных", () => {
    render(<MySkillsCard {...base({ self: null, selfLoading: true })} />);
    expect(screen.getByLabelText("Загрузка навыков")).toBeTruthy();
    expect(screen.queryByText("Python")).toBeNull();
  });

  it("ошибка чтения профиля показывается текстом", () => {
    render(<MySkillsCard {...base({ self: null, selfError: "Нет доступа" })} />);
    expect(screen.getByText("Нет доступа")).toBeTruthy();
  });

  it("пустой профиль -> EmptyState", () => {
    render(<MySkillsCard {...base({ self: { ...SELF, skills: [] }, skillsCount: 0 })} />);
    expect(screen.getByText("Навыков пока нет")).toBeTruthy();
  });

  it("счётчик в заголовке и удаление только своих навыков", () => {
    const onRemoveSkill = vi.fn();
    render(<MySkillsCard {...base({ onRemoveSkill })} />);
    expect(screen.getByText("Мои навыки (2)")).toBeTruthy();
    // Python «свой» - есть кнопка удаления; SQL из программы - кнопки нет.
    const del = screen.getByRole("button", { name: "Удалить Python" });
    expect(screen.queryByRole("button", { name: "Удалить SQL" })).toBeNull();
    fireEvent.click(del);
    expect(onRemoveSkill).toHaveBeenCalledWith("Python");
  });

  it("ввод и добавление навыка", () => {
    const onNewSkillChange = vi.fn();
    const onAddSkill = vi.fn();
    const { rerender } = render(<MySkillsCard {...base({ onNewSkillChange, onAddSkill })} />);
    const input = screen.getByLabelText("Новый навык") as HTMLInputElement;
    // Пустой ввод -> кнопка добавления заблокирована.
    expect((screen.getByRole("button", { name: "Добавить навык" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(input, { target: { value: "Docker" } });
    expect(onNewSkillChange).toHaveBeenCalledWith("Docker");
    rerender(<MySkillsCard {...base({ onNewSkillChange, onAddSkill, newSkill: "Docker" })} />);
    fireEvent.click(screen.getByRole("button", { name: "Добавить навык" }));
    expect(onAddSkill).toHaveBeenCalled();
    fireEvent.keyDown(screen.getByLabelText("Новый навык"), { key: "Enter" });
    expect(onAddSkill).toHaveBeenCalledTimes(2);
  });

  it("состояние сохранения и ошибка операции", () => {
    render(<MySkillsCard {...base({ saving: true, newSkill: "Docker", opError: "Не сохранилось" })} />);
    expect((screen.getByRole("button", { name: "Добавить навык" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Удалить Python" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("Не сохранилось")).toBeTruthy();
  });
});