/**
 * a11y: StudentsTab без нарушений axe после загрузки списка и карточки
 * студента.
 *
 * Сеть мокается на уровне lib/api.request (обёртка api() ходит через него).
 * Ждём реальные данные через findBy, прежде чем пускать axe: у скелетонов
 * нет смысловой разметки для проверки.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, it, vi } from "vitest";

import { request } from "../../lib/api";
import { expectNoA11yViolations } from "../../test/a11y";
import { StudentsTab } from "./StudentsTab";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));

const requestMock = vi.mocked(request);

const STUDENTS = {
  students: [
    {
      email: "ivan@sfedu.ru",
      full_name: "Иван Петров",
      target_level: "bachelor",
      skills_count: 3,
      competencies_count: 2,
      has_profile: true,
    },
    {
      email: "olga@sfedu.ru",
      full_name: "Ольга Смирнова",
      target_level: null,
      skills_count: 0,
      competencies_count: 0,
      has_profile: false,
    },
  ],
};

const DETAIL = {
  email: "ivan@sfedu.ru",
  full_name: "Иван Петров",
  target_level: "bachelor",
  skills: ["Python", "SQL", "Git"],
  user_added: ["Rust"],
  competencies: ["backend", "data"],
  updated_at: "2026-11-01T10:00:00Z",
  has_profile: true,
};

beforeEach(() => {
  requestMock.mockReset();
  requestMock.mockImplementation(((url: string) => {
    const u = String(url);
    if (u.startsWith("/api/admin/students/skills")) return Promise.resolve(DETAIL);
    if (u.startsWith("/api/teacher/students")) return Promise.resolve(STUDENTS);
    return Promise.reject(new Error(`unexpected endpoint: ${u}`));
  }) as never);
});

describe("StudentsTab (a11y)", () => {
  it("список студентов - без нарушений", async () => {
    const { container } = render(<StudentsTab />);
    await screen.findByText("Иван Петров");

    await expectNoA11yViolations(container);
  });

  it("карточка выбранного студента - без нарушений", async () => {
    const user = userEvent.setup();
    const { container } = render(<StudentsTab />);
    await screen.findByText("Иван Петров");

    await user.click(screen.getByLabelText("Показать профиль: Иван Петров"));
    await screen.findByText("Навыки (3)");

    await expectNoA11yViolations(container);
  });
});
