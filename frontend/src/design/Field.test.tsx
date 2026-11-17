/**
 * Field (primitives.tsx): label с htmlFor, условные hint/error (error имеет
 * приоритет), произвольные controls в children.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Field, inputClass } from "./primitives";

describe("Field", () => {
  it("ассоциирует label с контролом через htmlFor", () => {
    render(
      <Field label="Email" htmlFor="email-input">
        <input id="email-input" className={inputClass} type="email" />
      </Field>,
    );
    const input = screen.getByLabelText("Email");
    expect(input).toHaveAttribute("id", "email-input");
    expect(input).toBeInTheDocument();
  });

  it("рисует hint, когда ошибки нет", () => {
    render(
      <Field label="Название" hint="от 1 до 50 символов">
        <input />
      </Field>,
    );
    expect(screen.getByText("от 1 до 50 символов")).toBeInTheDocument();
  });

  it("error вытесняет hint: рендерится только ошибка", () => {
    render(
      <Field label="Название" hint="от 1 до 50 символов" error="Обязательное поле">
        <input />
      </Field>,
    );
    expect(screen.getByText("Обязательное поле")).toBeInTheDocument();
    expect(screen.queryByText("от 1 до 50 символов")).not.toBeInTheDocument();
  });

  it("без hint и error подтекста не рендерится", () => {
    const { container } = render(
      <Field label="Название" htmlFor="inp">
        <input id="inp" />
      </Field>,
    );
    expect(container.querySelector("p")).toBeNull();
    expect(screen.getByLabelText("Название")).toBeInTheDocument();
  });

  it("рендерит произвольный control в children", () => {
    render(
      <Field label="Отдел">
        <select aria-label="Отдел">
          <option value="math">Математика</option>
        </select>
      </Field>,
    );
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });
});