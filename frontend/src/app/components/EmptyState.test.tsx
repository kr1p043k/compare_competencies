import { render, screen } from "@testing-library/react";
import { Inbox } from "lucide-react";
import { describe, expect, it } from "vitest";

import { EmptyState } from "./EmptyState";

describe("EmptyState", () => {
  it("показывает заголовок и описание", () => {
    render(<EmptyState title="Ничего нет" description="Измените фильтры" />);
    expect(screen.getByText("Ничего нет")).toBeInTheDocument();
    expect(screen.getByText("Измените фильтры")).toBeInTheDocument();
  });

  it("отрисовывает иконку и действие, когда переданы", () => {
    render(
      <EmptyState
        icon={Inbox}
        title="Пусто"
        action={<button>Собрать</button>}
      />,
    );
    expect(screen.getByRole("button", { name: "Собрать" })).toBeInTheDocument();
  });

  it("компактный размер не ломает заголовок", () => {
    render(<EmptyState size="compact" title="Нет данных" />);
    expect(screen.getByText("Нет данных")).toBeInTheDocument();
  });
});
