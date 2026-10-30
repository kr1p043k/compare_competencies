import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Briefcase } from "lucide-react";

import { PageHeader } from "./PageHeader";

describe("PageHeader", () => {
  it("renders the title as a level-1 heading", () => {
    render(<PageHeader title="Вакансии" />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Вакансии");
  });

  it("renders description and icon", () => {
    const { container } = render(
      <PageHeader title="Вакансии" description="Список вакансий" icon={Briefcase} />,
    );
    expect(screen.getByText("Список вакансий")).toBeInTheDocument();
    expect(container.querySelector("svg")).not.toBeNull();
  });

  it("renders actions on the right", () => {
    render(<PageHeader title="Вакансии" actions={<button>Обновить</button>} />);
    expect(screen.getByRole("button", { name: "Обновить" })).toBeInTheDocument();
  });

  it("omits description and icon when not provided", () => {
    const { container } = render(<PageHeader title="Только заголовок" />);
    expect(container.querySelector("svg")).toBeNull();
    expect(container.querySelector("p")).toBeNull();
  });
});