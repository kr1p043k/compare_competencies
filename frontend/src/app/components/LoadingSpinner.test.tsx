import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { FullPageLoader, LoadingSpinner } from "./LoadingSpinner";

describe("LoadingSpinner", () => {
  it("объявляет себя статусом для скринридеров", () => {
    render(<LoadingSpinner />);
    const status = screen.getByRole("status", { name: "Загрузка" });
    expect(status).toBeInTheDocument();
  });

  it("FullPageLoader тоже роль status", () => {
    render(<FullPageLoader />);
    expect(screen.getByRole("status", { name: "Загрузка" })).toBeInTheDocument();
  });
});
