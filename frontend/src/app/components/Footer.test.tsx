import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Footer } from "./Footer";

describe("Footer", () => {
  it("рисует подвал с подписью проекта", () => {
    render(<Footer />);
    expect(screen.getByRole("contentinfo")).toBeInTheDocument();
    expect(screen.getByText(/Competency Gap Analyzer/i)).toBeInTheDocument();
  });
});
