import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { DataViewer } from "./DataViewer";

describe("DataViewer", () => {
  it("renders object keys and primitive values", () => {
    render(<DataViewer data={{ alpha: "one", beta: 42 }} />);
    expect(screen.getByText("alpha")).toBeTruthy();
    expect(screen.getByText("one")).toBeTruthy();
    expect(screen.getByText("beta")).toBeTruthy();
    expect(screen.getByText("42")).toBeTruthy();
  });

  it("renders booleans as badges", () => {
    render(<DataViewer data={{ ok: true }} />);
    expect(screen.getByText("true")).toBeTruthy();
  });

  it("truncates long strings to 120 chars with an ellipsis", () => {
    const long = "x".repeat(200);
    const { container } = render(<DataViewer data={{ s: long }} />);
    const text = container.textContent || "";
    expect(text).toContain("x".repeat(120));
    expect(text).not.toContain("x".repeat(121));
  });

  it("renders nothing for falsy data", () => {
    const { container } = render(<DataViewer data={null} />);
    expect(container.firstChild).toBeNull();
  });
});
