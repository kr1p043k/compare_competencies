import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { ProfileView } from "./ProfileView";

describe("ProfileView", () => {
  it("renders profile name, level and counts", () => {
    render(
      <ProfileView
        data={{
          profile_name: "base",
          target_level: "senior",
          skills: ["python", "sql"],
          competencies: ["UK-1"],
        }}
      />,
    );
    expect(screen.getByText("base")).toBeTruthy();
    expect(screen.getByText("Senior")).toBeTruthy();
    // skills_count не задан -> берётся длина массива
    expect(screen.getByText("2")).toBeTruthy();
    expect(screen.getByText("1")).toBeTruthy();
  });

  it("falls back to dash for empty profile", () => {
    render(<ProfileView data={{}} />);
    expect(screen.getAllByText("-").length).toBeGreaterThan(0);
  });

  it("uses explicit counts when provided", () => {
    render(<ProfileView data={{ skills_count: 7, competencies_count: 3 }} />);
    expect(screen.getByText("7")).toBeTruthy();
    expect(screen.getByText("3")).toBeTruthy();
  });
});
