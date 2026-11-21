import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import { CompetencyTree } from "./CompetencyTree";

describe("CompetencyTree", () => {
  it("renders nothing for an empty list", () => {
    const { container } = render(<CompetencyTree competencies={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders a leaf competency with coverage percent", () => {
    render(
      <CompetencyTree
        competencies={[
          { code: "UK-1", total_skills: 4, matched_skills: 3, coverage: 0.75 },
        ]}
      />,
    );
    expect(screen.getByText("UK-1")).toBeTruthy();
    expect(screen.getByText("3/4")).toBeTruthy();
    expect(screen.getByText("75%")).toBeTruthy();
  });

  it("shows a dash when coverage is undefined", () => {
    render(<CompetencyTree competencies={[{ code: "UK-2", total_skills: 2 }]} />);
    expect(screen.getByText("-%")).toBeTruthy();
  });

  it("expands children on click", () => {
    render(
      <CompetencyTree
        competencies={[
          { code: "PK-1" },
          { code: "PK-1.1", total_skills: 1, matched_skills: 1, coverage: 1 },
        ]}
      />,
    );
    // первый уровень раскрыт по умолчанию (depth < 2)
    expect(screen.getByText("PK-1.1")).toBeTruthy();
    fireEvent.click(screen.getByText("PK-1"));
    expect(screen.queryByText("PK-1.1")).toBeNull();
  });

  it("expands matched and gap skills on click", () => {
    render(
      <CompetencyTree
        competencies={[
          {
            code: "DL-3",
            total_skills: 2,
            matched_skills: 1,
            coverage: 0.5,
            matched: ["python"],
            gaps: ["rust"],
          },
        ]}
      />,
    );
    // навыки видны в раскрытом узле, после клика узел сворачивается
    expect(screen.getByText("python")).toBeTruthy();
    expect(screen.getByText("rust")).toBeTruthy();
    fireEvent.click(screen.getByText("DL-3"));
    expect(screen.queryByText("python")).toBeNull();
  });
});