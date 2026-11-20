import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { request } from "../../lib/api";
import { SelfProfileEditor } from "./SelfProfileEditor";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));
vi.mock("../../lib/errors", () => ({
  errMessage: (e: unknown) => (e instanceof Error ? e.message : String(e)),
}));

const requestMock = vi.mocked(request);

const SELF = {
  profile: "base",
  target_level: "middle",
  skills: ["python", "sql"],
  user_added: ["sql"],
  competencies: ["UK-1"],
};

describe("SelfProfileEditor", () => {
  beforeEach(() => {
    requestMock.mockReset();
  });

  it("shows the owner name and email as the description", async () => {
    requestMock.mockResolvedValue(SELF);
    render(<SelfProfileEditor displayName="Ivan" email="ivan@example.com" />);
    await waitFor(() => expect(screen.getByText(/Ivan/)).toBeTruthy());
    expect(screen.getByText(/ivan@example.com/)).toBeTruthy();
  });

  it("renders loaded skills and competencies", async () => {
    requestMock.mockResolvedValue(SELF);
    render(<SelfProfileEditor />);
    await waitFor(() => expect(screen.getByText("python")).toBeTruthy());
    expect(screen.getByText("sql")).toBeTruthy();
    expect(screen.getByText("UK-1")).toBeTruthy();
  });

  it("adds a skill on Enter with a PATCH body", async () => {
    requestMock.mockResolvedValue(SELF);
    render(<SelfProfileEditor />);
    const input = await screen.findByRole("textbox");
    fireEvent.change(input, { target: { value: "rust" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith("/api/profiles/self", {
        method: "PATCH",
        body: { add_skills: ["rust"] },
      }),
    );
  });

  it("shows the error message when loading fails", async () => {
    requestMock.mockRejectedValue(new Error("backend down"));
    render(<SelfProfileEditor />);
    await waitFor(() => expect(screen.getByText("backend down")).toBeTruthy());
  });
});
