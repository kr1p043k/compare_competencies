import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

import { request } from "../../lib/api";
import { StudentStartHere } from "./StudentStartHere";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));
vi.mock("../../lib/errors", () => ({ reportError: vi.fn() }));

const requestMock = vi.mocked(request);

describe("StudentStartHere", () => {
  beforeEach(() => {
    requestMock.mockReset();
  });

  it("renders the call to action when the self profile is empty", async () => {
    requestMock.mockResolvedValue({ skills: [], competencies: [], technologies: [], user_added: [] });
    const onNavigate = vi.fn();
    render(<StudentStartHere onNavigate={onNavigate} />);

    const button = await screen.findByRole("button");
    button.click();
    expect(onNavigate).toHaveBeenCalledWith("profile");
  });

  it("renders nothing when the self profile already has data", async () => {
    requestMock.mockResolvedValue({ skills: ["python"] });
    const { container } = render(<StudentStartHere />);
    await waitFor(() => expect(requestMock).toHaveBeenCalled());
    // дождаться завершения эффекта и отсутствия блока
    await new Promise((r) => setTimeout(r, 0));
    expect(container.firstChild).toBeNull();
  });
});
