import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import { PublicationDetailDialog } from "./PublicationDetailDialog";
import type { Publication } from "./publicationsFilter";

vi.mock("../api", () => ({ api: vi.fn() }));

afterEach(cleanup);

const apiMock = vi.mocked(api);

function makeDetail(extra: Partial<Publication> = {}): Publication {
  return {
    id: 7,
    title: "Deep Learning for Engineers",
    authors: ["Ivanov Ivan", "Petrov Petr"],
    type_id: 1,
    type_name: "Article",
    signature: null,
    doi: "10.1234/abcd",
    isbn: null,
    track_id: null,
    created_at: null,
    year: 2025,
    academic_subject: ["CS"],
    education_program: [],
    abstract: "Short abstract text",
    keywords: ["ml"],
    journal: "Journal of ML",
    publisher: null,
    language: "en",
    publishing_year: null,
    pages: "10-20",
    files: [],
    url: null,
    web_url: "https://hub.sfedu.ru/pub/7",
    creator_username: null,
    ...extra,
  };
}

describe("PublicationDetailDialog", () => {
  beforeEach(() => {
    apiMock.mockReset();
  });

  it("shows a skeleton and requests the publication by id", () => {
    apiMock.mockReturnValue(new Promise(() => {}));
    render(<PublicationDetailDialog id={7} onClose={() => {}} />);
    expect(document.querySelector('[role="status"]')).not.toBeNull();
    expect(apiMock).toHaveBeenCalledWith(
      "/publications/7",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it("renders title, authors, doi link, files and hub link", async () => {
    apiMock.mockResolvedValue(
      makeDetail({
        files: [
          { url: "https://cdn.example/f.pdf", filename: "paper.pdf", size: 2048, content_type: "application/pdf" },
        ],
      }),
    );
    render(<PublicationDetailDialog id={7} onClose={() => {}} />);

    expect(await screen.findByText("Deep Learning for Engineers")).toBeTruthy();
    expect(screen.getByText("Ivanov Ivan, Petrov Petr")).toBeTruthy();
    expect(screen.getByText("Short abstract text")).toBeTruthy();

    const doi = screen.getByText("10.1234/abcd");
    expect(doi.getAttribute("href")).toBe("https://doi.org/10.1234/abcd");

    expect(screen.getByText("paper.pdf")).toBeTruthy();
    expect(screen.getByText(/2\.0/)).toBeTruthy();

    const hub = document.querySelector('a[href="https://hub.sfedu.ru/pub/7"]');
    expect(hub).not.toBeNull();
  });

  it("shows an error and retries on demand", async () => {
    apiMock.mockRejectedValueOnce(new Error("boom")).mockResolvedValueOnce(makeDetail());
    render(<PublicationDetailDialog id={7} onClose={() => {}} />);

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText("boom")).toBeTruthy();

    await userEvent.click(within(alert).getByRole("button"));

    expect(await screen.findByText("Deep Learning for Engineers")).toBeTruthy();
    expect(apiMock).toHaveBeenCalledTimes(2);
  });

  it("ignores abort errors", async () => {
    apiMock.mockRejectedValue(new DOMException("aborted", "AbortError"));
    render(<PublicationDetailDialog id={7} onClose={() => {}} />);

    await waitFor(() => expect(apiMock).toHaveBeenCalled());
    expect(screen.queryByRole("alert")).toBeNull();
  });
});