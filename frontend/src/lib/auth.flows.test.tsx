import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const fetchMock = vi.fn();
let auth: typeof import("./auth");

beforeAll(async () => {
  vi.stubGlobal("fetch", fetchMock);
  auth = await import("./auth");
});

afterAll(() => {
  vi.unstubAllGlobals();
});

beforeEach(() => {
  fetchMock.mockReset();
  localStorage.clear();
  window.history.replaceState(null, "", "/");
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  cleanup();
});

function Probe() {
  const { role, name, username, authed, login, logout } = auth.useAuth();
  return (
    <div>
      <span data-testid="role">{role ?? "-"}</span>
      <span data-testid="name">{name ?? "-"}</span>
      <span data-testid="username">{username ?? "-"}</span>
      <span data-testid="authed">{String(authed)}</span>
      <button type="button" onClick={() => login(null, "user", "Bob")}>
        login
      </button>
      <button type="button" onClick={logout}>
        logout
      </button>
    </div>
  );
}

function renderProvider() {
  return render(
    <auth.AuthProvider>
      <Probe />
    </auth.AuthProvider>,
  );
}

describe("AuthProvider restore", () => {
  it("restores the session from /api/auth/me and marks the hint", async () => {
    fetchMock.mockResolvedValue(json({ role: "admin", name: "Ann", username: "ann" }));

    renderProvider();

    await waitFor(() => expect(screen.getByTestId("authed").textContent).toBe("true"));
    expect(screen.getByTestId("role").textContent).toBe("admin");
    expect(screen.getByTestId("name").textContent).toBe("Ann");
    expect(screen.getByTestId("username").textContent).toBe("ann");
    expect(localStorage.getItem("sessionHint")).toBe("1");
  });

  it("clears the hint when /api/auth/me is not ok", async () => {
    localStorage.setItem("sessionHint", "1");
    fetchMock.mockResolvedValue(json({}, 401));

    renderProvider();

    await waitFor(() => expect(screen.getByTestId("authed").textContent).toBe("false"));
    expect(localStorage.getItem("sessionHint")).toBeNull();
  });

  it("reports a failed session restore", async () => {
    fetchMock.mockRejectedValue(new TypeError("offline"));

    renderProvider();

    await waitFor(() =>
      expect(console.error).toHaveBeenCalledWith(
        "[lib/auth.tsx:restore]",
        expect.any(TypeError),
      ),
    );
    expect(screen.getByTestId("authed").textContent).toBe("false");
  });

  it("ignores the restore result after unmount", async () => {
    let resolveMe!: (value: Response) => void;
    fetchMock.mockImplementation(
      () =>
        new Promise<Response>((resolve) => {
          resolveMe = resolve;
        }),
    );

    const { unmount } = renderProvider();
    unmount();
    resolveMe(json({ role: "admin", name: "Ann", username: "ann" }));
    await act(async () => {
      await Promise.resolve();
    });

    expect(console.error).not.toHaveBeenCalled();
    expect(localStorage.getItem("sessionHint")).toBeNull();
  });

  it("treats an unreadable session hint as logged out", async () => {
    const get = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    fetchMock.mockResolvedValue(json({}, 401));

    renderProvider();

    await waitFor(() => expect(screen.getByTestId("authed").textContent).toBe("false"));
    get.mockRestore();
  });
});

describe("login", () => {
  it("logs in optimistically and refreshes the profile", async () => {
    fetchMock
      .mockResolvedValueOnce(json({}, 401))
      .mockResolvedValueOnce(json({ role: "admin", name: "Ann", username: "ann" }));

    renderProvider();
    await waitFor(() => expect(screen.getByTestId("authed").textContent).toBe("false"));

    fireEvent.click(screen.getByText("login"));

    expect(screen.getByTestId("role").textContent).toBe("user");
    expect(screen.getByTestId("name").textContent).toBe("Bob");
    await waitFor(() => expect(screen.getByTestId("username").textContent).toBe("ann"));
    expect(screen.getByTestId("role").textContent).toBe("admin");
    expect(screen.getByTestId("name").textContent).toBe("Ann");
    expect(localStorage.getItem("sessionHint")).toBe("1");
  });

  it("keeps the optimistic profile when the refresh is not ok", async () => {
    fetchMock.mockResolvedValueOnce(json({}, 401)).mockResolvedValueOnce(json({}, 401));

    renderProvider();
    await waitFor(() => expect(screen.getByTestId("authed").textContent).toBe("false"));

    fireEvent.click(screen.getByText("login"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    expect(screen.getByTestId("role").textContent).toBe("user");
    expect(screen.getByTestId("name").textContent).toBe("Bob");
    expect(screen.getByTestId("username").textContent).toBe("-");
  });

  it("reports a failed profile refresh without breaking the session", async () => {
    fetchMock
      .mockResolvedValueOnce(json({}, 401))
      .mockRejectedValueOnce(new TypeError("offline"));

    renderProvider();
    await waitFor(() => expect(screen.getByTestId("authed").textContent).toBe("false"));

    fireEvent.click(screen.getByText("login"));

    await waitFor(() =>
      expect(console.error).toHaveBeenCalledWith(
        "[lib/auth.tsx:login-refresh]",
        expect.any(TypeError),
      ),
    );
    expect(screen.getByTestId("authed").textContent).toBe("true");
    expect(screen.getByTestId("role").textContent).toBe("user");
  });
});

describe("session events", () => {
  it("applies an external hub session to a mounted provider", async () => {
    fetchMock.mockResolvedValue(json({}, 401));
    renderProvider();
    await waitFor(() => expect(screen.getByTestId("authed").textContent).toBe("false"));

    act(() => auth.setSession({ role: "dc", name: "Dana", username: "dana" }));

    expect(screen.getByTestId("role").textContent).toBe("dc");
    expect(screen.getByTestId("name").textContent).toBe("Dana");
    expect(screen.getByTestId("username").textContent).toBe("dana");
    expect(screen.getByTestId("authed").textContent).toBe("true");
    expect(localStorage.getItem("sessionHint")).toBe("1");
  });

  it("does not fail when an external session arrives with no provider", () => {
    expect(() => auth.setSession({ role: "base", name: "Solo" })).not.toThrow();
    expect(localStorage.getItem("sessionHint")).toBe("1");
  });

  it("resets state and leaves the current page on unauthorized", async () => {
    fetchMock.mockResolvedValue(json({ role: "admin", name: "Ann", username: "ann" }));
    renderProvider();
    await waitFor(() => expect(screen.getByTestId("authed").textContent).toBe("true"));
    localStorage.setItem("auth", "1");
    window.history.replaceState(null, "", "/vacancies");

    act(() => auth.handleUnauthorized());

    expect(localStorage.getItem("auth")).toBeNull();
    expect(localStorage.getItem("sessionHint")).toBeNull();
    expect(screen.getByTestId("authed").textContent).toBe("false");
  });

  it("does not navigate when already on the login page", async () => {
    fetchMock.mockResolvedValue(json({}, 401));
    renderProvider();
    await waitFor(() => expect(screen.getByTestId("authed").textContent).toBe("false"));

    expect(() => act(() => auth.handleUnauthorized())).not.toThrow();
    expect(screen.getByTestId("authed").textContent).toBe("false");
  });

  it("survives storage errors while clearing the session", async () => {
    const remove = vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    fetchMock.mockResolvedValue(json({}, 401));
    renderProvider();
    await waitFor(() => expect(screen.getByTestId("authed").textContent).toBe("false"));

    act(() => auth.handleUnauthorized());

    expect(console.error).toHaveBeenCalledWith(
      "[lib/auth.tsx:handleUnauthorized]",
      expect.any(Error),
    );
    remove.mockRestore();
  });

  it("reports a hint write failure but still updates the session", async () => {
    const set = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    fetchMock.mockResolvedValue(json({}, 401));
    renderProvider();
    await waitFor(() => expect(screen.getByTestId("authed").textContent).toBe("false"));

    act(() => auth.setSession({ role: "dc", name: "Dana" }));

    expect(console.error).toHaveBeenCalledWith("[lib/auth.tsx:hint]", expect.any(Error));
    expect(screen.getByTestId("authed").textContent).toBe("true");
    set.mockRestore();
  });
});

describe("logAction", () => {
  it("posts an action log with a JSON body", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));

    auth.logAction("open", "vacancies");
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/log",
        expect.objectContaining({ method: "POST" }),
      ),
    );

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({ action: "open", detail: "vacancies" });
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
  });

  it("reports a failing action log", async () => {
    fetchMock.mockRejectedValue(new TypeError("offline"));

    auth.logAction("open");

    await waitFor(() =>
      expect(console.error).toHaveBeenCalledWith("[lib/auth.tsx]", expect.any(TypeError)),
    );
  });
});

describe("default context", () => {
  it("provides safe no-op login and logout outside a provider", () => {
    render(<Probe />);

    expect(() => {
      fireEvent.click(screen.getByText("login"));
      fireEvent.click(screen.getByText("logout"));
    }).not.toThrow();
    expect(screen.getByTestId("authed").textContent).toBe("false");
  });
});
