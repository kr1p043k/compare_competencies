/**
 * logout должен очищать хэш вкладки: после выхода в адресной строке
 * оставался #/vacancies (баг: на логине показывался прошлый раздел).
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";

import { AuthProvider, useAuth } from "./auth";

afterEach(cleanup);

function Probe() {
  const { logout } = useAuth();
  return (
    <button type="button" onClick={logout}>
      logout
    </button>
  );
}

it("logout очищает хэш вкладки", () => {
  window.location.hash = "#/vacancies";
  render(
    <AuthProvider>
      <Probe />
    </AuthProvider>,
  );
  fireEvent.click(screen.getByText("logout"));
  expect(window.location.hash).toBe("");
});
