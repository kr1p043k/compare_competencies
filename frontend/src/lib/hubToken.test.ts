import { describe, expect, it } from "vitest";

import { extractHubToken } from "./hubToken";

describe("extractHubToken", () => {
  it("берёт токен из query и убирает его из адреса", () => {
    expect(extractHubToken("?token=abc", "")).toEqual({
      token: "abc",
      search: "",
      hash: "",
    });
  });

  it("сохраняет остальные query-параметры", () => {
    expect(extractHubToken("?token=abc&tab=publications", "#/logs")).toEqual({
      token: "abc",
      search: "?tab=publications",
      hash: "#/logs",
    });
  });

  it("берёт токен из фрагмента", () => {
    expect(extractHubToken("", "#token=abc")).toEqual({
      token: "abc",
      search: "",
      hash: "",
    });
  });

  it("сохраняет маршрут, если токен пришёл во фрагменте с query", () => {
    expect(extractHubToken("", "#/teacher?token=abc")).toEqual({
      token: "abc",
      search: "",
      hash: "#/teacher",
    });
  });

  it("не трогает обычный маршрут без токена", () => {
    expect(extractHubToken("", "#/teacher")).toEqual({
      token: null,
      search: "",
      hash: "#/teacher",
    });
  });

  it("не принимает пустой токен", () => {
    expect(extractHubToken("?token=", "#token=")).toEqual({
      token: null,
      search: "?token=",
      hash: "#token=",
    });
  });

  it("query важнее фрагмента, но токен вычищается из обоих мест", () => {
    expect(extractHubToken("?token=from-query", "#token=from-hash")).toEqual({
      token: "from-query",
      search: "",
      hash: "",
    });
  });
});
