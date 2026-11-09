import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import { reportError } from "../../lib/errors";
import { ShapExplanationCard } from "./ShapExplanationCard";

// Сетевой слой подменён: тест проверяет рендер и поведение карточки.
vi.mock("../api", () => ({ api: vi.fn() }));
// reportError логирует в консоль и на бэкенд - в тестах он подменён.
vi.mock("../../lib/errors", () => ({ reportError: vi.fn() }));

/** Ответ /results/ml-explanations/base: навык с вкладами, навык без вкладов
 *  и незнакомый признак (фолбэк подписи). */
const PAYLOAD = {
  profile: "base",
  model: "ltr_xgboost_shap",
  explanations: [
    {
      skill: "SQL",
      score: 87.25,
      explanation: "Часто встречается рядом с навыками профиля.",
      contributions: [
        { feature: "co_occurrence", value: 0.4, direction: "positive", feature_value: 3 },
        {
          feature: "student_skills_count",
          value: -0.2,
          direction: "negative",
          feature_value: 12,
        },
        { feature: "custom_metric", value: 0.1, direction: "positive", feature_value: 0.5 },
      ],
    },
    {
      skill: "Docker",
      score: 54,
      explanation: "Инфраструктурный навык из другой категории.",
      contributions: [],
    },
  ],
};

async function renderCard() {
  const utils = render(<ShapExplanationCard profile="base" />);
  await screen.findByText("SQL");
  return utils;
}

beforeEach(() => {
  vi.mocked(api).mockReset();
  vi.mocked(api).mockResolvedValue(PAYLOAD as never);
});

describe("ShapExplanationCard", () => {
  it("запрашивает /results/ml-explanations/{profile} один раз при монтировании", async () => {
    await renderCard();

    expect(vi.mocked(api)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(api).mock.calls[0][0]).toBe("/results/ml-explanations/base");
  });

  it("рисует навыки со скором, RU-подписями признаков и знаковыми SHAP-значениями", async () => {
    await renderCard();

    expect(screen.getByText("Оценка 87/100")).toBeInTheDocument();
    expect(screen.getByText("Оценка 54/100")).toBeInTheDocument();
    // RU-подписи из карты признаков; незнакомый ключ - как есть.
    expect(screen.getByText("Совместная встречаемость")).toBeInTheDocument();
    expect(screen.getByText("Размер профиля")).toBeInTheDocument();
    expect(screen.getByText("custom_metric")).toBeInTheDocument();
    expect(screen.getByText("+0.4000")).toBeInTheDocument();
    expect(screen.getByText("-0.2000")).toBeInTheDocument();
  });

  it("красит направление бейджем токен-тоном и масштабирует Meter по максимуму |SHAP|", async () => {
    const { container } = await renderCard();

    // Положительных вклада два (co_occurrence и custom_metric).
    expect(screen.getAllByText("повышает")).toHaveLength(2);
    expect(screen.getByText("снижает")).toBeInTheDocument();

    // Максимум |SHAP| в строке SQL = 0.4: co_occurrence 100%, student_skills_count 50%.
    const positiveFill = container.querySelector(".bg-success") as HTMLElement;
    const negativeFill = container.querySelector(".bg-destructive") as HTMLElement;
    expect(positiveFill.style.width).toBe("100%");
    expect(negativeFill.style.width).toBe("50%");
  });

  it("честно сообщает про навык, которому не рассчитаны вклады", async () => {
    await renderCard();

    expect(
      screen.getByText("Вклады признаков для этого навыка не рассчитаны."),
    ).toBeInTheDocument();
  });

  it("пустой payload с message показывает empty-стейт с текстом бэкенда", async () => {
    vi.mocked(api).mockResolvedValue({
      profile: "base",
      model: "ltr_xgboost_shap",
      message: "ML-модель не обучена.",
      explanations: [],
    } as never);

    render(<ShapExplanationCard profile="base" />);

    expect(await screen.findByText("Нет ML-объяснений")).toBeInTheDocument();
    expect(screen.getByText("ML-модель не обучена.")).toBeInTheDocument();
    expect(screen.queryByText("SQL")).not.toBeInTheDocument();
  });

  it("ошибка сети показывает alert, повтор запрашивает данные заново", async () => {
    const user = userEvent.setup();
    vi.mocked(api)
      .mockRejectedValueOnce(new Error("network down"))
      .mockResolvedValue(PAYLOAD as never);

    render(<ShapExplanationCard profile="base" />);

    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(
      screen.getByText("Не удалось загрузить объяснения ML-модели."),
    ).toBeInTheDocument();
    expect(vi.mocked(reportError)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(reportError).mock.calls[0][0]).toBe("ShapExplanationCard:ml-explanations");

    await user.click(screen.getByRole("button", { name: "Повторить" }));

    expect(await screen.findByText("SQL")).toBeInTheDocument();
    expect(vi.mocked(api)).toHaveBeenCalledTimes(2);
  });

  it("смена профиля перезапрашивает объяснения", async () => {
    const { rerender } = await renderCard();

    rerender(<ShapExplanationCard profile="dc" />);
    await screen.findByText("SQL");

    expect(vi.mocked(api)).toHaveBeenCalledTimes(2);
    expect(vi.mocked(api).mock.calls[1][0]).toBe("/results/ml-explanations/dc");
  });
});
