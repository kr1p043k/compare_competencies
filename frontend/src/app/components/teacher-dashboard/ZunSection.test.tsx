/**
 * ZunSection: презентационный шаг «Компетенции и ЗУН» (7.5.4).
 * Всё состояние приходит пропсами - сеть и роутер не нужны.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { CompFilter } from "../teacherFilters";
import { ZunSection } from "./ZunSection";
import type { Competency, DisciplineDetail } from "./types";

afterEach(cleanup);

const COMP_WITH_KSA: Competency = {
  id: "c1",
  code: "ПК-2.1",
  name: "Разработка веб-приложений",
  skills: ["HTML", "CSS", "JS"],
  ksa: {
    knowledge: [{ id: "k1", text: "Протокол HTTP" }],
    abilities: [{ id: "a1", text: "Верстать адаптивно" }],
    skills: [{ id: "s1", text: "React" }],
  },
};

const COMP_PLAIN: Competency = {
  id: "c2",
  code: "ПК-3.2",
  name: "",
  skills: ["SQL", "Индексы"],
};

const SELECTED: DisciplineDetail = {
  id: "d1",
  name: "Веб-технологии",
  competencies: [COMP_WITH_KSA, COMP_PLAIN],
};

type SectionProps = React.ComponentProps<typeof ZunSection>;

function props(over: Partial<SectionProps> = {}): SectionProps {
  return {
    selected: SELECTED,
    compForm: false,
    setCompForm: vi.fn(),
    compErr: {},
    setCompErr: vi.fn(),
    compCode: "",
    setCompCode: vi.fn(),
    compName: "",
    setCompName: vi.fn(),
    compSaving: false,
    saveCompetency: vi.fn().mockResolvedValue(undefined),
    compSearch: "",
    setCompSearch: vi.fn(),
    compFilter: "all" as CompFilter,
    setCompFilter: vi.fn(),
    compCounts: null,
    compsVisible: [COMP_WITH_KSA, COMP_PLAIN],
    compFiltersActive: false,
    openComp: null,
    setOpenComp: vi.fn(),
    zunForm: null,
    setZunForm: vi.fn(),
    zunMsg: "",
    setZunMsg: vi.fn(),
    zunSaving: false,
    addZunEntry: vi.fn().mockResolvedValue(undefined),
    ksaEditing: null,
    setKsaEditing: vi.fn(),
    saveKsaEdit: vi.fn().mockResolvedValue(undefined),
    delKsa: vi.fn().mockResolvedValue(undefined),
    ...over,
  };
}

describe("ZunSection", () => {
  it("минимальный фикстур: кнопка создания, счётчик и строки компетенций", () => {
    render(<ZunSection {...props()} />);
    expect(screen.getByText("+ Компетенция")).toBeTruthy();
    expect(screen.getByText("2 из 2")).toBeTruthy();
    expect(screen.getByText("ПК-2.1")).toBeTruthy();
    expect(screen.getByText("Разработка веб-приложений")).toBeTruthy();
    expect(screen.getByText("3 пункта")).toBeTruthy();
    expect(screen.getByText("2 пункта")).toBeTruthy();
    // Все свёрнуто: групп ЗУН в DOM нет.
    expect(screen.queryByText("Знания (1)")).toBeNull();
  });

  it("пустое состояние: «По фильтру ничего не найдено» и кнопка сброса", () => {
    const setCompSearch = vi.fn();
    const setCompFilter = vi.fn();
    render(
      <ZunSection
        {...props({
          selected: { id: "d1", name: "Веб-технологии", competencies: [] },
          compsVisible: [],
          setCompSearch,
          setCompFilter,
        })}
      />,
    );
    expect(screen.getByText(/По фильтру ничего не найдено/)).toBeTruthy();
    expect(screen.getByText("0 из 0")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "сбросить" }));
    expect(setCompSearch).toHaveBeenCalledWith("");
    expect(setCompFilter).toHaveBeenCalledWith("all");
  });

  it("счётчики фильтров подставляются из compCounts", () => {
    render(
      <ZunSection
        {...props({ compCounts: { all: 2, "no-ksa": 1, "with-ksa": 1 } })}
      />,
    );
    expect(screen.getByText("Все (2)")).toBeTruthy();
    expect(screen.getByText("Без ЗУН (1)")).toBeTruthy();
    expect(screen.getByText("С ЗУН (1)")).toBeTruthy();
  });

  it("клик по заголовку разворачивает компетенцию", () => {
    const setOpenComp = vi.fn();
    render(<ZunSection {...props({ setOpenComp })} />);
    fireEvent.click(screen.getByText("ПК-2.1").closest("[role='button']")!);
    expect(setOpenComp).toHaveBeenCalledWith("c1");
  });

  it("клавиатура Enter на заголовке тоже разворачивает", () => {
    const setOpenComp = vi.fn();
    render(<ZunSection {...props({ setOpenComp })} />);
    fireEvent.keyDown(screen.getByText("ПК-2.1").closest("[role='button']")!, {
      key: "Enter",
    });
    expect(setOpenComp).toHaveBeenCalledWith("c1");
  });

  it("развёрнутая компетенция показывает группы ЗУН и действия", () => {
    const setKsaEditing = vi.fn();
    render(
      <ZunSection {...props({ openComp: "c1", setKsaEditing })} />,
    );
    expect(screen.getByRole("button", { expanded: true })).toBeTruthy();
    expect(screen.getByText("Знания (1)")).toBeTruthy();
    expect(screen.getByText("Умения (1)")).toBeTruthy();
    expect(screen.getByText("Навыки (1)")).toBeTruthy();
    expect(screen.getByText("Протокол HTTP")).toBeTruthy();
    expect(screen.getByText("Верстать адаптивно")).toBeTruthy();
    expect(screen.getByText("React")).toBeTruthy();

    fireEvent.click(screen.getAllByText("Изменить")[0]);
    expect(setKsaEditing).toHaveBeenCalledWith({
      id: "k1",
      text: "Протокол HTTP",
    });
  });

  it("компетенция без ksa показывает плоский список навыков", () => {
    render(
      <ZunSection
        {...props({ compsVisible: [COMP_PLAIN], openComp: "c2" })}
      />,
    );
    expect(screen.getByText("SQL")).toBeTruthy();
    expect(screen.getByText("Индексы")).toBeTruthy();
  });

  it("«+ ЗУН» открывает форму пункта для компетенции", () => {
    const setZunForm = vi.fn();
    const setZunMsg = vi.fn();
    const setOpenComp = vi.fn();
    render(
      <ZunSection {...props({ setZunForm, setZunMsg, setOpenComp })} />,
    );
    fireEvent.click(screen.getAllByText("+ ЗУН")[0]);
    expect(setZunForm).toHaveBeenCalledWith({
      compId: "c1",
      ksaType: "skills",
      text: "",
    });
    expect(setZunMsg).toHaveBeenCalledWith("");
    expect(setOpenComp).toHaveBeenCalledWith("c1");
  });

  it("форма ЗУН: пустой текст блокирует «Добавить»", () => {
    const addZunEntry = vi.fn().mockResolvedValue(undefined);
    render(
      <ZunSection
        {...props({
          openComp: "c1",
          zunForm: { compId: "c1", ksaType: "knowledge", text: "" },
          addZunEntry,
        })}
      />,
    );
    expect(
      screen.getByPlaceholderText("Текст пункта: знание, умение или навык…"),
    ).toBeTruthy();
    const add = screen.getByRole("button", { name: "Добавить" }) as HTMLButtonElement;
    expect(add.disabled).toBe(true);
    fireEvent.click(add);
    expect(addZunEntry).not.toHaveBeenCalled();
  });

  it("форма ЗУН с текстом вызывает addZunEntry", () => {
    const addZunEntry = vi.fn().mockResolvedValue(undefined);
    render(
      <ZunSection
        {...props({
          openComp: "c1",
          zunForm: { compId: "c1", ksaType: "knowledge", text: "HTTP/2" },
          addZunEntry,
        })}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Добавить" }));
    expect(addZunEntry).toHaveBeenCalledTimes(1);
  });

  it("форма создания компетенции показывает ошибки и сохраняет", () => {
    const saveCompetency = vi.fn().mockResolvedValue(undefined);
    const setCompForm = vi.fn();
    render(
      <ZunSection
        {...props({
          compForm: true,
          compErr: { code: "Укажите код", msg: "Код уже занят" },
          saveCompetency,
          setCompForm,
        })}
      />,
    );
    expect(screen.getByLabelText("Код компетенции")).toBeTruthy();
    expect(screen.getByLabelText("Название компетенции")).toBeTruthy();
    expect(screen.getByText("Укажите код")).toBeTruthy();
    expect(screen.getByText("Код уже занят")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Создать" }));
    expect(saveCompetency).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Отмена" }));
    expect(setCompForm).toHaveBeenCalledWith(false);
  });

  it("«Сбросить» виден только при активных фильтрах и чистит их", () => {
    const setCompSearch = vi.fn();
    const setCompFilter = vi.fn();
    render(
      <ZunSection
        {...props({
          compFiltersActive: true,
          setCompSearch,
          setCompFilter,
        })}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Сбросить" }));
    expect(setCompSearch).toHaveBeenCalledWith("");
    expect(setCompFilter).toHaveBeenCalledWith("all");
  });
});
