/**
 * ZunSection: форма компетенции, фильтры, аккордеон ЗУН (7.5.4).
 * Всё состояние - из пропсов, сеть не нужна.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ZunSection } from "./ZunSection";
import type { Competency, DisciplineDetail } from "./types";

afterEach(cleanup);

const SELECTED: DisciplineDetail = { id: "d1", name: "Веб-технологии", competencies: [] };

const COMP_WITH_KSA: Competency = {
  id: "c1",
  code: "ПК-1",
  name: "Разработка",
  skills: ["python"],
  ksa: {
    knowledge: [{ id: "k1", text: "HTTP" }],
    abilities: [],
    skills: [{ id: "s1", text: "FastAPI" }],
  },
};
const COMP_PLAIN: Competency = { id: "c2", code: "ПК-2", name: "", skills: ["sql", "git"] };

type ZunProps = React.ComponentProps<typeof ZunSection>;

function props(over: Partial<ZunProps> = {}): ZunProps {
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
    compFilter: "all",
    setCompFilter: vi.fn(),
    compCounts: { all: 2, "no-ksa": 1, "with-ksa": 1 },
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
  it("кнопка «+ Компетенция» открывает форму", () => {
    const setCompForm = vi.fn();
    render(<ZunSection {...props({ setCompForm })} />);
    fireEvent.click(screen.getByText("+ Компетенция"));
    expect(setCompForm).toHaveBeenCalledWith(true);
  });

  it("форма показывает ошибки полей и зовёт saveCompetency", () => {
    const saveCompetency = vi.fn().mockResolvedValue(undefined);
    render(
      <ZunSection {...props({ compForm: true, compErr: { code: "обязателен" }, saveCompetency })} />,
    );
    expect(screen.getByText("обязателен")).toBeTruthy();
    fireEvent.click(screen.getByText("Создать"));
    expect(saveCompetency).toHaveBeenCalled();
  });

  it("рендерит компетенции и раскрывает ЗУН по клику", () => {
    const setOpenComp = vi.fn();
    render(<ZunSection {...props({ setOpenComp })} />);
    expect(screen.getByText("ПК-1")).toBeTruthy();
    expect(screen.getByText("ПК-2")).toBeTruthy();
    fireEvent.click(screen.getByText("ПК-1"));
    expect(setOpenComp).toHaveBeenCalledWith("c1");
  });

  it("раскрытая компетенция с KSA показывает группы", () => {
    render(<ZunSection {...props({ openComp: "c1" })} />);
    expect(screen.getByText(/Знания/)).toBeTruthy();
    expect(screen.getByText(/Навыки/)).toBeTruthy();
    expect(screen.getByText("HTTP")).toBeTruthy();
    expect(screen.getByText("FastAPI")).toBeTruthy();
  });

  it("поиск по компетенциям зовёт setCompSearch", () => {
    const setCompSearch = vi.fn();
    render(<ZunSection {...props({ setCompSearch })} />);
    fireEvent.change(screen.getByLabelText("Поиск компетенции"), { target: { value: "ПК-1" } });
    expect(setCompSearch).toHaveBeenCalledWith("ПК-1");
  });
});
