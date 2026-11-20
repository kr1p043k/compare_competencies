/**
 * MyProfilesCard: состояния загрузки/ошибки, созданные профили, валидация
 * и создание (7.5.4). Всё состояние - пропсы, сеть не нужна.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MyProfilesCard } from "./MyProfilesCard";

afterEach(cleanup);

type CardProps = React.ComponentProps<typeof MyProfilesCard>;

function props(over: Partial<CardProps> = {}): CardProps {
  return {
    optionsLoading: false,
    optionsError: null,
    createdProfiles: [],
    cpName: "",
    onCpNameChange: vi.fn(),
    cpNameValid: false,
    cpLevel: "middle",
    onCpLevelChange: vi.fn(),
    cpBase: "",
    onCpBaseChange: vi.fn(),
    baseProfiles: ["base"],
    compCodes: ["ПК-1"],
    compSearch: "",
    onCompSearchChange: vi.fn(),
    filteredCustomComps: [],
    onToggleCompCode: vi.fn(),
    draftCode: "",
    onDraftCodeChange: vi.fn(),
    draftTitle: "",
    onDraftTitleChange: vi.fn(),
    draftKnowledge: "",
    onDraftKnowledgeChange: vi.fn(),
    draftAbilities: "",
    onDraftAbilitiesChange: vi.fn(),
    draftSkills: "",
    onDraftSkillsChange: vi.fn(),
    onAddDraft: vi.fn(),
    onRemoveDraft: vi.fn(),
    newComps: [],
    cpTech: "",
    onCpTechChange: vi.fn(),
    triedCreate: false,
    totalComps: 0,
    createError: null,
    creating: false,
    onCreateProfile: vi.fn(),
    ...over,
  };
}

describe("MyProfilesCard", () => {
  it("показывает скелетон при загрузке опций", () => {
    render(<MyProfilesCard {...props({ optionsLoading: true })} />);
    expect(screen.getByLabelText("Загрузка опций профилей")).toBeTruthy();
  });

  it("показывает ErrorState при ошибке опций", () => {
    render(<MyProfilesCard {...props({ optionsError: "нет опций" })} />);
    expect(screen.getByText("нет опций")).toBeTruthy();
  });

  it("рендерит созданные профили с бейджами", () => {
    render(
      <MyProfilesCard
        {...props({
          createdProfiles: [
            { name: "my-prof", target_level: "senior", competencies: 5, created_new: 2, technologies: 1 },
          ],
        })}
      />,
    );
    expect(screen.getByText("Создано в этой сессии (1)")).toBeTruthy();
    expect(screen.getByText("my-prof")).toBeTruthy();
    expect(screen.getByText("компетенций: 5")).toBeTruthy();
    expect(screen.getByText("новых: 2")).toBeTruthy();
    expect(screen.getByText("технологий: 1")).toBeTruthy();
  });

  it("показывает ошибку создания", () => {
    render(<MyProfilesCard {...props({ createError: "имя занято" })} />);
    expect(screen.getByText("имя занято")).toBeTruthy();
  });

  it("кнопка создания зовёт onCreateProfile", () => {
    const onCreateProfile = vi.fn();
    render(<MyProfilesCard {...props({ cpName: "x", onCreateProfile })} />);
    const btn = screen.getByRole("button", { name: /Создать профиль/i });
    fireEvent.click(btn);
    expect(onCreateProfile).toHaveBeenCalled();
  });
});
