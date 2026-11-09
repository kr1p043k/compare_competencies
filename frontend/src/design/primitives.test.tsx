import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FlaskConical } from "lucide-react";

import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  Meter,
  MetricPair,
  PageHeader,
  Skeleton,
  Sparkbars,
  StatCard,
  TableFrame,
  inputClass,
  tableHeadClass,
  tableRowClass,
} from "./primitives";

describe("MetricPair", () => {
  it("рисует основное значение и парную метрику с подписью", () => {
    render(
      <MetricPair
        primaryValue="55%"
        primaryTitle="Покрытие"
        secondaryValue="40%"
        secondaryLabel="кач."
        secondaryTitle="Взвешенное покрытие"
      />,
    );
    expect(screen.getByText("55%")).toBeInTheDocument();
    expect(screen.getByText("кач.: 40%")).toBeInTheDocument();
    expect(screen.getByTitle("Покрытие")).toBeInTheDocument();
    expect(screen.getByTitle("Взвешенное покрытие")).toBeInTheDocument();
  });

  it("показывает только основное значение, когда пары нет в payload", () => {
    render(<MetricPair primaryValue="55%" />);
    expect(screen.getByText("55%")).toBeInTheDocument();
    // Одиночный процент без «кач.» - допустимо только если поля реально нет.
    expect(screen.queryByText(/кач\./)).not.toBeInTheDocument();
  });

  it("не рисует парную метрику при null", () => {
    render(<MetricPair primaryValue="55%" secondaryValue={null} secondaryLabel="кач." />);
    expect(screen.getByText("55%")).toBeInTheDocument();
    expect(screen.queryByText(/кач\./)).not.toBeInTheDocument();
  });

  it("рисует парное значение без подписи, если label не передан", () => {
    render(<MetricPair primaryValue="55%" secondaryValue="40%" />);
    expect(screen.getByText("55%")).toBeInTheDocument();
    expect(screen.getByText("40%")).toBeInTheDocument();
    expect(screen.queryByText(/: 40%/)).not.toBeInTheDocument();
  });
});

describe("PageHeader", () => {
  it("рисует заголовок h1, описание и действия", () => {
    render(
      <PageHeader
        title="Научные тренды"
        description="Навыки из публикаций"
        actions={<button type="button">Обновить</button>}
      />,
    );
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Научные тренды");
    expect(screen.getByText("Навыки из публикаций")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Обновить" })).toBeInTheDocument();
  });

  it("показывает иконку в квадрате, когда она передана", () => {
    const { container } = render(<PageHeader icon={FlaskConical} title="Научные тренды" />);
    expect(container.querySelector("svg")).not.toBeNull();
  });

  it("не рисует иконку, когда она не передана", () => {
    const { container } = render(<PageHeader title="Публикации" />);
    expect(container.querySelector("svg")).toBeNull();
  });
});

describe("Meter", () => {
  it("обрезает значение в 0..100 и красит трек/заливку переданными классами", () => {
    const { container } = render(
      <Meter value={150} trackClassName="bg-warning/40" fillClassName="bg-info" />,
    );
    const track = container.firstElementChild as HTMLElement;
    const fill = track.firstElementChild as HTMLElement;
    expect(fill.style.width).toBe("100%");
    expect(track.className).toContain("bg-warning/40");
    expect(fill.className).toContain("bg-info");
  });

  it("держит дефолтные цвета и 0% при undefined/NaN", () => {
    const { container } = render(<Meter value={undefined as unknown as number} />);
    const track = container.firstElementChild as HTMLElement;
    const fill = track.firstElementChild as HTMLElement;
    expect(track.className).toContain("bg-muted");
    expect(fill.className).toContain("bg-primary");
    expect(fill.style.width).toBe("0%");
  });

  it("не уходит в минус при отрицательном значении", () => {
    const { container } = render(<Meter value={-20} />);
    const fill = (container.firstElementChild as HTMLElement).firstElementChild as HTMLElement;
    expect(fill.style.width).toBe("0%");
  });
});

describe("Sparkbars", () => {
  const items = [
    { key: "a", value: 4, label: "03.02", title: "2026-02-03: 4" },
    { key: "b", value: 2, label: "04.02", title: "2026-02-04: 2" },
  ];

  it("считает высоту столбиков относительно максимума", () => {
    const { container } = render(<Sparkbars items={items} />);
    const bars = Array.from(
      container.querySelectorAll<HTMLElement>("div[title^='2026-02']"),
    );
    expect(bars).toHaveLength(2);
    expect(bars[0].style.height).toBe("100%");
    expect(bars[1].style.height).toBe("50%");
  });

  it("рисует значение над столбиком и подпись под ним", () => {
    render(<Sparkbars items={items} />);
    expect(screen.getByText("4")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("03.02")).toBeInTheDocument();
    expect(screen.getByText("04.02")).toBeInTheDocument();
  });

  it("не падает на пустом наборе", () => {
    const { container } = render(<Sparkbars items={[]} />);
    expect(container.querySelectorAll("div[title]")).toHaveLength(0);
  });
});

describe("Button", () => {
  it("вариант success красит кнопку в токен success", () => {
    render(<Button variant="success">Создать</Button>);
    const btn = screen.getByRole("button", { name: "Создать" });
    expect(btn.className).toContain("bg-success");
    expect(btn.className).toContain("text-success-foreground");
  });

  it("вариант danger остаётся на токене destructive", () => {
    render(<Button variant="danger">Удалить</Button>);
    const btn = screen.getByRole("button", { name: "Удалить" });
    expect(btn.className).toContain("bg-destructive");
  });
});

/* ---------------------------------------------------------------------------
 * Гварды дизайн-системы (backlog #56): примитивы обязаны краситься только
 * токенами темы (bg-card, text-foreground, border-border и т.д.).
 * Захардкоженная палитра Tailwind ломает тёмную тему: у таких классов нет
 * CSS-переменных в theme.css, и в .dark они остаются светлыми.
 * -------------------------------------------------------------------------- */
const HARDCODED_PALETTE_RE =
  /(bg|text|border)-(gray|slate|zinc|violet|indigo|emerald|amber|red|sky|blue|green|yellow|pink|rose)-/;

/* Список class-атрибутов поддерева (включая корень): нужен, чтобы сравнить
 * разметку светлого и тёмного рендера одного и того же композита. */
function classAttrsOf(root: Element): Array<string | null> {
  return [
    root.getAttribute("class"),
    ...Array.from(root.querySelectorAll("[class]")).map((el) => el.getAttribute("class")),
  ];
}

/* Элементы поддерева, в чьих классах проскочила хардкод-палитра. */
function paletteOffenders(root: HTMLElement): Element[] {
  return Array.from(root.querySelectorAll("[class]")).filter((el) =>
    HARDCODED_PALETTE_RE.test(el.getAttribute("class") ?? ""),
  );
}

describe("Button: варианты и размеры", () => {
  const VARIANTS = [
    { variant: "primary", bg: "bg-primary", text: "text-primary-foreground" },
    { variant: "secondary", bg: "bg-surface", text: "text-foreground" },
    { variant: "ghost", bg: "bg-transparent", text: "text-muted-foreground" },
    { variant: "danger", bg: "bg-destructive", text: "text-destructive-foreground" },
    { variant: "success", bg: "bg-success", text: "text-success-foreground" },
  ] as const;

  it.each(VARIANTS)("вариант $variant красит токенами $bg и $text", ({ variant, bg, text }) => {
    render(<Button variant={variant}>Действие</Button>);
    const btn = screen.getByRole("button", { name: "Действие" });
    expect(btn.className).toContain(bg);
    expect(btn.className).toContain(text);
    expect(btn.className).not.toMatch(HARDCODED_PALETTE_RE);
  });

  it("варианты дают попарно различающиеся классы", () => {
    const classNames = VARIANTS.map((row) => {
      const { container } = render(<Button variant={row.variant}>Действие</Button>);
      return (container.firstElementChild as HTMLElement).className;
    });
    expect(new Set(classNames).size).toBe(VARIANTS.length);
  });

  it("размеры sm и md дают разные классы (h-8 против h-10)", () => {
    const { container: cSm } = render(<Button size="sm">Действие</Button>);
    const { container: cMd } = render(<Button size="md">Действие</Button>);
    const sm = (cSm.firstElementChild as HTMLElement).className;
    const md = (cMd.firstElementChild as HTMLElement).className;
    expect(sm).toContain("h-8");
    expect(md).toContain("h-10");
    expect(sm).not.toBe(md);
  });

  it("loading дизейбит кнопку и рисует спиннер без палитры", () => {
    const { container } = render(<Button loading>Загрузка</Button>);
    const btn = screen.getByRole("button", { name: "Загрузка" });
    expect(btn).toBeDisabled();
    expect(btn.getAttribute("aria-busy")).toBe("true");
    const spinner = container.querySelector("span[aria-hidden='true']");
    expect(spinner).not.toBeNull();
    expect((spinner as HTMLElement).className).toContain("animate-spin");
    expect(btn.className).not.toMatch(HARDCODED_PALETTE_RE);
  });
});

describe("Badge: тоны", () => {
  const TONES = [
    { tone: "neutral", bg: "bg-muted", text: "text-muted-foreground" },
    { tone: "primary", bg: "bg-primary-soft", text: "text-primary-soft-foreground" },
    { tone: "success", bg: "bg-success-soft", text: "text-success" },
    { tone: "warning", bg: "bg-warning-soft", text: "text-warning" },
    { tone: "danger", bg: "bg-destructive/10", text: "text-destructive" },
    { tone: "info", bg: "bg-info-soft", text: "text-info" },
  ] as const;

  it.each(TONES)("тон $tone красит бейдж токенами $bg и $text", ({ tone, bg, text }) => {
    const { container } = render(<Badge tone={tone}>Метка</Badge>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain(bg);
    expect(root.className).toContain(text);
    expect(root.className).not.toMatch(HARDCODED_PALETTE_RE);
  });

  it("все тона дают попарно различающиеся классы", () => {
    const classNames = TONES.map((row) => {
      const { container } = render(<Badge tone={row.tone}>Метка</Badge>);
      return (container.firstElementChild as HTMLElement).className;
    });
    expect(new Set(classNames).size).toBe(TONES.length);
  });
});

describe("Примитивы: рендер без падения и без хардкод-палитры", () => {
  it.each([
    { name: "Button", ui: <Button>Кнопка</Button> },
    { name: "Badge", ui: <Badge tone="danger">Ошибка</Badge> },
    { name: "Card", ui: <Card>Содержимое</Card> },
    { name: "CardHeader", ui: <CardHeader title="Заголовок" /> },
    { name: "CardBody", ui: <CardBody>Контент</CardBody> },
    { name: "PageHeader", ui: <PageHeader title="Страница" /> },
    { name: "EmptyState", ui: <EmptyState title="Пусто" /> },
    { name: "ErrorState", ui: <ErrorState /> },
    { name: "StatCard", ui: <StatCard label="Отзывы" value="12" /> },
    { name: "TableFrame", ui: <TableFrame>Таблица</TableFrame> },
    { name: "Skeleton", ui: <Skeleton /> },
    { name: "MetricPair", ui: <MetricPair primaryValue="55%" /> },
  ])("$name: корень с классами, без палитры", ({ ui }) => {
    const { container } = render(ui);
    const root = container.firstElementChild as HTMLElement;
    expect(root).not.toBeNull();
    expect(root.className.trim()).not.toBe("");
    expect(root.className).not.toMatch(HARDCODED_PALETTE_RE);
  });
});

describe("inputClass и табличные константы", () => {
  it.each([
    {
      name: "inputClass",
      value: inputClass,
      mustContain: ["border", "border-input", "bg-input-background", "text-foreground"],
    },
    {
      name: "tableHeadClass",
      value: tableHeadClass,
      mustContain: ["bg-muted", "text-muted-foreground"],
    },
    {
      name: "tableRowClass",
      value: tableRowClass,
      mustContain: ["border-border", "hover:bg-muted/40"],
    },
  ])("$name: непустая, с токенами и без палитры", ({ value, mustContain }) => {
    expect(value.trim()).not.toBe("");
    for (const token of mustContain) {
      expect(value).toContain(token);
    }
    expect(value).not.toMatch(HARDCODED_PALETTE_RE);
  });
});

describe("Тёмная тема (smoke)", () => {
  const Composite = () => (
    <Card>
      <CardHeader
        title="Покрытие навыков"
        description="Сводка по источникам"
        actions={<Badge tone="success">активно</Badge>}
      />
      <CardBody>
        <Button variant="secondary">Синхронизировать</Button>
        <EmptyState title="Нет данных" description="Источники пока не подключены" />
      </CardBody>
    </Card>
  );

  it("композит в обёртке .dark рендерится теми же классами и без inline-стилей", () => {
    const light = render(<Composite />);
    const dark = render(
      <div className="dark">
        <Composite />
      </div>,
    );

    const cardLight = light.container.firstElementChild as HTMLElement;
    const wrapper = dark.container.firstElementChild as HTMLElement;
    expect(wrapper.getAttribute("class")).toBe("dark");
    const cardDark = wrapper.firstElementChild as HTMLElement;

    // Рендер не упал: заголовок, бейдж, кнопка и пустое состояние на месте.
    expect(within(cardDark).getByRole("heading", { level: 3 })).toHaveTextContent(
      "Покрытие навыков",
    );
    expect(within(cardDark).getByText("активно")).toBeInTheDocument();
    expect(
      within(cardDark).getByRole("button", { name: "Синхронизировать" }),
    ).toBeInTheDocument();
    expect(within(cardDark).getByText("Нет данных")).toBeInTheDocument();

    // Токены держат темизацию через CSS-переменные: классы от .dark не меняются.
    expect(classAttrsOf(cardDark)).toEqual(classAttrsOf(cardLight));

    // Inline-стилей от примитивов не ждём, палитра нигде не должна проскочить.
    expect(dark.container.querySelectorAll("[style]")).toHaveLength(0);
    expect(paletteOffenders(dark.container)).toHaveLength(0);
  });
});
