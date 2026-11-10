import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { useFocusTrap } from "./useFocusTrap";

function Harness({ onClose }: { onClose?: () => void }) {
  const [open, setOpen] = useState(false);
  const ref = useFocusTrap<HTMLDivElement>(open, () => {
    setOpen(false);
    onClose?.();
  });
  return (
    <div>
      <button onClick={() => setOpen(true)}>открыть</button>
      {open && (
        <div ref={ref} role="dialog" aria-modal="true" tabIndex={-1}>
          <button>первый</button>
          <button>последний</button>
        </div>
      )}
    </div>
  );
}

function open_() {
  fireEvent.click(screen.getByText("открыть"));
}

describe("useFocusTrap", () => {
  it("ставит фокус на первый элемент при открытии", () => {
    render(<Harness />);
    open_();
    expect(document.activeElement).toBe(screen.getByText("первый"));
  });

  it("Tab с последнего переводит на первый", () => {
    render(<Harness />);
    open_();
    const last = screen.getByText("последний");
    last.focus();
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Tab" });
    expect(document.activeElement).toBe(screen.getByText("первый"));
  });

  it("Shift+Tab с первого переводит на последний", () => {
    render(<Harness />);
    open_();
    const first = screen.getByText("первый");
    first.focus();
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(screen.getByText("последний"));
  });

  it("Escape вызывает onClose", () => {
    const onClose = vi.fn();
    render(<Harness onClose={onClose} />);
    open_();
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("возвращает фокус на элемент-триггер при закрытии", () => {
    render(<Harness />);
    const opener = screen.getByText("открыть");
    opener.focus();
    open_();
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(document.activeElement).toBe(opener);
  });
});
