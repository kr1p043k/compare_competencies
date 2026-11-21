import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RegionCombobox } from "./RegionCombobox";

afterEach(cleanup);

const regions = [
  { id: 1, name: "Moscow" },
  { id: 2, name: "Kazan" },
];

describe("RegionCombobox", () => {
  it("shows the placeholder when nothing is selected", () => {
    render(<RegionCombobox regions={regions} value="" onChange={() => {}} />);
    const trigger = screen.getByRole("combobox");
    expect(trigger.textContent).toContain("...");
  });

  it("shows the selected region name", () => {
    render(<RegionCombobox regions={regions} value="2" onChange={() => {}} />);
    expect(screen.getByRole("combobox").textContent).toContain("Kazan");
  });

  it("calls onChange with the region id when an item is picked", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<RegionCombobox regions={regions} value="" onChange={onChange} />);

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByText("Moscow"));

    expect(onChange).toHaveBeenCalledWith("1");
  });
});