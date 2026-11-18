import { describe, expect, it } from "vitest";

import { PROFILE_LABELS, profileLabel } from "./profiles";

describe("profileLabel", () => {
  it("returns the shared label for known profiles", () => {
    expect(profileLabel("base")).toBe("Base (junior)");
    expect(profileLabel("dc")).toBe(PROFILE_LABELS.dc);
    expect(profileLabel("top_dc")).toBe("Top (senior)");
  });

  it("falls back to the raw id for unknown profiles", () => {
    expect(profileLabel("custom")).toBe("custom");
  });
});
