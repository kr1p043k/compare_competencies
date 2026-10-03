/** Единые подписи профилей: BASE/DC/TOP_DC во всех компонентах одинаковы. */
export const PROFILE_LABELS: Record<string, string> = {
  base: "Base (junior)",
  dc: "Data Scientist (middle)",
  top_dc: "Top (senior)",
};

export function profileLabel(id: string): string {
  return PROFILE_LABELS[id] ?? id;
}
