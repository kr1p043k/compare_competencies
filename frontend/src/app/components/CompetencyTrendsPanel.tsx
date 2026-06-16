import { useState, useEffect } from "react";
import { api } from "../api";

type SkillTrend = {
  name: string;
  change_pct: number;
  frequency: number;
  prev_frequency: number;
  history?: { date: string; freq: number }[];
};

type CompetencyTrend = {
  code: string;
  name: string;
  skill_count: number;
  active_skills_count: number;
  change_pct: number;
  direction: string;
  skills: SkillTrend[];
};

type TrendsResponse = {
  total: number;
  trends: CompetencyTrend[];
};

type Props = {
  dirCode: string;
  competencyCodes?: string[];
};

function trendArrow(pct: number): string {
  if (pct > 0) return "↑";
  if (pct < 0) return "↓";
  return "→";
}

function trendColor(pct: number): string {
  if (pct > 5) return "#059669";
  if (pct < -5) return "#dc2626";
  return "#d97706";
}

export default function CompetencyTrendsPanel({ dirCode, competencyCodes }: Props) {
  const [data, setData] = useState<TrendsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ limit: "200" });
    if (filter && filter !== "all") params.set("direction", filter);
    api(`/competency-trends?${params}`)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [dirCode, filter]);

  const card: React.CSSProperties = {
    background: "#0f0f23",
    borderRadius: 8,
    border: "1px solid #2d2d4a",
    padding: 14,
    marginBottom: 10,
  };

  if (loading) {
    return (
      <div style={{ color: "#555", fontSize: 13, padding: 12 }}>
        Loading competency trends...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ color: "#fca5a5", fontSize: 13, padding: 12 }}>
        {error}
      </div>
    );
  }

  if (!data || data.trends.length === 0) {
    return (
      <div style={{ color: "#555", fontSize: 13, padding: 12 }}>
        No competency trends data. Run teacher analysis first.
      </div>
    );
  }

  const displayed = competencyCodes
    ? data.trends.filter((c) => competencyCodes.includes(c.code))
    : data.trends;
  const rising = displayed.filter((c) => c.direction === "rising").length;
  const stable = displayed.filter((c) => c.direction === "stable").length;
  const falling = displayed.filter((c) => c.direction === "falling").length;
  const avgTrend = displayed.length
    ? displayed.reduce((s, c) => s + c.change_pct, 0) / displayed.length
    : 0;

  const filtered = data.trends
    .filter((c) => !competencyCodes || competencyCodes.includes(c.code))
    .filter((c) => !filter || filter === "all" || c.direction === filter);

  return (
    <div>
      {/* Summary bar */}
      <div
        style={{
          display: "flex",
          gap: 12,
          marginBottom: 16,
          flexWrap: "wrap",
        }}
      >
        <div style={card}>
          <div style={{ fontSize: 11, color: "#666" }}>Total Competencies</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: "#c4b5fd" }}>
            {data.total}
          </div>
        </div>
        <div style={card}>
          <div style={{ fontSize: 11, color: "#666" }}>Rising</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: "#6ee7b7" }}>
            {rising}
          </div>
        </div>
        <div style={card}>
          <div style={{ fontSize: 11, color: "#666" }}>Stable</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: "#fbbf24" }}>
            {stable}
          </div>
        </div>
        <div style={card}>
          <div style={{ fontSize: 11, color: "#666" }}>Falling</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: "#fca5a5" }}>
            {falling}
          </div>
        </div>
        <div style={card}>
          <div style={{ fontSize: 11, color: "#666" }}>Avg Trend</div>
          <div
            style={{
              fontSize: 20,
              fontWeight: 700,
              color: trendColor(avgTrend),
            }}
          >
            {avgTrend > 0 ? "+" : ""}
            {avgTrend.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Filter */}
      <div
        style={{
          display: "flex",
          gap: 6,
          marginBottom: 12,
          flexWrap: "wrap",
        }}
      >
        {[
          { key: null as string | null, label: `All (${data.total})` },
          { key: "rising", label: `Rising (${rising})` },
          { key: "stable", label: `Stable (${stable})` },
          { key: "falling", label: `Falling (${falling})` },
        ].map((f) => (
          <button
            key={f.key || "all"}
            onClick={() => setFilter(f.key)}
            style={{
              padding: "4px 12px",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 12,
              fontWeight: filter === f.key ? 700 : 400,
              background: filter === f.key ? "#7c3aed" : "#16213e",
              color: filter === f.key ? "#fff" : "#a78bfa",
            }}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Competency rows */}
      {filtered.map((comp) => {
        const isOpen = expanded === comp.code;
        const clsColor =
          comp.direction === "rising"
            ? "#6ee7b7"
            : comp.direction === "falling"
              ? "#fca5a5"
              : "#fbbf24";
        return (
          <div
            key={comp.code}
            style={{
              marginBottom: 8,
              border: "1px solid #2d2d4a",
              borderRadius: 8,
              overflow: "hidden",
            }}
          >
            <div
              onClick={() => setExpanded(isOpen ? null : comp.code)}
              style={{
                padding: "10px 16px",
                background: "#16213e",
                cursor: "pointer",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div
                style={{
                  display: "flex",
                  gap: 10,
                  alignItems: "center",
                }}
              >
                <span
                  style={{
                    fontWeight: 600,
                    color: "#a78bfa",
                    fontSize: 13,
                  }}
                >
                  {comp.code}
                </span>
                <span style={{ fontSize: 11, color: "#666" }}>
                  {comp.skill_count} skill{comp.skill_count !== 1 ? "s" : ""}
                  {comp.active_skills_count !== undefined &&
                    comp.active_skills_count !== comp.skill_count && (
                      <span
                        style={{
                          fontSize: 10,
                          color: "#888",
                          marginLeft: 4,
                        }}
                      >
                        ({comp.active_skills_count} tracked)
                      </span>
                    )}
                </span>
              </div>
              <div
                style={{
                  display: "flex",
                  gap: 8,
                  alignItems: "center",
                }}
              >
                <span
                  style={{
                    fontWeight: 700,
                    fontSize: 14,
                    color: clsColor,
                  }}
                >
                  {trendArrow(comp.change_pct)}{" "}
                  {comp.change_pct > 0 ? "+" : ""}
                  {comp.change_pct}%
                </span>
                <span
                  style={{
                    display: "inline-block",
                    padding: "2px 8px",
                    borderRadius: 4,
                    fontSize: 11,
                    background:
                      comp.direction === "rising"
                        ? "#065f46"
                        : comp.direction === "falling"
                          ? "#7f1d1d"
                          : "#713f12",
                    color: clsColor,
                    fontWeight: 600,
                  }}
                >
                  {comp.direction}
                </span>
                <span style={{ fontSize: 12, color: "#555" }}>
                  {isOpen ? "v" : ">"}
                </span>
              </div>
            </div>

            {/* Expanded per-skill trends */}
            {isOpen && (
              <div style={{ padding: "8px 16px 12px" }}>
                {comp.skills.length === 0 && (
                  <div style={{ color: "#555", fontSize: 12 }}>
                    No skills extracted
                  </div>
                )}
                {comp.skills.map((sk, i) => (
                  <div key={i}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        padding: "4px 0",
                        fontSize: 12,
                        borderBottom: "1px solid #1f1f3a",
                      }}
                    >
                      <span style={{ color: "#ccc" }}>{sk.name}</span>
                      <span
                        style={{
                          color: trendColor(sk.change_pct),
                          fontWeight: 600,
                        }}
                      >
                        {trendArrow(sk.change_pct)}{" "}
                        {sk.change_pct > 0 ? "+" : ""}
                        {sk.change_pct}%
                        <span
                          style={{
                            color: "#555",
                            fontWeight: 400,
                            marginLeft: 4,
                          }}
                        >
                          (freq: {sk.frequency})
                        </span>
                      </span>
                    </div>
                    {/* History chart */}
                    {sk.history && sk.history.length > 1 && (
                      <div
                        style={{
                          padding: "8px 0 12px",
                          fontSize: 11,
                          color: "#666",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "flex-end",
                            gap: 4,
                            height: 40,
                            padding: "4px 0",
                          }}
                        >
                          {sk.history.map((h, hi) => {
                            const maxFreq = Math.max(
                              ...sk.history!.map((x) => x.freq)
                            );
                            const pct =
                              maxFreq > 0 ? (h.freq / maxFreq) * 100 : 0;
                            return (
                              <div
                                key={hi}
                                title={`${h.date}: ${h.freq}`}
                                style={{
                                  flex: 1,
                                  height: `${Math.max(pct, 2)}%`,
                                  background: "#7c3aed",
                                  borderRadius: "2px 2px 0 0",
                                  minWidth: 12,
                                  opacity: 0.7 + 0.3 * (pct / 100),
                                }}
                              />
                            );
                          })}
                        </div>
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            marginTop: 2,
                          }}
                        >
                          {sk.history.map((h, hi) => (
                            <span key={hi} style={{ fontSize: 9 }}>
                              {h.date.slice(5)}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
