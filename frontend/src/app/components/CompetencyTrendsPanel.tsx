import { useState, useEffect } from "react";
import { api } from "../api";

type SkillTrend = {
  name: string;
  change_pct: number;
  frequency: number;
  prev_frequency: number;
};

type CompetencyTrend = {
  code: string;
  name: string;
  skills_count: number;
  aggregate_trend_pct: number;
  classification: string;
  skills: SkillTrend[];
};

type TrendsData = {
  direction_code: string;
  direction_name: string;
  snapshot_latest: string;
  snapshot_previous: string;
  competencies: CompetencyTrend[];
  summary: {
    total_competencies: number;
    rising: number;
    falling: number;
    stable: number;
    average_trend_pct: number;
  };
};

type Props = {
  dirCode: string;
};

function trendArrow(pct: number): string {
  if (pct > 0) return "↑";
  if (pct < 0) return "↓";
  return "→";
}

function trendColor(pct: number): string {
  if (pct > 5) return "#6ee7b7";
  if (pct < -5) return "#fca5a5";
  return "#fbbf24";
}

export default function CompetencyTrendsPanel({ dirCode }: Props) {
  const [data, setData] = useState<TrendsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("all");
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api(`/competency-trends?dir_code=${dirCode}&classification=${filter}`)
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

  if (!data || data.competencies.length === 0) {
    return (
      <div style={{ color: "#555", fontSize: 13, padding: 12 }}>
        No competency trends data. Run teacher analysis first.
      </div>
    );
  }

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
          <div
            style={{ fontSize: 20, fontWeight: 700, color: "#c4b5fd" }}
          >
            {data.summary.total_competencies}
          </div>
        </div>
        <div style={card}>
          <div style={{ fontSize: 11, color: "#666" }}>Rising</div>
          <div
            style={{ fontSize: 20, fontWeight: 700, color: "#6ee7b7" }}
          >
            {data.summary.rising}
          </div>
        </div>
        <div style={card}>
          <div style={{ fontSize: 11, color: "#666" }}>Stable</div>
          <div
            style={{ fontSize: 20, fontWeight: 700, color: "#fbbf24" }}
          >
            {data.summary.stable}
          </div>
        </div>
        <div style={card}>
          <div style={{ fontSize: 11, color: "#666" }}>Falling</div>
          <div
            style={{ fontSize: 20, fontWeight: 700, color: "#fca5a5" }}
          >
            {data.summary.falling}
          </div>
        </div>
        <div style={card}>
          <div style={{ fontSize: 11, color: "#666" }}>
            Avg Trend
          </div>
          <div
            style={{
              fontSize: 20,
              fontWeight: 700,
              color: trendColor(data.summary.average_trend_pct),
            }}
          >
            {data.summary.average_trend_pct > 0 ? "+" : ""}
            {data.summary.average_trend_pct}%
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
        {["all", "rising", "stable", "falling"].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              padding: "4px 12px",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 12,
              fontWeight: filter === f ? 700 : 400,
              background:
                filter === f ? "#7c3aed" : "#16213e",
              color: filter === f ? "#fff" : "#a78bfa",
            }}
          >
            {f === "all"
              ? `All (${data.summary.total_competencies})`
              : f === "rising"
              ? `Rising (${data.summary.rising})`
              : f === "stable"
              ? `Stable (${data.summary.stable})`
              : `Falling (${data.summary.falling})`}
          </button>
        ))}
      </div>

      {/* Competency rows */}
      {data.competencies.map((comp) => {
        const isOpen = expanded === comp.code;
        const clsColor =
          comp.classification === "rising"
            ? "#6ee7b7"
            : comp.classification === "falling"
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
            {/* Header row */}
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
                  {comp.skills_count} skill{comp.skills_count !== 1 ? "s" : ""}
                  {comp.active_skills_count !== undefined && comp.active_skills_count !== comp.skills_count && (
                    <span style={{ fontSize: 10, color: "#888", marginLeft: 4 }}>
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
                  {trendArrow(comp.aggregate_trend_pct)}{" "}
                  {comp.aggregate_trend_pct > 0 ? "+" : ""}
                  {comp.aggregate_trend_pct}%
                </span>
                <span
                  style={{
                    display: "inline-block",
                    padding: "2px 8px",
                    borderRadius: 4,
                    fontSize: 11,
                    background:
                      comp.classification === "rising"
                        ? "#065f46"
                        : comp.classification === "falling"
                        ? "#7f1d1d"
                        : "#713f12",
                    color: clsColor,
                    fontWeight: 600,
                  }}
                >
                  {comp.classification}
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
                  <div
                    key={i}
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
                ))}
              </div>
            )}
          </div>
        );
      })}

      {/* Timestamps */}
      <div
        style={{
          fontSize: 11,
          color: "#555",
          marginTop: 16,
          textAlign: "right",
        }}
      >
        Snapshots: {data.snapshot_previous} → {data.snapshot_latest}
      </div>
    </div>
  );
}
