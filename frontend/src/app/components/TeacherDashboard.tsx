import { useState, useEffect } from "react";
import { api } from "../api";
import { AnalysisPanel } from "./AnalysisPanel";
import CompetencyTrendsPanel from "./CompetencyTrendsPanel";

const API = "/api/teacher";

type Direction = {
  id: string;
  code: string;
  name: string;
  profile: string;
};

type DirectionAnalysis = {
  direction: string;
  direction_name: string;
  profile: string;
  total_disciplines: number;
  average_coverage: number;
  coverage_level: string;
  total_gaps_across_all: number;
  top_cross_discipline_gaps: { skill: string; disciplines: number }[];
  top_emerging_across_all: { skill: string; frequency: number }[];
  recommendations: { type: string; priority: string; message: string }[];
  trends: { rising: any[]; declining: any[] };
  disciplines: { name: string; coverage_ratio: number; coverage_level: string; gaps: number; emerging: number }[];
  generated_at: string;
};

type Discipline = {
  id: string;
  name: string;
  competencies_count: number;
  skills_count: number;
};

type Competency = {
  id: string;
  code: string;
  name: string;
  skills: { id: string; name: string }[];
};

type DisciplineDetail = {
  id: string;
  name: string;
  competencies: Competency[];
};

type Recommendation = {
  id: string;
  discipline: string;
  competency: string;
  suggestion: string;
  type: string;
};

type Stats = {
  total_disciplines: number;
  total_competencies: number;
  total_skills: number;
};

export function TeacherDashboard() {
  const [disciplines, setDisciplines] = useState<Discipline[]>([]);
  const [selected, setSelected] = useState<DisciplineDetail | null>(null);
  const [expandedComp, setExpandedComp] = useState<string | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [suggestion, setSuggestion] = useState("");
  const [recType, setRecType] = useState("modify");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [directions, setDirections] = useState<Direction[]>([]);
  const [selectedDir, setSelectedDir] = useState("09.03.02");
  const [analysis, setAnalysis] = useState<DirectionAnalysis | null>(null);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<"coverage" | "trends">("coverage");
  const [showAddForm, setShowAddForm] = useState(false);
  const [runLoading, setRunLoading] = useState(false);

  useEffect(() => {
    Promise.all([
      api("/teacher/stats"),
      api("/teacher/krm/disciplines"),
      api("/teacher/krm/recommendations"),
      api("/teacher/krm/directions"),
    ]).then(([s, d, r, dirs]) => {
      setStats(s);
      setDisciplines(d as Discipline[]);
      setRecs(r as Recommendation[]);
      setDirections(Array.isArray(dirs) ? (dirs as Direction[]) : [dirs as Direction]);
      setLoading(false);
    }).catch((e) => {
      console.error("TeacherDashboard init failed", e);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!selectedDir) return;
    api(`/teacher/analysis?dir_code=${selectedDir}`)
      .then(setAnalysis)
      .catch(() => setAnalysis(null));
  }, [selectedDir]);

  async function loadDiscipline(name: string) {
    try {
      const data = await api(`/teacher/krm/disciplines/${encodeURIComponent(name)}`);
      setSelected(data as DisciplineDetail);
      setShowAnalysis(false);
    } catch {}
  }

  async function addRec() {
    if (!selected || !suggestion.trim()) return;
    try {
      const resp = await api("/teacher/krm/recommendations", {
        method: "POST",
        body: JSON.stringify({
          discipline_id: selected.name,
          competency_id: expandedComp || null,
          suggestion: suggestion.trim(),
          suggestion_type: recType,
        }),
      });
      setRecs((prev) => [
        ...prev,
        {
          id: resp.id,
          discipline_id: selected.name,
          competency_id: expandedComp || "",
          suggestion: suggestion.trim(),
          suggestion_type: recType,
        },
      ]);
      setSuggestion("");
    } catch {}
  }

  async function deleteRec(id: number) {
    try {
      await api(`/teacher/krm/recommendations/${id}`, { method: "DELETE" });
      setRecs((prev) => prev.filter((r) => r.id !== id));
    } catch {}
  }

  if (loading) {
    return (
      <div style={{ padding: "40px", fontFamily: "system-ui, sans-serif" }}>
        Loading...
      </div>
    );
  }

  const filtered = disciplines.filter((d) =>
    d.name.toLowerCase().includes(search.toLowerCase())
  );

  const containerStyle: React.CSSProperties = {
    display: "flex",
    height: "calc(100vh - 100px)",
    fontFamily: "system-ui, -apple-system, sans-serif",
    color: "#111827",
    background: "#fff",
  };

  const sidebarStyle: React.CSSProperties = {
    width: 380,
    minWidth: 380,
    borderRight: "1px solid #e5e7eb",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  };

  const mainStyle: React.CSSProperties = {
    flex: 1,
    overflow: "auto",
    padding: "24px",
  };

  const card: React.CSSProperties = {
    background: "#fff",
    borderRadius: 8,
    border: "1px solid #e5e7eb",
    padding: 14,
    marginBottom: 10,
    boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
  };

  function covColor(cov: number) {
    return cov > 0.5 ? "#059669" : cov > 0.2 ? "#d97706" : "#dc2626";
  }

  return (
    <div style={containerStyle}>
      <div style={sidebarStyle}>
          <div style={{ padding: "16px", borderBottom: "1px solid #e5e7eb" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h1 style={{ fontSize: 18, margin: 0, color: "#111827" }}>
                KRM Teacher
              </h1>
              {stats && (
                  <div style={{ fontSize: 11, color: "#6b7280", textAlign: "right" }}>
                    <div>{(stats as any).total_disciplines ?? (stats as any).total_reports ?? 0} disc</div>
                    <div>{(stats as any).total_competencies ?? Object.keys((stats as any).by_profession || {}).length} comp</div>
                    <div>{((stats as any).total_skills ?? 0).toLocaleString()} skills</div>
                  </div>
                )}
          </div>

          {/* Direction selector */}
          <select
            value={selectedDir}
            onChange={(e) => setSelectedDir(e.target.value)}
            style={{
              width: "100%",
              marginTop: 10,
              padding: "6px 8px",
              background: "#fff",
              border: "1px solid #e5e7eb",
              borderRadius: 6,
              color: "#111827",
              fontSize: 12,
              outline: "none",
            }}
          >
            {directions.map((d) => (
              <option key={d.code} value={d.code}>{d.code} - {d.name}</option>
            ))}
          </select>

          {/* Run analysis button */}
          <button
            onClick={async () => {
              setRunLoading(true);
              try {
                await api("/teacher/krm/run-analysis", { method: "POST" });
                // wait a bit then reload
                setTimeout(async () => {
                  try {
                    const a = await api(`/teacher/analysis?dir_code=${selectedDir}`);
                    setAnalysis(a);
                  } catch {}
                  setRunLoading(false);
                }, 3000);
              } catch {
                setRunLoading(false);
              }
            }}
            disabled={runLoading}
            style={{
              width: "100%",
              marginTop: 10,
              padding: "8px 12px",
              background: runLoading ? "#9ca3af" : "#7c3aed",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              cursor: runLoading ? "default" : "pointer",
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            {runLoading ? "Анализ запущен..." : "Запустить анализ"}
          </button>

          {/* Analysis summary button */}
          {analysis && (
            <div
              onClick={() => setShowAnalysis(!showAnalysis)}
              style={{
                marginTop: 10,
                padding: "8px 12px",
                background: "#f9fafb",
                border: `1px solid ${covColor(analysis.average_coverage)}`,
                borderRadius: 6,
                cursor: "pointer",
                fontSize: 12,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "#7c3aed", fontWeight: 600 }}>Analysis</span>
                <span style={{ color: covColor(analysis.average_coverage), fontWeight: 700 }}>
                  {(analysis.average_coverage * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{ color: "#6b7280", marginTop: 2 }}>
                {analysis.total_disciplines} disciplines, {analysis.total_gaps_across_all} gaps
              </div>
            </div>
          )}

          <input
            placeholder="Search disciplines..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              width: "100%",
              marginTop: 10,
              padding: "8px 12px",
              background: "#fff",
              border: "1px solid #e5e7eb",
              borderRadius: 6,
              color: "#111827",
              fontSize: 13,
              outline: "none",
              boxSizing: "border-box",
            }}
          />
        </div>
        <div style={{ flex: 1, overflow: "auto" }}>
          {filtered.map((d) => {
            const discAnalysis = analysis?.disciplines.find((a) => a.name === d.name);
            return (
              <div
                key={d.name}
                onClick={() => loadDiscipline(d.name)}
                style={{
                  padding: "10px 16px",
                  cursor: "pointer",
                  borderBottom: "1px solid #e5e7eb",
                  background:
                    selected?.name === d.name ? "#eef2ff" : "transparent",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#7c3aed" }}>
                    {d.name}
                  </div>
                  {discAnalysis && (
                    <span style={{
                      fontSize: 11,
                      fontWeight: 700,
                      color: covColor(discAnalysis.coverage_ratio),
                    }}>
                      {(discAnalysis.coverage_ratio * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
                <div style={{ fontSize: 11, color: "#6b7280", marginTop: 2 }}>
                  {d.competencies_count} comps / {d.skills_count} skills
                  {discAnalysis && ` / ${discAnalysis.gaps} gaps`}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div style={mainStyle}>
        {!selected && !showAnalysis && (
          <div style={{ textAlign: "center", marginTop: 80, color: "#9ca3af", fontSize: 14 }}>
            Select a discipline or open analysis
          </div>
        )}

        {/* Always visible mode toggle */}
        <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
          <button
            onClick={() => setAnalysisMode("coverage")}
            style={{
              padding: "6px 14px",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 12,
              fontWeight: analysisMode === "coverage" ? 700 : 400,
              background: analysisMode === "coverage" ? "#7c3aed" : "#f9fafb",
              color: analysisMode === "coverage" ? "#fff" : "#7c3aed",
            }}
          >
            Coverage
          </button>
          <button
            onClick={() => setAnalysisMode("trends")}
            style={{
              padding: "6px 14px",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 12,
              fontWeight: analysisMode === "trends" ? 700 : 400,
              background: analysisMode === "trends" ? "#7c3aed" : "#f9fafb",
              color: analysisMode === "trends" ? "#fff" : "#7c3aed",
            }}
          >
            Competency Trends
          </button>
        </div>

        {analysisMode === "coverage" && analysis && (<>
          <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
            <div style={card}>
              <div style={{ fontSize: 11, color: "#6b7280" }}>Average Coverage</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: covColor(analysis.average_coverage) }}>
                    {(analysis.average_coverage * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: 11, color: covColor(analysis.average_coverage), fontWeight: 600 }}>
                    {analysis.coverage_level.toUpperCase()}
                  </div>
                </div>
                <div style={card}>
                  <div style={{ fontSize: 11, color: "#6b7280" }}>Total Gaps</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: "#fca5a5" }}>
                    {analysis.total_gaps_across_all}
                  </div>
                </div>
                <div style={card}>
                  <div style={{ fontSize: 11, color: "#6b7280" }}>Disciplines</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: "#93c5fd" }}>
                    {analysis.total_disciplines}
                  </div>
                </div>
              </div>

              {/* Direction-level recommendations */}
              {analysis.recommendations.length > 0 && (
                <div style={card}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "#7c3aed", marginBottom: 8 }}>
                    Рекомендации
                  </div>
                  {analysis.recommendations.map((r, i) => (
                    <div key={i} style={{ padding: "8px 10px", marginBottom: 6, background: "#f9fafb", borderRadius: 6, borderLeft: `3px solid ${r.priority === "high" ? "#dc2626" : r.priority === "medium" ? "#d97706" : "#2563eb"}`, fontSize: 12 }}>
                      <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 4 }}>
                        <span style={{ display: "inline-block", padding: "2px 8px", borderRadius: 4, fontSize: 11, background: r.priority === "high" ? "#fee2e2" : r.priority === "medium" ? "#fef3c7" : "#dbeafe", color: r.priority === "high" ? "#dc2626" : r.priority === "medium" ? "#92400e" : "#1d4ed8", fontWeight: 600 }}>{r.priority === "high" ? "высокий" : r.priority === "medium" ? "средний" : "низкий"}</span>
                        <span style={{ fontSize: 11, color: "#6b7280" }}>{r.type}</span>
                      </div>
                      <div style={{ color: "#374151", lineHeight: 1.4 }}>{r.message}</div>
                    </div>
                  ))}
                </div>
              )}

                 <div style={card}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#7c3aed", marginBottom: 8 }}>Междисциплинарные разрывы</div>
                {analysis.top_cross_discipline_gaps.map((g, i) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid #e5e7eb", fontSize: 12 }}>
                    <span style={{ color: "#b91c1c", fontWeight: 500 }}>{g.skill}</span>
                    <span style={{ color: "#6b7280" }}>{g.disciplines} дисциплин</span>
                  </div>
                ))}
              </div>

              <div style={card}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#7c3aed", marginBottom: 8 }}>Востребованные навыки рынка</div>
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                  {analysis.top_emerging_across_all.map((s, i) => (
                    <span key={i} style={{ display: "inline-block", padding: "2px 8px", borderRadius: 4, fontSize: 11, background: "#e0e7ff", color: "#4338ca", margin: 2 }}>
                      {s.skill} <span style={{ opacity: 0.5 }}>×{s.frequency}</span>
                    </span>
                  ))}
                </div>
              </div>

              {analysis.trends && (
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                  {analysis.trends.rising?.length > 0 && (
                    <div style={{ ...card, flex: 1, minWidth: 200 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: "#059669", marginBottom: 8 }}>Растущие навыки</div>
                      {analysis.trends.rising.map((t, i) => (
                        <div key={i} style={{ fontSize: 11, padding: "2px 0", color: "#4b5563" }}>
                          {t.skill} <span style={{ color: "#059669" }}>+{t.change_pct}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {analysis.trends.declining?.length > 0 && (
                    <div style={{ ...card, flex: 1, minWidth: 200 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: "#dc2626", marginBottom: 8 }}>Падающие навыки</div>
                      {analysis.trends.declining.map((t, i) => (
                        <div key={i} style={{ fontSize: 11, padding: "2px 0", color: "#4b5563" }}>
                          {t.skill} <span style={{ color: "#dc2626" }}>{t.change_pct}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div style={card}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#7c3aed", marginBottom: 8 }}>Разбивка по дисциплинам</div>
                {analysis.disciplines.map((d, i) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid #e5e7eb", fontSize: 12, cursor: "pointer" }}
                    onClick={() => { const found = disciplines.find((dd) => dd.name === d.name); if (found) loadDiscipline(found.name); }}
                  >
                    <span style={{ color: "#374151", maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
                    <div style={{ display: "flex", gap: 12 }}>
                      <span style={{ color: covColor(d.coverage_ratio), fontWeight: 600 }}>{(d.coverage_ratio * 100).toFixed(1)}%</span>
                      <span style={{ color: "#dc2626" }}>{d.gaps}g</span>
                      <span style={{ color: "#2563eb" }}>{d.emerging}e</span>
                    </div>
                  </div>
                ))}
              </div>
            </>)}

          {analysisMode === "trends" && (
            <CompetencyTrendsPanel
              dirCode={selectedDir}
              competencyCodes={selected ? selected.competencies.map((c) => c.code) : undefined}
            />
          )}

        {/* Discipline detail */}
        {selected && (
          <>
            <h2 style={{ fontSize: 20, margin: "0 0 20px", color: "#111827" }}>
              {selected.name}
            </h2>

            {/* Analysis panel for this discipline */}
            <AnalysisPanel disciplineName={selected.name} />

            {selected.competencies.map((comp) => {
              const isOpen = expandedComp === comp.code;
              return (
                <div
                  key={comp.code}
                  style={{
                    marginBottom: 8,
                    border: "1px solid #d1d5db",
                    borderRadius: 8,
                    overflow: "hidden",
                  }}
                >
                  <div
                    onClick={() => setExpandedComp(isOpen ? null : comp.code)}
                    style={{
                      padding: "10px 16px",
                      background: "#f9fafb",
                      cursor: "pointer",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <span style={{ fontWeight: 600, color: "#7c3aed" }}>
                      {comp.code}
                    </span>
                    <span style={{ fontSize: 12, color: "#6b7280" }}>
                      {comp.skills.length} skills {isOpen ? "v" : ">"}
                    </span>
                  </div>
                  {isOpen && (
                    <div style={{ padding: "8px 16px 12px" }}>
                      {comp.skills.length === 0 && (
                        <div style={{ color: "#9ca3af", fontSize: 12 }}>
                          No skills extracted
                        </div>
                      )}
                      {comp.skills.map((sk, i) => (
                        <div
                          key={i}
                          className="py-1 text-xs leading-relaxed border-b border-gray-100"
                        >
                          {sk.name}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}

            <div className="mt-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="flex items-center justify-center w-8 h-8 bg-purple-600 rounded-lg">
                  <span className="text-white text-sm font-bold">!</span>
                </div>
                <h3 className="text-sm font-semibold text-gray-900">Recommendations</h3>
                <span className="text-xs text-gray-400">({recs.filter((r) => r.discipline_id === selected?.name).length})</span>
                <button
                  onClick={() => setShowAddForm(!showAddForm)}
                  className="ml-auto text-xs bg-purple-600 text-white px-3 py-1.5 rounded-lg hover:bg-purple-700 transition-colors cursor-pointer border-0"
                >
                  {showAddForm ? "Cancel" : "Add Recommendation"}
                </button>
              </div>
              {showAddForm && (
                <div className="mb-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <textarea
                    placeholder="Your recommendation for this competency..."
                    value={suggestion}
                    onChange={(e) => setSuggestion(e.target.value)}
                    rows={2}
                    className="w-full p-2 text-sm border border-gray-300 rounded-lg resize-vertical outline-none box-border"
                    style={{ background: "#fff", color: "#1f2937" }}
                  />
                  <div className="flex gap-2 mt-2 items-center">
                    <select
                      value={recType}
                      onChange={(e) => setRecType(e.target.value)}
                      className="h-9 px-2 text-sm bg-white border border-gray-300 rounded-lg outline-none text-gray-900"
                    >
                      <option value="modify">Modify</option>
                      <option value="add">Add</option>
                      <option value="remove">Remove</option>
                    </select>
                    <button
                      onClick={() => { addRec(); setShowAddForm(false); }}
                      className="h-9 px-4 text-sm bg-purple-600 text-white border-0 rounded-lg cursor-pointer hover:bg-purple-700 transition-colors"
                    >
                      Send
                    </button>
                  </div>
                </div>
              )}
              {recs.filter((r) => r.discipline_id === selected?.name).length === 0 ? (
                <div className="text-xs text-gray-400 bg-gray-50 rounded-lg p-4 text-center">
                  No recommendations for this discipline yet
                </div>
              ) : (
                recs
                  .filter((r) => r.discipline_id === selected?.name)
                  .map((r, i) => (
                    <div
                      key={i}
                      className="border border-gray-100 rounded-lg p-3 mb-2 text-sm"
                      style={{
                        borderLeft: "3px solid #7c3aed",
                      }}
                    >
                      <div className="text-gray-400 mb-1 text-xs">
                        [{r.suggestion_type}] {r.competency_id}
                      </div>
                      <div className="text-gray-900">{r.suggestion}</div>
                      <button
                        onClick={() => deleteRec(r.id)}
                        className="mt-2 text-xs text-red-500 border border-red-500 rounded px-2 py-0.5 hover:bg-red-50 transition-colors bg-transparent cursor-pointer"
                      >
                        Delete
                      </button>
                    </div>
                  ))
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
