"""Pydantic models for the ZUN (знать/уметь/навыки) router.

Request models were extracted from ``zun.py`` to shrink the router module. The
models are imported back into ``zun.py`` verbatim, so any external importer of
``ZUNIn`` & co. keeps working unchanged.

Response models (6.2.1) describe what each endpoint actually returns. They are
declared with ``extra="allow"`` and used together with
``response_model_exclude_unset=True``: fields present in the payload are passed
through untouched (including ones not listed here), and fields absent from the
payload are not invented as ``null``. That keeps the wire format byte-identical
while giving OpenAPI a real schema.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ZUNIn(BaseModel):
    ksa_type: str
    text: str


class ZUNPatch(BaseModel):
    text: str


class ScopeChange(BaseModel):
    discipline_name: str
    included: bool


class ScopePut(BaseModel):
    dir_code: str = "09.03.02"
    changes: list[ScopeChange] = []


class CompetencyIn(BaseModel):
    code: str
    name: str = ""
    description: str = ""


class SkillLinkIn(BaseModel):
    skill_name: str
    ksa_type: str = "skills"


# ---------- response models (6.2.1) ----------


class ZunDirectionItem(BaseModel):
    """Элемент GET /teacher/zun/directions."""

    model_config = ConfigDict(extra="allow")

    dir_code: str
    name: str | None = None
    profile: str | None = None
    disciplines_count: Any = None
    ksa_total: Any = None
    has_ksa: bool | None = None


class ZunMyDirectionItem(BaseModel):
    """Элемент GET /teacher/zun/my-directions (набор полей зависит от роли)."""

    model_config = ConfigDict(extra="allow")

    dir_code: str
    name: str | None = None
    profile: str | None = None


class ZunKsaCounts(BaseModel):
    model_config = ConfigDict(extra="allow")

    knowledge: Any = None
    abilities: Any = None
    skills: Any = None
    total: Any = None


class ZunStatsCoverage(BaseModel):
    model_config = ConfigDict(extra="allow")

    latest_analysis_date: str | None = None
    avg_coverage_ratio: float | None = None
    covered_disciplines: Any = None


class ZunStats(BaseModel):
    """GET /teacher/zun/stats."""

    model_config = ConfigDict(extra="allow")

    dir_code: str
    direction_names: list[str] = Field(default_factory=list)
    profiles: list[Any] = Field(default_factory=list)
    multiple_profiles: bool | None = None
    total_disciplines: Any = None
    total_competencies: Any = None
    total_indicators: Any = None
    ksa_counts: ZunKsaCounts
    competencies_with_ksa: Any = None
    disciplines_with_ksa: Any = None
    linked_skills: Any = None
    coverage: ZunStatsCoverage


class ZunDisciplineItem(BaseModel):
    """Элемент GET /teacher/zun/disciplines."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str | None = None
    semester: Any = None
    control_form: Any = None
    competencies_count: Any = None
    indicators_count: Any = None
    ksa_total: Any = None
    ksa_by_type: dict[str, Any] | None = None
    linked_skills: Any = None
    coverage_ratio: float | None = None
    weighted_coverage: float | None = None
    coverage_level: str | None = None


class ZunCoverageBlock(BaseModel):
    """Блок coverage в дереве дисциплины."""

    model_config = ConfigDict(extra="allow")

    total_skills: Any = None
    matched_skills: Any = None
    coverage_ratio: float | None = None
    weighted_coverage: float | None = None
    coverage_level: str | None = None
    analysis_date: str | None = None


class ZunDisciplineTree(BaseModel):
    """GET /teacher/zun/disciplines/{discipline_id} и lookup по имени."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str | None = None
    semester: Any = None
    control_form: Any = None
    direction_code: str | None = None
    coverage: ZunCoverageBlock | None = None
    competencies: list[dict[str, Any]] = Field(default_factory=list)


class ZunSearchItem(BaseModel):
    """Элемент GET /teacher/zun/search."""

    model_config = ConfigDict(extra="allow")

    ksa_id: str
    ksa_type: str | None = None
    original_text: str | None = None
    cleaned_text: str | None = None
    competence_id: str | None = None
    competence_code: str | None = None
    is_indicator: bool | None = None
    parent_code: str | None = None
    discipline_id: str | None = None
    discipline_name: str | None = None


class ZunSemanticMatch(BaseModel):
    """Элемент matches в GET /teacher/zun/search/semantic."""

    model_config = ConfigDict(extra="allow")

    ksa_id: str | None = None
    ksa_type: str | None = None
    original_text: str | None = None
    competence_id: str | None = None
    competence_code: str | None = None
    is_indicator: bool | None = None
    parent_code: str | None = None
    discipline_id: str | None = None
    discipline_name: str | None = None
    similarity: float | None = None


class ZunSemanticResponse(BaseModel):
    """GET /teacher/zun/search/semantic."""

    model_config = ConfigDict(extra="allow")

    mode: str
    query: str | None = None
    matches: list[ZunSemanticMatch] = Field(default_factory=list)
    note: str | None = None


class ZunFilterItem(BaseModel):
    """Элемент GET /teacher/zun/filter."""

    model_config = ConfigDict(extra="allow")

    competence_id: str
    code: str | None = None
    category: Any = None
    name: str | None = None
    development_level: Any = None
    discipline_id: str | None = None
    discipline_name: str | None = None
    ksa_total: Any = None


class ZunCompetencySkill(BaseModel):
    """Элемент GET /teacher/zun/competencies/{id}/skills."""

    model_config = ConfigDict(extra="allow")

    skill_id: str
    skill_name: str | None = None
    match_type: str | None = None
    source_text: str | None = None
    ksa_type: str | None = None


class ZunCompetencyCoverageDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_skills: Any = None
    matched_skills: Any = None
    coverage_ratio: float | None = None
    analysis_date: str | None = None


class ZunCompetencyCoverage(BaseModel):
    """GET /teacher/zun/competencies/{id}/coverage."""

    model_config = ConfigDict(extra="allow")

    competence_id: str
    competence_code: str | None = None
    discipline_id: str | None = None
    discipline_name: str | None = None
    coverage: ZunCompetencyCoverageDetail | None = None


class ZunCompetencyPickerItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    code: str | None = None
    discipline_name: str | None = None


class ZunCompetencyPicker(BaseModel):
    """GET /teacher/zun/competencies/search."""

    model_config = ConfigDict(extra="allow")

    items: list[ZunCompetencyPickerItem] = Field(default_factory=list)


class ZunScopeDiscipline(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    in_scope: bool | None = None
    source: str | None = None


class ZunScope(BaseModel):
    """GET /teacher/zun/scope."""

    model_config = ConfigDict(extra="allow")

    dir_code: str
    disciplines: list[ZunScopeDiscipline] = Field(default_factory=list)
    methodology_excluded: list[str] = Field(default_factory=list)


class ZunScopeUpdated(BaseModel):
    """PUT /teacher/zun/scope."""

    model_config = ConfigDict(extra="allow")

    dir_code: str
    updated: int


class ZunEntry(BaseModel):
    """POST/PATCH записи KSA."""

    model_config = ConfigDict(extra="allow")

    ksa_id: str
    ksa_type: str | None = None
    text: str | None = None


class ZunCompetencyCreated(BaseModel):
    """POST /teacher/zun/disciplines/{id}/competencies."""

    model_config = ConfigDict(extra="allow")

    competency_id: str
    code: str | None = None
    category: Any = None


class ZunSkillLinked(BaseModel):
    """POST /teacher/zun/competencies/{id}/skills."""

    model_config = ConfigDict(extra="allow")

    skill_id: str
    skill_name: str | None = None
    ksa_type: str | None = None
    match_type: str | None = None


class ZunDeleteAck(BaseModel):
    """DELETE /teacher/zun/entries/{ksa_id}."""

    model_config = ConfigDict(extra="allow")

    status: str


class ZunAnalyzeStarted(BaseModel):
    """POST /teacher/zun/analyze."""

    model_config = ConfigDict(extra="allow")

    status: str
    dir_code: str | None = None


class ZunAnalyzeStatus(BaseModel):
    """GET /teacher/zun/analyze/status/{run_id}."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    action: str | None = None
    status: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


class ZunImportResult(BaseModel):
    """POST /teacher/zun/import/{dir_code}."""

    model_config = ConfigDict(extra="allow")

    dir_code: str
    inserted: int | None = None
    skipped_dups: int | None = None
    skipped_gap: int | None = None
    disciplines_processed: int | None = None
