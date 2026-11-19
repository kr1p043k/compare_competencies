import { useEffect, useMemo, useState } from "react";
import { User } from "lucide-react";
import { useAuth } from "../../../lib/auth";
import { request } from "../../../lib/api";
import { errMessage, reportError } from "../../../lib/errors";
import {
  Card,
  CardBody,
  CardHeader,
  ErrorState,
  PageHeader,
  Skeleton,
  StatCard,
} from "../../../design";
import { TwoFactorCard } from "../TwoFactorCard";
import { CUSTOM_NAME_RE, NO_BASE, norm, splitLines } from "./helpers";
import { MySkillsCard } from "./student/MySkillsCard";
import { MyTechnologiesCard } from "./student/MyTechnologiesCard";
import { MyProfilesCard } from "./student/MyProfilesCard";
import { MyCompetenciesCard } from "./student/MyCompetenciesCard";
import type {
  AccountProps,
  CreatedCustomProfile,
  CustomOptionCompetency,
  CustomOptions,
  DisciplineStat,
  KrmData,
  MissingSkill,
  NewCompetencyDraft,
  SelfData,
} from "./types";

/** Полная страница профиля студента: identity + KPI + свои навыки + drill-down по КРМ. */
export function StudentProfilePage({ displayName, email, onNavigate }: AccountProps & { onNavigate: (tab: string) => void }) {
  const auth = useAuth();
  const name = displayName ?? auth.name ?? undefined;
  const login = email ?? auth.username ?? undefined;

  const [self, setSelf] = useState<SelfData | null>(null);
  const [selfLoading, setSelfLoading] = useState(true);
  const [selfError, setSelfError] = useState<string | null>(null);
  const [krm, setKrm] = useState<KrmData | null>(null);
  const [krmLoading, setKrmLoading] = useState(true);
  const [krmError, setKrmError] = useState<string | null>(null);
  const [dirCode, setDirCode] = useState("09.03.02");
  const [openDisc, setOpenDisc] = useState<Record<string, boolean>>({});
  const [newSkill, setNewSkill] = useState("");
  const [saving, setSaving] = useState(false);
  const [opError, setOpError] = useState<string | null>(null);

  // --- My technologies: chips + suggest + optimistic PATCH add_technologies/remove_technologies ---
  const [techInput, setTechInput] = useState("");
  const [techSaving, setTechSaving] = useState(false);
  const [techError, setTechError] = useState<string | null>(null);
  // --- My profiles: GET custom/options + POST custom (404-safe, inline errors) ---
  const [customOptions, setCustomOptions] = useState<CustomOptions | null>(null);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [baseProfiles, setBaseProfiles] = useState<string[]>(["base"]);
  const [cpName, setCpName] = useState("");
  const [cpLevel, setCpLevel] = useState("middle");
  const [cpBase, setCpBase] = useState(NO_BASE);
  const [compSearch, setCompSearch] = useState("");
  const [compCodes, setCompCodes] = useState<string[]>([]);
  const [draftCode, setDraftCode] = useState("");
  const [draftTitle, setDraftTitle] = useState("");
  const [draftKnowledge, setDraftKnowledge] = useState("");
  const [draftAbilities, setDraftAbilities] = useState("");
  const [draftSkills, setDraftSkills] = useState("");
  const [newComps, setNewComps] = useState<NewCompetencyDraft[]>([]);
  const [cpTech, setCpTech] = useState("");
  const [creating, setCreating] = useState(false);
  const [triedCreate, setTriedCreate] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [createdProfiles, setCreatedProfiles] = useState<CreatedCustomProfile[]>([]);

  const techList = useMemo(() => self?.technologies ?? [], [self]);

  const techSuggestions = useMemo(() => {
    const pool = customOptions?.technologies_suggest ?? [];
    return pool.filter((t) => !techList.some((x) => norm(x) === norm(t))).slice(0, 10);
  }, [customOptions, techList]);

  const filteredCustomComps = useMemo(() => {
    const all = customOptions?.competencies ?? [];
    const q = norm(compSearch);
    if (!q) return all.slice(0, 200);
    return all.filter((c) => norm(c.code).includes(q) || norm(c.title ?? "").includes(q)).slice(0, 200);
  }, [customOptions, compSearch]);

  const cpNameNorm = cpName.trim().toLowerCase();
  const cpNameValid = CUSTOM_NAME_RE.test(cpNameNorm);
  const cpTechList = useMemo(
    () => cpTech.split(",").map((t) => t.trim()).filter((t) => t.length > 0),
    [cpTech]
  );
  const totalComps = compCodes.length + newComps.length;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const d = await request<{
          competencies?: CustomOptionCompetency[];
          technologies_suggest?: string[];
        }>("/api/profiles/custom/options");
        if (cancelled) return;
        setCustomOptions({
          competencies: Array.isArray(d.competencies) ? d.competencies : [],
          technologies_suggest: Array.isArray(d.technologies_suggest) ? d.technologies_suggest : [],
        });
      } catch (e) {
        if (!cancelled) setOptionsError("Не удалось загрузить опции своих профилей (" + (errMessage(e) || "Не удалось загрузить опции своих профилей") + ")");
      } finally {
        if (!cancelled) setOptionsLoading(false);
      }
      try {
        const d = await request<{ profiles?: string[] }>("/api/profiles");
        if (!cancelled && Array.isArray(d?.profiles) && d.profiles.length > 0) {
          setBaseProfiles(d.profiles);
        }
      } catch (err) {
        reportError("ProfilePages:baseProfiles", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const patchTechnologies = async (body: { add_technologies?: string[]; remove_technologies?: string[] }): Promise<string[] | null> => {
    const updated = await request<{ technologies?: string[] }>("/api/profiles/self", {
      method: "PATCH",
      body,
    });
    return Array.isArray(updated.technologies) ? updated.technologies : null;
  };

  const handleAddTech = async (rawTech?: string) => {
    const text = (rawTech ?? techInput).trim();
    if (!text || techSaving || !self) return;
    if (techList.some((t) => norm(t) === norm(text))) {
      if (!rawTech) setTechInput("");
      return;
    }
    const backup = techList;
    setTechSaving(true);
    setTechError(null);
    setSelf({ ...self, technologies: [...backup, text] });
    if (!rawTech) setTechInput("");
    try {
      const server = await patchTechnologies({ add_technologies: [text] });
      if (server !== null) setSelf((prev) => (prev ? { ...prev, technologies: server } : prev));
    } catch (e) {
      setSelf((prev) => (prev ? { ...prev, technologies: backup } : prev));
      if (!rawTech) setTechInput(text);
      setTechError(errMessage(e) || "Не удалось добавить технологию");
    } finally {
      setTechSaving(false);
    }
  };

  const handleRemoveTech = async (text: string) => {
    if (techSaving || !self) return;
    const backup = techList;
    setTechSaving(true);
    setTechError(null);
    setSelf({ ...self, technologies: backup.filter((t) => norm(t) !== norm(text)) });
    try {
      const server = await patchTechnologies({ remove_technologies: [text] });
      if (server !== null) setSelf((prev) => (prev ? { ...prev, technologies: server } : prev));
    } catch (e) {
      setSelf((prev) => (prev ? { ...prev, technologies: backup } : prev));
      setTechError(errMessage(e) || "Не удалось убрать технологию");
    } finally {
      setTechSaving(false);
    }
  };

  const toggleCompCode = (code: string) => {
    setCompCodes((prev) => (prev.includes(code) ? prev.filter((c) => c !== code) : [...prev, code]));
  };

  const addDraft = () => {
    const code = draftCode.trim();
    if (!code) {
      setCreateError("Укажите код новой компетенции");
      return;
    }
    if (compCodes.includes(code) || newComps.some((d) => d.code === code)) {
      setCreateError(`Компетенция ${code} уже добавлена`);
      return;
    }
    setCreateError(null);
    setNewComps((prev) => [...prev, { code: code, title: draftTitle.trim(), knowledge: draftKnowledge, abilities: draftAbilities, skills: draftSkills }]);
    setDraftCode("");
    setDraftTitle("");
    setDraftKnowledge("");
    setDraftAbilities("");
    setDraftSkills("");
  };

  const removeDraft = (code: string) => {
    setNewComps((prev) => prev.filter((d) => d.code !== code));
  };

  const handleCreateProfile = async () => {
    setTriedCreate(true);
    if (!cpNameValid) {
      setCreateError("Название: латиница, цифры и подчёркивание, 2-32 символа");
      return;
    }
    if (totalComps < 1) {
      setCreateError("Выберите хотя бы одну компетенцию или добавьте новую");
      return;
    }
    setCreating(true);
    setCreateError(null);
    const payload = {
      name: cpNameNorm,
      target_level: cpLevel,
      ...(cpBase !== NO_BASE ? { base: cpBase } : {}),
      competency_codes: compCodes,
      new_competencies: newComps.map((d) => ({
        code: d.code,
        ...(d.title ? { title: d.title } : {}),
        knowledge: splitLines(d.knowledge),
        abilities: splitLines(d.abilities),
        skills: splitLines(d.skills),
      })),
      technologies: cpTechList,
    };
    try {
      const d = await request<{
        profile?: string;
        name?: string;
        target_level?: string;
        competencies_count?: number;
        new_competencies_count?: number;
        technologies_count?: number;
      }>("/api/profiles/custom", { method: "POST", body: payload });
      setCreatedProfiles((prev) => [
        ...prev,
        {
          name: d.profile ?? d.name ?? cpNameNorm,
          target_level: d.target_level ?? cpLevel,
          competencies: d.competencies_count ?? compCodes.length,
          created_new: d.new_competencies_count ?? newComps.length,
          technologies: d.technologies_count ?? cpTechList.length,
        },
      ]);
      setCpName("");
      setCpLevel("middle");
      setCpBase(NO_BASE);
      setCompSearch("");
      setCompCodes([]);
      setNewComps([]);
      setCpTech("");
      setTriedCreate(false);
    } catch (e) {
      setCreateError(errMessage(e) || "Не удалось создать профиль");
    } finally {
      setCreating(false);
    }
  };

  const loadKrm = async (profileName: string, dir: string) => {
    setKrmLoading(true);
    setKrmError(null);
    try {
      const d = await request<KrmData>(
        `/api/krm/student/competencies?direction=${encodeURIComponent(dir)}&profile=${encodeURIComponent(profileName)}`,
      );
      setKrm(d);
    } catch (e) {
      setKrmError(errMessage(e) || "Ошибка загрузки компетенций");
      setKrm(null);
    } finally {
      setKrmLoading(false);
    }
  };

  const loadAll = async () => {
    setSelfLoading(true);
    setSelfError(null);
    setKrmLoading(true);
    setKrmError(null);
    try {
      const d = await request<SelfData>("/api/profiles/self");
      setSelf(d);
      const profileName = d.profile || "base";
      let dir = "09.03.02";
      try {
        const dd = await request<{ directions?: { dir_code?: string }[] }>("/api/krm/directions");
        const first = dd?.directions?.[0]?.dir_code;
        if (typeof first === "string" && first) dir = first;
      } catch (err) {
        reportError("ProfilePages:krmDirections", err);
      }
      setDirCode(dir);
      await loadKrm(profileName, dir);
    } catch (e) {
      setSelfError(errMessage(e) || "Ошибка загрузки профиля");
      setKrmLoading(false);
    } finally {
      setSelfLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (cancelled) return;
      await loadAll();
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const patchSelf = async (body: Record<string, unknown>): Promise<SelfData> => {
    return request<SelfData>("/api/profiles/self", { method: "PATCH", body });
  };

  const handleAddSkill = async (raw?: string) => {
    const text = (raw ?? newSkill).trim();
    if (!text || saving || !self) return;
    const backup = self;
    setSaving(true);
    setOpError(null);
    const already = self.skills.some((s) => norm(s) === norm(text));
    if (!already) {
      setSelf({ ...self, skills: [...self.skills, text], user_added: [...(self.user_added || []), text] });
    }
    if (!raw) setNewSkill("");
    try {
      const updated = await patchSelf({ add_skills: [text] });
      setSelf(updated);
      await loadKrm(updated.profile || backup.profile || "base", dirCode);
    } catch (e) {
      setSelf(backup);
      if (!raw) setNewSkill(text);
      setOpError(errMessage(e) || "Не удалось добавить навык");
    } finally {
      setSaving(false);
    }
  };

  const handleRemoveSkill = async (text: string) => {
    if (saving || !self) return;
    const backup = self;
    setSaving(true);
    setOpError(null);
    setSelf({
      ...self,
      skills: self.skills.filter((s) => norm(s) !== norm(text)),
      user_added: (self.user_added || []).filter((s) => norm(s) !== norm(text)),
    });
    try {
      const updated = await patchSelf({ remove_skills: [text] });
      setSelf(updated);
      await loadKrm(updated.profile || backup.profile || "base", dirCode);
    } catch (e) {
      setSelf(backup);
      setOpError(errMessage(e) || "Не удалось удалить навык");
    } finally {
      setSaving(false);
    }
  };

  const discStats = useMemo<DisciplineStat[]>(() => {
    if (!krm) return [];
    return krm.disciplines.map((d) => {
      const total = d.competencies.reduce((a, c) => a + c.skills.length, 0);
      const has = d.competencies.reduce((a, c) => a + c.skills.filter((s) => s.student_has).length, 0);
      return { name: d.discipline, total, has, missing: total - has };
    });
  }, [krm]);

  const topGaps = useMemo(
    () => [...discStats].filter((d) => d.missing > 0).sort((a, b) => b.missing - a.missing).slice(0, 5),
    [discStats]
  );

  const missingPool = useMemo<MissingSkill[]>(() => {
    if (!krm) return [];
    const out: MissingSkill[] = [];
    for (const d of krm.disciplines) {
      for (const c of d.competencies) {
        for (const s of c.skills) {
          if (!s.student_has) {
            out.push({ text: s.text, code: c.code, discipline: d.discipline });
            if (out.length >= 8) return out;
          }
        }
      }
    }
    return out;
  }, [krm]);

  const skillsCount = self?.skills.length ?? 0;
  const krmTotal = krm?.counts.krm ?? 0;
  const overlap = krm?.counts.overlap ?? 0;
  const discsCount = krm?.disciplines.length ?? 0;
  const owned = new Set((self?.user_added || []).map((s) => norm(s)));
  const loading = selfLoading || krmLoading;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Мой профиль"
        description="Учётная запись, навыки и свои профили для сравнения с рынком."
      />
      {/* Identity header */}
      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <User className="size-5 text-success" />
              {name || "Мой профиль"}
            </span>
          }
          description={
            login || self ? (
              <>
                {login || "-"}
                {self ? ` · уровень ${self.target_level} · профиль ${self.profile}` : ""}
              </>
            ) : undefined
          }
        />
        <CardBody>
          {selfLoading && (
            <div className="flex items-center gap-3" aria-label="Загрузка профиля">
              <Skeleton className="size-10 rounded-full" />
              <div className="space-y-2 flex-1">
                <Skeleton className="h-3 w-40" />
                <Skeleton className="h-3 w-24" />
              </div>
            </div>
          )}
          {selfError && <ErrorState description={selfError} onRetry={() => void loadAll()} />}
        </CardBody>
      </Card>

      <TwoFactorCard />

      {/* KPI cards row */}
      {loading ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2" aria-label="Загрузка показателей">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="rounded-lg border border-border bg-card p-3">
              <Skeleton className="h-6 w-12 mx-auto" />
              <Skeleton className="mt-2 h-3 w-20 mx-auto" />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
          {[
            { label: "Навыки", value: self ? String(skillsCount) : "-" },
            { label: "Компетенции КРМ", value: krm ? String(krmTotal) : "-" },
            { label: "Совпало", value: krm ? String(overlap) : "-" },
            { label: "Дисциплин", value: krm ? String(discsCount) : "-" },
          ].map((k) => (
            <StatCard key={k.label} className="p-3" label={k.label} value={k.value} />
          ))}
        </div>
      )}

      {/* Мои навыки */}
      <MySkillsCard
        self={self}
        selfLoading={selfLoading}
        selfError={selfError}
        skillsCount={skillsCount}
        owned={owned}
        saving={saving}
        newSkill={newSkill}
        onNewSkillChange={setNewSkill}
        onAddSkill={() => void handleAddSkill()}
        onRemoveSkill={(s) => void handleRemoveSkill(s)}
        opError={opError}
      />

      {/* Мои компетенции - drill-down */}

      <MyTechnologiesCard
        self={self}
        selfLoading={selfLoading}
        techList={techList}
        techSaving={techSaving}
        techInput={techInput}
        onTechInputChange={setTechInput}
        onAddTech={(raw) => void handleAddTech(raw)}
        onRemoveTech={(t) => void handleRemoveTech(t)}
        techSuggestions={techSuggestions}
        suggestPool={customOptions?.technologies_suggest ?? []}
        techError={techError}
      />

      <MyProfilesCard
        optionsLoading={optionsLoading}
        optionsError={optionsError}
        createdProfiles={createdProfiles}
        cpName={cpName}
        onCpNameChange={setCpName}
        cpNameValid={cpNameValid}
        cpLevel={cpLevel}
        onCpLevelChange={setCpLevel}
        cpBase={cpBase}
        onCpBaseChange={setCpBase}
        baseProfiles={baseProfiles}
        compCodes={compCodes}
        compSearch={compSearch}
        onCompSearchChange={setCompSearch}
        filteredCustomComps={filteredCustomComps}
        onToggleCompCode={toggleCompCode}
        draftCode={draftCode}
        onDraftCodeChange={setDraftCode}
        draftTitle={draftTitle}
        onDraftTitleChange={setDraftTitle}
        draftKnowledge={draftKnowledge}
        onDraftKnowledgeChange={setDraftKnowledge}
        draftAbilities={draftAbilities}
        onDraftAbilitiesChange={setDraftAbilities}
        draftSkills={draftSkills}
        onDraftSkillsChange={setDraftSkills}
        onAddDraft={addDraft}
        onRemoveDraft={removeDraft}
        newComps={newComps}
        cpTech={cpTech}
        onCpTechChange={setCpTech}
        triedCreate={triedCreate}
        totalComps={totalComps}
        createError={createError}
        creating={creating}
        onCreateProfile={() => void handleCreateProfile()}
      />

      <MyCompetenciesCard
        krmLoading={krmLoading}
        krmError={krmError}
        onRetry={() => void loadKrm(self?.profile || "base", dirCode)}
        krm={krm}
        dirCode={dirCode}
        topGaps={topGaps}
        missingPool={missingPool}
        discStats={discStats}
        openDisc={openDisc}
        onToggleDisc={(discipline) => setOpenDisc((p) => ({ ...p, [discipline]: !p[discipline] }))}
        saving={saving}
        onAddSkill={(text) => void handleAddSkill(text)}
        onNavigate={onNavigate}
      />
    </div>
  );
}
