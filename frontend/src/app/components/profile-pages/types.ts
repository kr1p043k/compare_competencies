export type AccountProps = { displayName?: string; email?: string };

export type DisciplineRow = {
  name: string;
  competencies_count?: number;
  skills_count?: number;
  semester?: number | null;
  in_scope?: boolean;
};

export interface SelfData {
  profile: string;
  target_level: string;
  skills: string[];
  user_added: string[];
  competencies?: string[];
  technologies?: string[];
}

export interface CustomOptionCompetency {
  code: string;
  title?: string;
  skills_count?: number;
}

export interface CustomOptions {
  competencies: CustomOptionCompetency[];
  technologies_suggest: string[];
}

export interface NewCompetencyDraft {
  code: string;
  title: string;
  knowledge: string;
  abilities: string;
  skills: string;
}

export interface CreatedCustomProfile {
  name: string;
  target_level: string;
  competencies: number;
  created_new: number;
  technologies: number;
}

export interface KrmSkill {
  text: string;
  student_has: boolean;
}

export interface KrmCompetency {
  code: string;
  skills: KrmSkill[];
}

export interface KrmDiscipline {
  discipline: string;
  competencies: KrmCompetency[];
}

export interface KrmData {
  direction: string;
  direction_name: string;
  profile: string;
  disciplines: KrmDiscipline[];
  counts: { krm: number; student: number; overlap: number; merged: number; student_only: number };
}

export type DisciplineStat = { name: string; total: number; has: number; missing: number };

export type MissingSkill = { text: string; code: string; discipline: string };
