"""LevelBuilder — готовит level_vacancies_data и vacancies_skills."""

import structlog
from tqdm import tqdm

from src import Err, LevelBuildError, Ok, Result, timed
from src.models.enums import ExperienceLevel
from src.models.vacancy import Vacancy

logger = structlog.get_logger("level_builder")


class LevelBuilder:
    @timed("LevelBuilder.build")
    def build(self, vacancies: list, parser) -> Result[tuple[list, list], LevelBuildError]:
        try:
            levels = []
            skills = []
            for vac in tqdm(vacancies, desc="Разбор вакансий по уровням"):
                if isinstance(vac, Vacancy):
                    vac_skills = []
                    if hasattr(vac, "key_skills") and vac.key_skills:
                        vac_skills = [s.name if hasattr(s, "name") else str(s) for s in vac.key_skills]
                    elif hasattr(vac, "extracted_skills") and vac.extracted_skills:
                        vac_skills = vac.extracted_skills

                    vac_experience = ExperienceLevel.MIDDLE
                    if hasattr(vac, "experience") and vac.experience:
                        exp_obj = vac.experience
                        if hasattr(exp_obj, "id"):
                            exp_id = exp_obj.id.lower()
                            if "less1" in exp_id or "junior" in exp_id or "no_experience" in exp_id:
                                vac_experience = ExperienceLevel.JUNIOR
                            elif "between1and3" in exp_id or "between3and6" in exp_id:
                                vac_experience = ExperienceLevel.MIDDLE
                            elif "between6and10" in exp_id or "morethan10" in exp_id:
                                vac_experience = ExperienceLevel.SENIOR
                        elif isinstance(exp_obj, str):
                            exp_lower = exp_obj.lower()
                            if "junior" in exp_lower or "нет опыта" in exp_lower or "стажер" in exp_lower:
                                vac_experience = ExperienceLevel.JUNIOR
                            elif "senior" in exp_lower or "более 6" in exp_lower:
                                vac_experience = ExperienceLevel.SENIOR
                            else:
                                vac_experience = ExperienceLevel.MIDDLE

                    if vac_experience == ExperienceLevel.MIDDLE:
                        name = vac.name.lower() if hasattr(vac, "name") else ""
                        if "junior" in name or "младший" in name or "стажер" in name or "intern" in name:
                            vac_experience = ExperienceLevel.JUNIOR
                        elif "senior" in name or "старший" in name or "ведущий" in name:
                            vac_experience = ExperienceLevel.SENIOR

                    if vac_skills:
                        levels.append(
                            {"skills": vac_skills, "description": vac.description or "", "experience": vac_experience}
                        )
                        skills.append(vac_skills)
                else:
                    vac_skills = [s["name"] for s in vac.get("key_skills", [])]
                    if not vac_skills:
                        vac_skills = vac.get("extracted_skills", [])
                    if vac_skills:
                        experience = ExperienceLevel.MIDDLE
                        exp_obj = vac.get("experience", {})
                        if isinstance(exp_obj, dict):
                            exp_id = exp_obj.get("id", "").lower()
                            if "less1" in exp_id or "junior" in exp_id or "no_experience" in exp_id:
                                experience = ExperienceLevel.JUNIOR
                            elif "between1and3" in exp_id or "between3and6" in exp_id:
                                experience = ExperienceLevel.MIDDLE
                            elif "between6and10" in exp_id or "morethan10" in exp_id:
                                experience = ExperienceLevel.SENIOR
                        elif isinstance(exp_obj, str):
                            exp_lower = exp_obj.lower()
                            if "junior" in exp_lower or "нет опыта" in exp_lower or "стажер" in exp_lower:
                                experience = ExperienceLevel.JUNIOR
                            elif "senior" in exp_lower or "более 6" in exp_lower:
                                experience = ExperienceLevel.SENIOR
                        if experience == ExperienceLevel.MIDDLE:
                            name = vac.get("name", "").lower()
                            if "junior" in name or "младший" in name or "стажер" in name or "intern" in name:
                                experience = ExperienceLevel.JUNIOR
                            elif "senior" in name or "старший" in name or "ведущий" in name:
                                experience = ExperienceLevel.SENIOR
                        levels.append(
                            {"skills": vac_skills, "description": vac.get("description", ""), "experience": experience}
                        )
                        skills.append(vac_skills)
            print(f"  Подготовлено {len(levels)} вакансий для анализа уровней")
            return Ok((levels, skills))
        except Exception as e:
            logger.exception("level_building_failed", error=str(e))
            return Err(LevelBuildError(message=f"Ошибка построения уровней: {e}"))
