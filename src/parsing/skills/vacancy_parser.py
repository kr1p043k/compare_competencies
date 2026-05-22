"""РџР°СЂСЃРµСЂ РІР°РєР°РЅСЃРёР№ (С„Р°СЃР°Рґ) СЃ РіРёР±СЂРёРґРЅС‹РјРё РІРµСЃР°РјРё Рё РєСЌС€РёСЂРѕРІР°РЅРёРµРј."""

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import structlog

from src import config
from src.models.data_contracts import SkillExtractionResult
from src.models.vacancy import Vacancy
from src.parsing.skills.bm25_ranker import BM25Ranker
from src.parsing.skills.hybrid_weight_calculator import HybridWeightCalculator
from src.parsing.skills.skill_embedding_cache import SkillEmbeddingCache
from src.parsing.skills.skill_normalizer import SkillNormalizer
from src.parsing.skills.skill_parser import SkillParser, SkillSource
from src.parsing.skills.skill_validator import SkillValidator
from src.parsing.utils import filter_skills_by_whitelist, load_it_skills
from src.utils import atomic_write_json

logger = structlog.get_logger(__name__)


class VacancyParser:
    def __init__(self):
        self.skill_parser = SkillParser()
        self.skill_validator = SkillValidator(whitelist=None)
        self.bm25_ranker = BM25Ranker()
        self.embedding_cache = SkillEmbeddingCache()
        self.hybrid_calc = HybridWeightCalculator(self.bm25_ranker, self.embedding_cache)

    # --------------------- РїСѓР±Р»РёС‡РЅС‹Рµ РјРµС‚РѕРґС‹ -------------------------
    def extract_skills_from_description(self, description: str) -> list[str]:
        if not description:
            return []
        extracted = self.skill_parser._extract_from_text(description, source=SkillSource.DESCRIPTION)
        return [skill.text for skill in extracted]

    def extract_skills_from_vacancies(self, vacancies: list[dict] | list[Vacancy]) -> dict[str, Any]:
        # Шаг 1: конвертация
        vacancy_objects = []
        for vac in vacancies:
            if isinstance(vac, dict):
                try:
                    vacancy_objects.append(Vacancy.from_api(vac))
                except ValueError:
                    continue
            else:
                vacancy_objects.append(vac)

        # Шаг 2: подсчёт частот
        skill_freq = Counter()
        for vacancy in vacancy_objects:
            extracted = self.skill_parser.parse_vacancy(vacancy)
            skill_texts = [s.text for s in extracted if s.text]
            normalized = SkillNormalizer.normalize_batch(skill_texts)
            unique = list(dict.fromkeys([s for s in normalized if s]))
            for skill in unique:
                skill_freq[skill] += 1
        logger.info("Уникальных навыков", count=len(skill_freq))

        # Шаг 3: гибридные веса (BM25 + embeddings)
        hybrid_weights = self.hybrid_calc.calculate(vacancies)

        return {"frequencies": dict(skill_freq), "hybrid_weights": hybrid_weights}

    def extract_from_detailed(self, detailed_vacancies: list[dict]) -> dict[str, float]:
        self._validate_vacancies(detailed_vacancies)
        skill_freq = {}
        for vac in detailed_vacancies:
            skills = self._extract_from_description(vac.get("description", ""))
            for s in skills:
                skill_freq[s] = skill_freq.get(s, 0) + 1
        final_freq = self._validate_skills(skill_freq)
        logger.info("РџРѕСЃР»Рµ РІР°Р»РёРґР°С†РёРё", count=len(final_freq))

        # РЁР°Рі 4: РіРёР±СЂРёРґРЅС‹Рµ РІРµСЃР° (РІРЅСѓС‚СЂРё РІС‹Р·РѕРІРµС‚ BM25)
        hybrid_weights = self.hybrid_calc.calculate(vacancies)

        # РЁР°Рі 5: СЌРјР±РµРґРґРёРЅРіРё РґР»СЏ РІР°Р»РёРґРЅС‹С… РЅР°РІС‹РєРѕРІ
        skill_embeddings = self.embedding_cache.get_embeddings(list(final_freq.keys()))

        result = SkillExtractionResult(
            frequencies=final_freq,
            hybrid_weights=hybrid_weights,
            skill_embeddings={skill: emb.tolist() for skill, emb in skill_embeddings.items()},
        )
        return result.model_dump()

    # --------------------- СѓС‚РёР»РёС‚С‹ СЃРѕС…СЂР°РЅРµРЅРёСЏ Рё РІС‹РІРѕРґР° -------------------------
    def save_raw_vacancies(self, vacancies, filename="hh_vacancies.json"):
        filepath = config.DATA_RAW_DIR / filename
        data = [v.raw_data if isinstance(v, Vacancy) else v for v in vacancies]
        atomic_write_json(data, filepath)
        logger.info("Сохранено", path=str(filepath))

    def save_processed_frequencies(self, frequencies, filename="competency_frequency.json", apply_filter=True):
        if apply_filter:
            whitelist = load_it_skills()
            if whitelist:
                frequencies = filter_skills_by_whitelist(frequencies, whitelist)
        filepath = config.DATA_PROCESSED_DIR / filename
        atomic_write_json(frequencies, filepath)
        logger.info("Р§Р°СЃС‚РѕС‚С‹ СЃРѕС…СЂР°РЅРµРЅС‹", path=str(filepath))

    @staticmethod
    def _strip_html(text):
        return re.sub(r"<[^>]+>", " ", text).strip() if text else ""

    @staticmethod
    def clean_highlighttext(text):
        return re.sub(r"</?highlighttext[^>]*>", "", text, flags=re.IGNORECASE) if text else ""

    # =========================================================================
    # EXCEL
    # =========================================================================
    def aggregate_to_dataframe(self, vacancies: list[dict] | list[Vacancy]) -> pd.DataFrame:
        """
        РђРіСЂРµРіРёСЂСѓРµС‚ РґР°РЅРЅС‹Рµ РІ DataFrame РґР»СЏ Excel.
        РќР°РІС‹РєРё СЃРѕР±РёСЂР°СЋС‚СЃСЏ РёР· key_skills + С‚РµРєСЃС‚РѕРІРѕРіРѕ РїР°СЂСЃРµСЂР° (РѕР±СЉРµРґРёРЅРµРЅРёРµ).
        """
        rows = []

        for vac in vacancies:
            key_skill_names = []
            if isinstance(vac, Vacancy):
                key_skill_names = vac.get_skill_names()
                vac_name = vac.name
                employer_name = vac.employer.name
                area_name = vac.area.name
                vac_id = vac.id
                salary = str(vac.salary) if vac.salary else "РќРµ СѓРєР°Р·Р°РЅР°"
                parsed_skills = self.skill_parser.parse_vacancy(vac)
                text_skill_names = [s.text for s in parsed_skills if s.text]
                description = vac.description or ""
                snippet_req = vac.snippet.requirement if vac.snippet else ""
                snippet_resp = vac.snippet.responsibility if vac.snippet else ""
            else:
                key_skills = vac.get("key_skills", [])
                key_skill_names = [s["name"] for s in key_skills if isinstance(s, dict) and "name" in s]
                vac_name = vac.get("name", "Unknown")
                employer = vac.get("employer", {}) or {}
                employer_name = employer.get("name", "Unknown")
                area = vac.get("area", {}) or {}
                area_name = area.get("name", "Unknown")
                vac_id = vac.get("id")
                salary = "РќРµ СѓРєР°Р·Р°РЅР°"
                description = vac.get("description", "") or ""
                snippet = vac.get("snippet", {}) or {}
                snippet_req = snippet.get("requirement", "") or ""
                snippet_resp = snippet.get("responsibility", "") or ""
                text_skill_names = self.extract_skills_from_description(f"{description} {snippet_req} {snippet_resp}")

            all_skills = list(dict.fromkeys(key_skill_names + text_skill_names))
            try:
                from src.parsing.skills.skill_normalizer import SkillNormalizer

                all_skills = SkillNormalizer.deduplicate(all_skills)
            except Exception:
                pass

            rows.append(
                {
                    "Вакансия": vac_name,
                    "Компания": employer_name,
                    "Регион": area_name,
                    "ID": vac_id,
                    "Зарплата": salary,
                    "Навыков": len(all_skills),
                    "Навыки": ", ".join(all_skills),
                }
            )

        return pd.DataFrame(rows)

    def save_to_excel(self, df: pd.DataFrame, filename: str):
        """РЎРѕС…СЂР°РЅСЏРµС‚ DataFrame РІ Excel"""
        filepath = config.DATA_RESULT_DIR / filename
        df.to_excel(filepath, index=False, engine="openpyxl")
        logger.info("Excel С„Р°Р№Р» СЃРѕС…СЂР°РЅС‘РЅ", path=str(filepath))

    def print_vacancies_list(self, vacancies: list[dict] | list[Vacancy]):
        """Р’С‹РІРѕРґРёС‚ СЃРїРёСЃРѕРє РІР°РєР°РЅСЃРёР№ (РЅР°РІС‹РєРё: key_skills + С‚РµРєСЃС‚РѕРІРѕРµ РёР·РІР»РµС‡РµРЅРёРµ)"""
        for i, vac in enumerate(vacancies[:20], 1):
            key_skill_names = []
            text_skill_names = []

            if isinstance(vac, Vacancy):
                vac_name = vac.name
                employer_name = vac.employer.name
                area_name = vac.area.name
                key_skill_names = vac.get_skill_names()
                parsed = self.skill_parser.parse_vacancy(vac)
                text_skill_names = [s.text for s in parsed if s.text]
            else:
                vac_name = vac.get("name", "Unknown")
                employer = vac.get("employer", {}) or {}
                employer_name = employer.get("name", "Unknown")
                area = vac.get("area", {}) or {}
                area_name = area.get("name", "Unknown")
                ks = vac.get("key_skills", [])
                key_skill_names = [s["name"] for s in ks if isinstance(s, dict) and "name" in s]
                desc = vac.get("description", "") or ""
                snip = vac.get("snippet", {}) or {}
                req = snip.get("requirement", "") or ""
                resp = snip.get("responsibility", "") or ""
                text_skill_names = self.extract_skills_from_description(f"{desc} {req} {resp}")

            all_skills = list(dict.fromkeys(key_skill_names + text_skill_names))
            try:
                from src.parsing.skills.skill_normalizer import SkillNormalizer

                all_skills = SkillNormalizer.deduplicate(all_skills)
            except Exception:
                pass

            print(f"{i}. {vac_name} @ {employer_name} ({area_name})")
            if all_skills:
                print(f"   РќР°РІС‹РєРё: {', '.join(all_skills[:5])}")
