"""SkillExtractor — извлекает навыки из вакансий или загружает из кэша."""

import json
from pathlib import Path

import joblib
import structlog

from src import config
from src.analyzers.skills.trends import TrendAnalyzer
from src.artifacts import ArtifactManifest
from src.parsing.skills.vacancy_parser import VacancyParser
from src.parsing.utils import (
    filter_skills_by_whitelist,
    load_it_skills,
    map_to_competencies,
    print_top_competencies,
    print_top_skills,
)
from src.utils import load_competency_mapping

logger = structlog.get_logger("skill_extractor")


class SkillExtractor:
    """Извлекает частоты и hybrid_weights, инициализирует тренды."""

    def __init__(self, args):
        self.args = args

    def extract(self, vacancies: list, parser: VacancyParser, raw_file: Path | None = None):
        cache_path = config.PARSED_SKILLS_CACHE_PATH
        vacancies_hash = None
        if raw_file:
            vacancies_hash = self._get_file_hash(raw_file)

        # Попытка загрузить кэш
        cached_result = None
        if cache_path.exists() and vacancies_hash:
            self._check_manifest(cache_path)
            cached = joblib.load(cache_path)
            if cached and cached.get("source_hash") == vacancies_hash:
                self._console_info("✅ Загружен кэш результатов парсинга навыков")
                cached_result = cached["result"]

        if cached_result:
            skill_freq = cached_result["frequencies"]
            hybrid_weights_raw = cached_result.get("hybrid_weights", {})
        else:
            self._console_info("Извлечение навыков из вакансий...")
            result = parser.extract_skills_from_vacancies(vacancies)
            skill_freq = result["frequencies"]
            hybrid_weights_raw = result.get("hybrid_weights", {})
            # Сохраняем кэш
            cache_data = {"source_hash": vacancies_hash, "result": result}
            joblib.dump(cache_data, cache_path)
            self._console_info("💾 Кэш результатов сохранён")
            manifest = ArtifactManifest(artifact_path=cache_path, metrics={"num_skills": len(skill_freq)})
            manifest.save()

        # Тренды
        whitelist = load_it_skills()
        skill_freq_filtered = filter_skills_by_whitelist(skill_freq, whitelist) if whitelist else skill_freq
        trend_analyzer = TrendAnalyzer(skill_freq_filtered)
        trend_analyzer.save_snapshot(skill_freq_filtered, apply_whitelist=False)

        # Доп. выводы (маппинг компетенций, топ-навыков)
        parser.save_processed_frequencies(skill_freq, apply_filter=not self.args.no_filter)
        print_top_skills(skill_freq)

        try:
            mapping = load_competency_mapping()
            if mapping:
                comp_counter = map_to_competencies(skill_freq, mapping)
                if comp_counter:
                    from src.analyzers.skills.skill_filter import SkillFilter

                    filter_engine = SkillFilter()
                    cleaned_comp = {}
                    for skill, count in comp_counter.most_common():
                        skill_clean = skill.lower().strip()
                        if skill_clean in filter_engine.GENERIC_WORDS:
                            continue
                        cleaned_comp[skill_clean] = count
                    comp_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency_mapped.json"
                    with open(comp_freq_path, "w", encoding="utf-8") as f:
                        json.dump(cleaned_comp, f, ensure_ascii=False, indent=2)
                    print_top_competencies(comp_counter)
        except Exception as e:
            logger.exception("competency_mapping_error", error=str(e))

        return skill_freq, hybrid_weights_raw, trend_analyzer

    def _get_file_hash(self, filepath: Path) -> str:
        import hashlib

        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _check_manifest(self, cache_path: Path):
        manifest_path = cache_path.with_suffix(".manifest.json")
        if manifest_path.exists():
            try:
                manifest = ArtifactManifest.load(cache_path)
                if not manifest.is_compatible():
                    logger.warning("parsed_skills_cache_incompatible_manifest")
                    cache_path.unlink()
                    manifest_path.unlink()
            except Exception as e:
                logger.warning("parsed_skills_manifest_check_failed", error=str(e))

    def _console_info(self, msg):
        print(f"  {msg}")
