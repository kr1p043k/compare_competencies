import json
from pathlib import Path

import structlog

from src import config
from src.models.market_metrics import DomainMetrics

logger = structlog.get_logger(__name__)


class ProfessionTaxonomy:
    """Загружает profession_taxonomy.json и предоставляет:
    - список профессий
    - домены для профессии
    - скиллы для профессии (из domain_map.json)
    - целевую профессию для профиля (base/dc/top_dc)
    - КРМ компетенции для профессии"""

    def __init__(self, taxonomy_path: Path | None = None, domain_map_path: Path | None = None,
                 krm_mapping_path: Path | None = None):
        self.taxonomy_path = taxonomy_path or config.PROFESSION_TAXONOMY_PATH
        self.domain_map_path = domain_map_path or config.DOMAIN_MAP_PATH
        self.krm_mapping_path = krm_mapping_path or config.KRM_MAPPING_PATH
        self._taxonomy: dict | None = None
        self._domain_map: dict | None = None
        self._krm_mapping: dict | None = None
        self._load()

    def _load(self):
        if not self.taxonomy_path.exists():
            logger.error("taxonomy_file_not_found", path=str(self.taxonomy_path))
            self._taxonomy = {"professions": {}, "profile_targets": {}}
            return
        with open(self.taxonomy_path, encoding="utf-8") as f:
            self._taxonomy = json.load(f)
        logger.info("taxonomy_loaded", professions=len(self._taxonomy.get("professions", {})))

        if not self.domain_map_path.exists():
            logger.error("domain_map_not_found", path=str(self.domain_map_path))
            self._domain_map = {}
            return
        with open(self.domain_map_path, encoding="utf-8") as f:
            self._domain_map = json.load(f)
        logger.info("domain_map_loaded", domains=len(self._domain_map))

        if not self.krm_mapping_path.exists():
            logger.warning("krm_mapping_not_found", path=str(self.krm_mapping_path))
            self._krm_mapping = {}
            return
        with open(self.krm_mapping_path, encoding="utf-8") as f:
            self._krm_mapping = json.load(f)
        logger.info("krm_mapping_loaded", codes=len(self._krm_mapping))

    @property
    def professions(self) -> list[str]:
        return list(self._taxonomy.get("professions", {}).keys())

    def get_profession_info(self, profession_name: str) -> dict | None:
        return self._taxonomy.get("professions", {}).get(profession_name)

    def get_domains_for_profession(self, profession_name: str) -> list[str]:
        info = self.get_profession_info(profession_name)
        if not info:
            return []
        return info.get("domains", [])

    def get_domain_skills(self, domain_name: str) -> list[str]:
        if not self._domain_map:
            return []
        return [s.lower().strip() for s in self._domain_map.get(domain_name, [])]

    def get_profession_skills(self, profession_name: str) -> set[str]:
        """Возвращает объединённый набор скиллов для профессии из всех её доменов."""
        domains = self.get_domains_for_profession(profession_name)
        skills = set()
        for domain in domains:
            domain_skills = self.get_domain_skills(domain)
            skills.update(domain_skills)
        return skills

    def get_profile_target(self, profile_name: str) -> dict | None:
        """Возвращает целевую профессию и домены для профиля (base/dc/top_dc)."""
        return self._taxonomy.get("profile_targets", {}).get(profile_name)

    def get_profession_competency_codes(self, profession_name: str) -> list[str]:
        info = self.get_profession_info(profession_name)
        if not info:
            return []
        return info.get("competency_codes", [])

    def get_competency_skills(self, competency_code: str) -> list[str]:
        """Возвращает скиллы для КРМ-компетенции."""
        if not self._krm_mapping:
            return []
        return [s.lower().strip() for s in self._krm_mapping.get(competency_code, [])]

    def get_profession_competency_skills(self, profession_name: str) -> set[str]:
        """Возвращает все навыки из КРМ-компетенций профессии."""
        codes = self.get_profession_competency_codes(profession_name)
        skills = set()
        for code in codes:
            skills.update(self.get_competency_skills(code))
        return skills

    def compute_krm_coverage(self, profession_name: str, user_skills: list[str]) -> dict:
        """Вычисляет покрытие каждой КРМ-компетенции профессии."""
        codes = self.get_profession_competency_codes(profession_name)
        user_set = set(s.lower().strip() for s in user_skills)
        result = {}
        for code in codes:
            req_skills = self.get_competency_skills(code)
            if not req_skills:
                result[code] = {"required": 0, "matched": 0, "coverage": 0.0}
                continue
            matched = sum(1 for s in req_skills if s in user_set)
            result[code] = {
                "required": len(req_skills),
                "matched": matched,
                "coverage": round(matched / len(req_skills), 4)
            }
        return result

    def compute_domain_coverage_for_profession(
        self, profession_name: str, user_skills: list[str]
    ) -> dict[str, DomainMetrics]:
        """Считает покрытие доменов целевой профессии."""
        domains = self.get_domains_for_profession(profession_name)
        user_set = set(s.lower().strip() for s in user_skills)
        result = {}
        for domain in domains:
            skills = self.get_domain_skills(domain)
            dm = DomainMetrics(domain=domain, required_skills=skills)
            dm.compute_coverage(user_set)
            result[domain] = dm
        return result

    def compute_weighted_profession_score(
        self, profession_name: str, user_skills: list[str], domain_weights: dict[str, float] | None = None
    ) -> tuple[float, dict[str, float]]:
        """Средневзвешенное покрытие по всем доменам профессии.
        Если domain_weights не заданы — равные веса."""
        coverages = self.compute_domain_coverage_for_profession(profession_name, user_skills)
        if not coverages:
            return 0.0, {}
        per_domain = {}
        total_weight = 0.0
        weighted_sum = 0.0
        for domain, dm in coverages.items():
            w = (domain_weights or {}).get(domain, 1.0 / len(coverages))
            weighted_sum += dm.coverage * w
            total_weight += w
            per_domain[domain] = dm.coverage
        final_score = (weighted_sum / total_weight * 100) if total_weight > 0 else 0.0
        return final_score, per_domain