# src/analyzers/comparison/domain_analyzer.py
import json

import structlog

from src import config
from src.models.market_metrics import DomainMetrics

logger = structlog.get_logger(__name__)  # ← используется напрямую


class DomainAnalyzer:
    def __init__(self, domain_map_path=None):
        self.domain_map_path = domain_map_path or config.DOMAIN_MAP_PATH
        if self.domain_map_path.exists():
            with open(self.domain_map_path, encoding="utf-8") as f:
                self.domain_map = json.load(f)
            logger.info("domain_map_loaded", path=str(self.domain_map_path), domains=len(self.domain_map))
        else:
            logger.error("domain_map_file_not_found", path=str(self.domain_map_path))
            self.domain_map = {}

    def compute_domain_coverage(self, user_skills: list[str]) -> dict[str, DomainMetrics]:
        user_set = set(skill.lower().strip() for skill in user_skills)
        result = {}
        total_domains = len(self.domain_map)

        for domain_name, skills in self.domain_map.items():
            dm = DomainMetrics(domain=domain_name, required_skills=skills)
            dm.compute_coverage(user_set)
            dm.importance = 1.0 / max(total_domains, 1)
            result[domain_name] = dm

            # Детализация по каждому домену — только в DEBUG
            logger.debug(  # ← исправлено: self.logger → logger
                "Домен обработан",
                domain=domain_name,
                coverage=round(dm.coverage, 3),
                user_has=dm.user_has,
                total_required=dm.total_required,
            )

        # Сводка — INFO
        top_domain = max(result.items(), key=lambda x: x[1].coverage)
        avg_coverage = sum(d.coverage for d in result.values()) / len(result)

        logger.info(  # ← исправлено: self.logger → logger
            "Доменное покрытие рассчитано",
            total_domains=len(result),
            avg_coverage=round(avg_coverage, 3),
            top_domain=top_domain[0],
            top_coverage=round(top_domain[1].coverage, 3),
            user_skills_count=len(user_skills),
        )

        return result
