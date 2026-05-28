"""
Анализатор совместной встречаемости навыков в вакансиях.
Строит матрицу корреляций (co-occurrence) для топ-N навыков.
"""

from collections import defaultdict

import numpy as np
import structlog

from src import Result, Ok, Err
from src.errors import DomainError
from src.parsing.skills.skill_normalizer import SkillNormalizer

logger = structlog.get_logger(__name__)


class SkillCorrelationAnalyzer:
    """
    Анализирует, какие навыки часто встречаются вместе в вакансиях.
    """

    def __init__(self):
        self._cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
        self._skill_freq: dict[str, int] = defaultdict(int)
        self._total_vacancies = 0

    def fit(self, vacancies_skills: list[list[str]]):
        """
        Обучает анализатор на списке вакансий.

        Args:
            vacancies_skills: список списков навыков для каждой вакансии
        """
        self._cooccurrence.clear()
        self._skill_freq.clear()
        self._total_vacancies = len(vacancies_skills)

        for skills in vacancies_skills:
            # Нормализуем и убираем дубли в рамках одной вакансии
            normalized: set[str] = set()
            for s in skills:
                match SkillNormalizer.normalize(s):
                    case Ok(n):
                        normalized.add(n)

            # Считаем частоты
            for skill in normalized:
                self._skill_freq[skill] += 1

            # Считаем совместную встречаемость
            skill_list = sorted(normalized)
            for i in range(len(skill_list)):
                for j in range(i + 1, len(skill_list)):
                    pair = (skill_list[i], skill_list[j])
                    self._cooccurrence[pair] += 1

        logger.info(
            "correlation_analyzer_fitted",
            total_vacancies=self._total_vacancies,
            unique_skills=len(self._skill_freq),
            unique_pairs=len(self._cooccurrence),
        )

    def get_top_skills(self, top_n: int = 30) -> Result[list[str], DomainError]:
        """Возвращает топ-N навыков по частоте."""
        return Ok([s for s, _ in sorted(self._skill_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]])

    def get_correlation_matrix(self, skills: list[str] | None = None, top_n: int = 30) -> Result[np.ndarray, DomainError]:
        """
        Возвращает матрицу корреляций (нормированных) для указанных навыков.

        Нормировка: Jaccard = |A ∩ B| / (|A| + |B| - |A ∩ B|)
        где |A| — количество вакансий с навыком A.
        """
        if skills is None:
            top_result = self.get_top_skills(top_n)
            if top_result.is_err():
                return Err(top_result.err())
            skills = top_result.ok()

        n = len(skills)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    pair = (skills[i], skills[j]) if skills[i] < skills[j] else (skills[j], skills[i])
                    cooc = self._cooccurrence.get(pair, 0)
                    freq_a = self._skill_freq.get(skills[i], 0)
                    freq_b = self._skill_freq.get(skills[j], 0)
                    # Jaccard-нормировка: исключает влияние абсолютной популярности
                    denom = freq_a + freq_b - cooc
                    if denom > 0:
                        matrix[i][j] = round(cooc / denom, 3)
                        matrix[j][i] = matrix[i][j]

        return Ok(matrix)

    def get_correlation_labeled(self, skills: list[str] | None = None, top_n: int = 30) -> Result[tuple[list[str], np.ndarray], DomainError]:
        """
        Возвращает (список навыков, матрица корреляций).
        Удобно для визуализации.
        """
        if skills is None:
            top_result = self.get_top_skills(top_n)
            if top_result.is_err():
                return Err(top_result.err())
            skills = top_result.ok()
        matrix_result = self.get_correlation_matrix(skills)
        if matrix_result.is_err():
            return Err(matrix_result.err())
        return Ok((skills, matrix_result.ok()))

    def get_related_skills(self, skill: str, top_k: int = 10, min_cooccurrence: int = 3) -> Result[list[tuple[str, float]], DomainError]:
        """
        Возвращает навыки, наиболее связанные с указанным.
        """
        related = []
        match SkillNormalizer.normalize(skill):
            case Ok(n):
                skill_norm = n
            case _:
                skill_norm = ""
        freq_a = self._skill_freq.get(skill_norm, 0)

        for (a, b), cooc in self._cooccurrence.items():
            if a == skill_norm or b == skill_norm:
                other = b if a == skill_norm else a
                freq_b = self._skill_freq.get(other, 0)
                denom = freq_a + freq_b - cooc
                if denom > 0 and cooc >= min_cooccurrence:
                    jaccard = cooc / denom
                    related.append((other, round(jaccard, 3)))

        return Ok(sorted(related, key=lambda x: x[1], reverse=True)[:top_k])
