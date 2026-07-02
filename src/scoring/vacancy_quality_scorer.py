"""
VacancyQualityScorer — quality scoring and spam filtering for vacancies.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.errors import ScorerError
from src.models.vacancy import Vacancy
from src.parsing.skills.skill_parser import SkillParser
from src.result import Err, Ok, Result

logger = structlog.get_logger(__name__)


@dataclass
class SpamFlag:
    reason: str
    detail: str = ""

    def __repr__(self) -> str:
        return f"SpamFlag({self.reason}: {self.detail})"


@dataclass
class QualityScore:
    vacancy_id: str
    vacancy_name: str
    employer_name: str
    score: float
    is_spam: bool
    flags: list[SpamFlag] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        return [f.reason for f in self.flags]

    def __repr__(self) -> str:
        status = "SPAM" if self.is_spam else "OK"
        return f"[{status}] {self.vacancy_name} @ {self.employer_name} (score={self.score:.2f})"


SUSPICIOUS_EMPLOYER_PATTERNS = [
    r"кадров[а-я]+\s+агентств",
    r"рекрутинг",
    r"аутсорс",
    r"аутстаф",
    r"подбор\s+персонал",
    r"head[-\s]?hunter",
    r"рекрутмент",
    r"job\s*(?:solution|search|find)",
    r"ваканси[яи]\s+(?:от|в|на)",
    r"работа\s+(?:дл|в|на)",
]

GENERIC_VACANCY_PATTERNS = [
    r"^ваканси[яи]\b",
    r"^требует[ья]\b",
    r"^срочно\b",
    r"^разнорабоч\b",
    r"^подработк\b",
    r"^удаленн[аяо]+\s+работ\b",
    r"^менеджер\s*(?:по\s+)?(?:продаж|закупк|снабж|реклам)\b",
    r"^водител\b",
    r"^грузчик\b",
    r"^уборщик\b",
    r"^курьер\b",
    r"^охранник\b",
    r"^продавец\b",
    r"^кассир\b",
]

NON_IT_VACANCY_WORDS = [
    r"копи-?центр",
    r"ксерокопи",
    r"типографи",
    r"печать\b",
]

NON_IT_VACANCY_PATTERNS = [
    r"^копи-центр\b",
    r"^повар\b",
    r"^кондитер\b",
    r"^бармен\b",
    r"^официант\b",
    r"^шеф-повар\b",
    r"^пекарь\b",
    r"^мойщик\b",
    r"^уборщиц\b",
    r"^клининг\b",
    r"^горничн\b",
    r"^дворник\b",
    r"^врач\b",
    r"^медсестр\b",
    r"^медбрат\b",
    r"^фельдшер\b",
    r"^медицинск\b",
    r"^провизор\b",
    r"^фармацевт\b",
    r"^санитар\b",
    r"^водител\b",
    r"^экспедитор\b",
    r"^такс\b",
    r"^дальнобой\b",
    r"^тракторист\b",
    r"^машинист\b",
    r"^сторож\b",
    r"^вахтер\b",
    r"^консьерж\b",
    r"^штукатур\b",
    r"^маляр\b",
    r"^каменщик\b",
    r"^бетонщик\b",
    r"^арматурщик\b",
    r"^сварщик\b",
    r"^электромонт\b",
    r"^сантехник\b",
    r"^плотник\b",
    r"^столяр\b",
    r"^отделочник\b",
    r"^монтажник\b",
    r"^разнорабоч\b",
    r"^комплектовщик\b",
    r"^фасовщик\b",
    r"^упаковщик\b",
    r"^сортировщик\b",
    r"^кладовщик\b",
    r"^мерчендайзер\b",
    r"^товаровед\b",
    r"^администратор\s+(?:магазин|салон|торгов|зал|ресторан|кафе|гостиниц|склада)\b",
    r"^администратор\s*$",
    r"^бармен-официант\b",
    r"^хостес\b",
    r"^аниматор\b",
    r"^воспитател\b",
    r"^нян\b",
    r"^гувернантк\b",
    r"^сиделк\b",
    r"^социальн\s+работ\b",
    r"^парикмахер\b",
    r"^косметолог\b",
    r"^массажист\b",
    r"^мастер\s+маникюр\b",
    r"^фитнес\b",
    r"^тренер\b",
    r"^инструктор\b",
    r"^промоутер\b",
    r"^курьер\b",
    r"^грумер\b",
    r"^ветеринар\b",
    r"^флорист\b",
    r"^декоратор\b",
    r"^креативн(?:ый)?\s+директор\b",
    r"^администратор\s+склада\b",
    r"^менеджер\s+проектных\s+продаж\b",
    r"^менеджер\s+(?:по\s+)?продаж\b",
    r"^продавец\b",
    r"^руководитель\s+строительн\b",
    r"^охранник\b",
]

PROMO_KEYWORDS = [
    r"самая высокая зарплата",
    r"зарплата от \d{6}",
    r"доход от \d{6}",
    r"пассивный доход",
    r"быстрый заработок",
    r"работа без опыта",
    r"гибкий график",
    r"лучшие условия",
]

GENERIC_NAME_EXACT = {
    "вакансия", "работа", "сотрудник", "персонал",
    "менеджер", "водитель", "грузчик", "уборщик",
    "курьер", "продавец", "кассир", "охранник",
    "администратор", "секретарь", "разнорабочий",
}


def _count_urls(text: str) -> int:
    return len(re.findall(r'https?://\S+', text))


class VacancyQualityScorer:
    """
    Scores vacancies for quality and filters out spam.
    """

    def __init__(
        self,
        min_skills: int = 2,
        min_description_chars: int = 100,
        spam_threshold: float = 0.5,
    ):
        self.min_skills = min_skills
        self.min_description_chars = min_description_chars
        self.spam_threshold = spam_threshold
        self._skill_parser = SkillParser()
        self._suspicious_employer_re = re.compile(
            "|".join(f"(?:{p})" for p in SUSPICIOUS_EMPLOYER_PATTERNS),
            re.IGNORECASE,
        )
        self._generic_vacancy_re = re.compile(
            "|".join(f"(?:{p})" for p in GENERIC_VACANCY_PATTERNS),
            re.IGNORECASE,
        )
        self._promo_re = re.compile(
            "|".join(f"(?:{p})" for p in PROMO_KEYWORDS),
            re.IGNORECASE,
        )
        self._non_it_re = re.compile(
            "|".join(f"(?:{p})" for p in NON_IT_VACANCY_PATTERNS),
            re.IGNORECASE,
        )
        self._non_it_search_re = re.compile(
            "|".join(f"(?:{p})" for p in NON_IT_VACANCY_WORDS),
            re.IGNORECASE,
        )

    def score(self, vacancy: Vacancy) -> Result[QualityScore, ScorerError]:
        try:
            flags = []
            deductions = 0.0
            name_lower = (vacancy.name or "").lower().strip()
            employer_lower = (vacancy.employer.name or "").lower().strip()
            description = vacancy.description or ""
            snippet_req = (vacancy.snippet.requirement or "") if vacancy.snippet else ""
            snippet_resp = (vacancy.snippet.responsibility or "") if vacancy.snippet else ""
            all_text = f"{description} {snippet_req} {snippet_resp}"
            key_skills_count = len(vacancy.key_skills)
            match self._skill_parser.parse_vacancy(vacancy):
                case Ok(parsed):
                    total_skills = len(set(s.text.lower() for s in parsed))
                case Err(_):
                    total_skills = 0

            if not description.strip():
                flags.append(SpamFlag("NO_DESCRIPTION", "No description"))
                deductions += 0.25

            if len(description.strip()) < self.min_description_chars and description.strip():
                flags.append(SpamFlag("TOO_SHORT_DESCRIPTION", f"Description < {self.min_description_chars} chars"))
                deductions += 0.15

            if total_skills == 0:
                flags.append(SpamFlag("нет навыков", "No skills in key_skills, description, or snippet"))
                deductions += 0.6

            if key_skills_count == 0 and total_skills > 0:
                flags.append(SpamFlag("NO_KEY_SKILLS", "No key skills (but found in description/snippet)"))
                deductions += 0.05

            if 0 < key_skills_count < self.min_skills:
                flags.append(SpamFlag("TOO_FEW_KEY_SKILLS", f"Less than {self.min_skills} key skills"))
                deductions += 0.1

            if self._suspicious_employer_re.search(employer_lower):
                flags.append(SpamFlag("SUSPICIOUS_EMPLOYER", f"Employer: {vacancy.employer.name}"))
                deductions += 0.3

            if name_lower in GENERIC_NAME_EXACT or self._generic_vacancy_re.match(name_lower):
                flags.append(SpamFlag("GENERIC_NAME", f"Name: {vacancy.name}"))
                deductions += 0.3

            if self._non_it_re.match(name_lower) or self._non_it_search_re.search(name_lower):
                flags.append(SpamFlag("NOT_RELEVANT", f"Non IT: {vacancy.name}"))
                deductions += 0.7

            if self._promo_re.search(all_text):
                flags.append(SpamFlag("PROMO_DESCRIPTION", "Promotional text detected"))
                deductions += 0.2

            url_count = _count_urls(all_text)
            if url_count > 3:
                flags.append(SpamFlag("EXCESSIVE_URLS", f"{url_count} URLs in description"))
                deductions += 0.15

            if vacancy.salary:
                midpoint = vacancy.salary.get_midpoint()
                if midpoint and midpoint > 1_000_000:
                    flags.append(SpamFlag("SALARY_ANOMALY", f"Salary > 1M: {midpoint} {vacancy.salary.currency}"))
                    deductions += 0.2

            score = max(0.0, 1.0 - deductions)
            import math
            spam_prob = 1.0 / (1.0 + math.exp(-10 * (self.spam_threshold - score)))
            is_spam = score < self.spam_threshold
            return Ok(QualityScore(
                vacancy_id=vacancy.id,
                vacancy_name=vacancy.name,
                employer_name=vacancy.employer.name,
                score=score,
                is_spam=is_spam,
                flags=flags,
            ))
        except Exception as exc:
            return Err(ScorerError(message=str(exc), detail=f"vacancy={vacancy.id}"))

    def filter_vacancies(
        self, vacancies: list[Any]
    ) -> Result[tuple[list[Any], list[Any], dict[str, Any]], ScorerError]:
        try:
            clean = []
            spam = []
            scores = []

            for v in vacancies:
                match self.score(v):
                    case Ok(s):
                        scores.append(s)
                        if s.is_spam:
                            spam.append((v, s))
                        else:
                            clean.append(v)
                    case Err(e):
                        return Err(e)

            report = self._build_report(scores, len(vacancies))
            logger.info(
                "quality_scoring_done",
                total=len(vacancies),
                clean=len(clean),
                spam=len(spam),
                threshold=self.spam_threshold,
            )
            return Ok((clean, spam, report))
        except Exception as exc:
            return Err(ScorerError(message=str(exc), detail=f"total={len(vacancies)}"))

    def _build_report(self, scores: list[QualityScore], total: int) -> dict[str, Any]:
        spam_scores = [s for s in scores if s.is_spam]
        reason_counts: dict[str, int] = {}
        for s in spam_scores:
            for f in s.flags:
                reason_counts[f.reason] = reason_counts.get(f.reason, 0) + 1

        avg_score = sum(s.score for s in scores) / len(scores) if scores else 1.0

        return {
            "total_vacancies": total,
            "clean_count": total - len(spam_scores),
            "spam_count": len(spam_scores),
            "spam_rate": round(len(spam_scores) / total, 3) if total else 0.0,
            "avg_quality_score": round(avg_score, 3),
            "spam_reasons": dict(sorted(reason_counts.items(), key=lambda x: -x[1])),
            "spam_vacancies": [
                {
                    "id": s.vacancy_id,
                    "name": s.vacancy_name,
                    "employer": s.employer_name,
                    "score": s.score,
                    "flags": [{"reason": f.reason, "detail": f.detail} for f in s.flags],
                }
                for s in spam_scores
            ],
        }

    def print_report(self, report: dict[str, Any]) -> None:
        print(f"\n  {'=' * 50}")
        print("  VACANCY QUALITY REPORT")
        print(f"  {'=' * 50}")
        print(f"  Total vacancies:  {report['total_vacancies']}")
        print(f"  Clean:            {report['clean_count']}")
        print(f"  Spam:             {report['spam_count']} ({report['spam_rate']*100:.1f}%)")
        print(f"  Avg score:        {report['avg_quality_score']:.3f}")
        if report["spam_reasons"]:
            print("\n  Spam reasons:")
            for reason, count in report["spam_reasons"].items():
                print(f"    * {reason}: {count}")
        if report["spam_vacancies"]:
            print("\n  Spam vacancies:")
            for sv in report["spam_vacancies"]:
                reasons = "; ".join(f["reason"] for f in sv["flags"])
                print(f"    [{sv['score']:.2f}] {sv['name']} @ {sv['employer']}")
                print(f"         Reasons: {reasons}")
        print(f"  {'=' * 50}\n")