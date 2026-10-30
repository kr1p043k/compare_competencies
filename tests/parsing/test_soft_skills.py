"""Regression: soft skills must be extracted from vacancy text.

Before this, ``SkillParser`` only matched the IT whitelist (it_skills.json), so
requirements like «коммуникативные навыки» were silently dropped and never
appeared in the vacancy card.
"""
from __future__ import annotations

import pytest

from src.models.vacancy import Vacancy
from src.parsing.skills.skill_parser import SkillParser

VERIFICATION_DESC = """
Обязанности:
- Обрабатывать запросы на верификацию страниц, в частности на получение
  специализированных меток, в соответствии с критериями на разных площадках.
- Взаимодействовать со смежными командами и партнёрами.
- Соблюдать требования к срокам и качеству обработки заявок.

Требования:
- Опыт работы в поддержке пользователей или модерации контента от года.
- Умение работать с большим объёмом данных и всесторонне оценивать информацию.
- Коммуникативные навыки и грамотная письменная речь.
- Наличие ПК/ноутбука и стабильного интернета.
"""


def _parse(description: str) -> set[str]:
    vac = Vacancy.from_api({"id": "soft-1", "name": "Специалист по верификации"})
    vac.description = description
    result = SkillParser().parse_vacancy(vac)
    assert result.is_ok()
    return {s.text for s in result.unwrap()}


class TestSoftSkills:
    def test_communication_is_extracted(self) -> None:
        assert "коммуникация" in _parse(VERIFICATION_DESC)

    def test_large_data_volume_is_extracted(self) -> None:
        assert "многозадачность" in _parse(VERIFICATION_DESC)

    @pytest.mark.parametrize(
        ("phrase", "expected"),
        [
            ("Коммуникабельность обязательна", "коммуникация"),
            ("Грамотная письменная речь", "коммуникация"),
            ("Умеет работать в команде", "работа в команде"),
            ("Клиентоориентированность приветствуется", "клиентоориентированность"),
            ("Стрессоустойчивость", "стрессоустойчивость"),
            ("Навыки переговоров с партнёрами", "переговоры"),
            ("Наставничество junior-разработчиков", "наставничество"),
        ],
    )
    def test_phrase_variants(self, phrase: str, expected: str) -> None:
        assert expected in _parse(phrase)

    def test_no_false_positive_on_substring(self) -> None:
        # «переговорная» не должна давать навык «переговоры»
        assert "переговоры" not in _parse("Переговорная комната на этаже")

    def test_tech_skills_still_extracted(self) -> None:
        skills = {s.lower() for s in _parse("Требуется Python и PostgreSQL, коммуникативные навыки")}
        assert "python" in skills
        assert "коммуникация" in skills
