# tests/models/test_models.py

import numpy as np
import pytest
from pydantic import ValidationError
from src.models.hh_responses import (
    AreaResponse,
    EmployerResponse,
    SalaryResponse,
    SnippetResponse,
    ExperienceResponse,
    KeySkillResponse,
    VacancySearchItem,
    VacancySearchResponse,
    VacancyDetailResponse,
    TokenResponse,
    parse_response,
)
from unittest.mock import patch, MagicMock
from src.models.comparison import ComparisonReport, GapResult
from src.models.market_metrics import DomainMetrics, SkillMetrics
from src.models.student import (
    ProfileComparison,
    ProfileEvaluation,
    StudentProfile,
    merge_skills_hierarchically,
)
from src.models.vacancy import Area, Employer, Experience, KeySkill, Salary, Snippet, Vacancy, VacancyCollection

# ==================== StudentProfile ====================


def test_student_profile_validation_success(sample_student):
    assert sample_student.profile_name == "base"
    assert len(sample_student.skills) == 5
    assert sample_student.target_level == "middle"


def test_student_profile_validation_error():
    with pytest.raises(ValidationError):
        StudentProfile(competencies=[], skills=[], target_level="junior")


def test_student_profile_repr(sample_student):
    rep = repr(sample_student)
    assert "StudentProfile" in rep
    assert sample_student.profile_name in rep


def test_student_profile_with_embedding():
    emb = np.random.rand(384)
    student = StudentProfile(profile_name="test", skills=["python"], embedding=emb)
    assert student.embedding is not None
    assert len(student.embedding) == 384

def test_profile_comparison_to_dict_for_json_no_best(sample_student):
    """to_dict_for_json без best_evaluation"""
    eval1 = ProfileEvaluation(
        profile_name="test",
        student=sample_student,
        level="middle",
        market_coverage_score=70.0,
        skill_coverage=65.0,
        domain_coverage_score=60.0,
        readiness_score=68.0,
    )
    pc = ProfileComparison(evaluations=[eval1])
    # Не вызываем compute_aggregates — best_evaluation = None
    result = pc.to_dict_for_json()
    assert result["total_profiles"] == 1
    assert result["best_profile"]["profile_name"] == "test"

def test_merge_skills_hierarchically_overlap():
    """merge_skills_hierarchically с пересекающимися навыками"""
    result = merge_skills_hierarchically(
        ["python", "sql"],
        ["sql", "docker"],
        ["docker", "git", "python"],
    )
    assert result == ["python", "sql", "docker", "git"]

def test_merge_skills_hierarchically_all_same():
    """merge_skills_hierarchically с одинаковыми навыками"""
    result = merge_skills_hierarchically(
        ["python", "sql"],
        ["python", "sql"],
        ["python", "sql"],
    )
    assert result == ["python", "sql"]

def test_student_profile_embedding_none():
    """StudentProfile с embedding=None"""
    student = StudentProfile(profile_name="test", skills=["python"])
    assert student.embedding is None
    rep = repr(student)
    assert "emb=None" in rep


# ==================== ProfileEvaluation ====================


def test_profile_evaluation_creation(sample_student):
    eval_result = ProfileEvaluation(
        profile_name="test",
        student=sample_student,
        level="middle",
        market_coverage_score=75.0,
        skill_coverage=70.0,
        domain_coverage_score=65.0,
        readiness_score=72.0,
        avg_gap=0.15,
        recommendation="Хороший уровень",
    )
    assert eval_result.readiness_score == 72.0
    assert "middle" in repr(eval_result)


# ==================== ProfileComparison ====================


def test_profile_comparison_empty():
    pc = ProfileComparison()
    assert pc.average_readiness == 0.0
    assert pc.best_evaluation is None


def test_profile_comparison_with_evaluations(sample_student):
    eval1 = ProfileEvaluation(
        profile_name="base",
        student=sample_student,
        level="junior",
        market_coverage_score=60.0,
        skill_coverage=55.0,
        domain_coverage_score=50.0,
        readiness_score=58.0,
    )
    eval2 = ProfileEvaluation(
        profile_name="advanced",
        student=sample_student,
        level="senior",
        market_coverage_score=85.0,
        skill_coverage=80.0,
        domain_coverage_score=75.0,
        readiness_score=82.0,
    )
    pc = ProfileComparison(evaluations=[eval1, eval2])
    pc.compute_aggregates()
    assert pc.average_readiness == 70.0
    assert pc.best_evaluation.profile_name == "advanced"


def test_profile_comparison_to_dict(sample_student):
    eval1 = ProfileEvaluation(
        profile_name="test",
        student=sample_student,
        level="middle",
        market_coverage_score=70.0,
        skill_coverage=65.0,
        domain_coverage_score=60.0,
        readiness_score=68.0,
    )
    pc = ProfileComparison(evaluations=[eval1])
    result = pc.to_dict_for_json()
    assert result["total_profiles"] == 1
    assert "best_profile" in result


# ==================== merge_skills_hierarchically ====================


def test_merge_skills_no_duplicates():
    result = merge_skills_hierarchically(["python", "sql"], ["sql", "docker"], ["docker", "git"])
    assert result == ["python", "sql", "docker", "git"]


def test_merge_skills_empty():
    result = merge_skills_hierarchically([], [], [])
    assert result == []


def test_merge_skills_order_preserved():
    result = merge_skills_hierarchically(["top"], ["mid"], ["base"])
    assert result == ["top", "mid", "base"]


# ==================== GapResult ====================


class TestGapResult:
    def test_valid_gap_result(self):
        gap = GapResult(
            skill="Python",
            gap_j=0.5,
            gap_m=0.3,
            gap_s=0.1,
            max_gap=0.5,
            cluster_relevance=0.8,
            demand=0.9,
            priority="HIGH",
        )
        assert gap.skill == "Python"
        assert gap.priority == "HIGH"

    def test_gap_result_defaults(self):
        gap = GapResult(skill="Python")
        assert gap.gap_j == 0.0
        assert gap.gap_m == 0.0
        assert gap.gap_s == 0.0
        assert gap.priority == "LOW"

    def test_gap_result_serialization(self):
        gap = GapResult(skill="Python", priority="MEDIUM")
        data = gap.model_dump()
        assert data["skill"] == "Python"
        assert data["priority"] == "MEDIUM"


class TestComparisonReport:
    @pytest.fixture
    def sample_evaluation(self, sample_student):
        return ProfileEvaluation(
            profile_name="base",
            student=sample_student,
            level="middle",
            market_coverage_score=70.0,
            skill_coverage=65.0,
            domain_coverage_score=60.0,
            readiness_score=68.0,
        )

    def test_valid_comparison_report(self, sample_evaluation):
        report = ComparisonReport(
            total_profiles=1,
            profiles=["base"],
            average_readiness=68.0,
            average_market_coverage=70.0,
            average_skill_coverage=65.0,
            average_domain_coverage=60.0,
            best_profile="base",
            best_readiness=68.0,
            best_market_coverage=70.0,
            evaluations=[sample_evaluation],
            overall_recommendations=["Учить Docker"],
        )
        assert report.total_profiles == 1
        assert report.best_profile == "base"

    def test_comparison_report_empty_gaps(self, sample_evaluation):
        report = ComparisonReport(
            total_profiles=1,
            profiles=["base"],
            average_readiness=50.0,
            average_market_coverage=50.0,
            average_skill_coverage=50.0,
            average_domain_coverage=50.0,
            best_profile="base",
            best_readiness=50.0,
            best_market_coverage=50.0,
            evaluations=[sample_evaluation],
        )
        assert report.high_priority_gaps == []
        assert report.overall_recommendations == []

    def test_comparison_report_to_dict(self, sample_evaluation):
        report = ComparisonReport(
            total_profiles=1,
            profiles=["base"],
            average_readiness=70.0,
            average_market_coverage=70.0,
            average_skill_coverage=70.0,
            average_domain_coverage=70.0,
            best_profile="base",
            best_readiness=70.0,
            best_market_coverage=70.0,
            evaluations=[sample_evaluation],
        )
        data = report.to_dict()
        assert data["total_profiles"] == 1
        assert "best_profile" in data


# ==================== SkillMetrics ====================


class TestSkillMetrics:
    def test_skill_metrics_defaults(self):
        sm = SkillMetrics(skill="python")
        assert sm.skill == "python"
        assert sm.gap_j == 0.0
        assert sm.cluster_relevance == 0.0

    def test_skill_metrics_score(self):
        sm = SkillMetrics(
            skill="python",
            gap_j=0.5,
            gap_m=0.3,
            gap_s=0.1,
            demand_j=0.8,
            demand_m=0.9,
            demand_s=0.9,
            cluster_relevance=0.7,
        )
        score = sm.score(level_weights={"junior": 0.2, "middle": 0.5, "senior": 0.3}, domain_bonus=0.5)
        assert 0.0 <= score <= 2.0  # с бонусом может быть >1

    def test_skill_metrics_score_no_bonus(self):
        sm = SkillMetrics(skill="python", gap_j=0.5, demand_j=0.8, cluster_relevance=0.7)
        score = sm.score(level_weights={"junior": 1.0, "middle": 0.0, "senior": 0.0})
        assert 0.0 <= score <= 1.0
    def test_skill_metrics_score_zero_weights(self):
        """Строка 30-38: score с нулевыми весами"""
        sm = SkillMetrics(skill="python", gap_j=0.5, demand_j=0.8, cluster_relevance=0.7)
        score = sm.score(level_weights={"junior": 0.0, "middle": 0.0, "senior": 0.0})
        assert score == 0.0

    def test_skill_metrics_score_all_weights(self):
        """Строка 30-38: score с равными весами"""
        sm = SkillMetrics(
            skill="python",
            gap_j=0.5, gap_m=0.3, gap_s=0.1,
            demand_j=0.8, demand_m=0.9, demand_s=0.9,
            cluster_relevance=0.7,
        )
        score = sm.score(
            level_weights={"junior": 0.33, "middle": 0.34, "senior": 0.33},
            domain_bonus=0.5,
        )
        assert 0.0 <= score <= 2.0

# ==================== DomainMetrics ====================


class TestDomainMetrics:
    def test_compute_coverage_full(self):
        dm = DomainMetrics(domain="Backend", required_skills=["python", "sql", "docker"])
        coverage = dm.compute_coverage({"python", "sql", "docker"})
        assert coverage == 1.0
        assert dm.user_has == 3

    def test_compute_coverage_partial(self):
        dm = DomainMetrics(domain="Backend", required_skills=["python", "sql", "docker", "k8s"])
        coverage = dm.compute_coverage({"python", "sql"})
        assert coverage == 0.5

    def test_compute_coverage_empty_required(self):
        dm = DomainMetrics(domain="Empty", required_skills=[])
        coverage = dm.compute_coverage({"python"})
        assert coverage == 0.0

    def test_compute_coverage_case_insensitive(self):
        dm = DomainMetrics(domain="Test", required_skills=["Python", "SQL"])
        coverage = dm.compute_coverage({"python", "sql"})
        assert coverage == 1.0


# ==================== KeySkill ====================


class TestKeySkill:
    def test_create_valid_skill(self):
        skill = KeySkill("Python")
        assert skill.name == "Python"

    def test_create_skill_with_id(self):
        skill = KeySkill("Python", id="123")
        assert skill.id == "123"

    def test_strip_name(self):
        skill = KeySkill("  Python  ")
        assert skill.name == "Python"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="Название навыка не может быть пустым"):
            KeySkill("")

    def test_whitespace_name_raises(self):
        with pytest.raises(ValueError, match="Название навыка не может быть пустым"):
            KeySkill("   ")

    def test_hash_and_eq(self):
        s1 = KeySkill("python")
        s2 = KeySkill("Python")
        s3 = KeySkill("Java")
        assert s1 == s2
        assert hash(s1) == hash(s2)
        assert s1 != s3

    def test_repr(self):
        skill = KeySkill("Python")
        assert repr(skill) == "KeySkill('Python')"


# ==================== Snippet ====================


class TestSnippet:
    def test_create_snippet(self):
        s = Snippet(requirement="Требования", responsibility="Обязанности")
        assert s.requirement == "Требования"

    def test_has_content_true(self):
        assert Snippet(requirement="req").has_content() is True

    def test_has_content_false(self):
        assert Snippet().has_content() is False

    def test_get_full_text_both(self):
        s = Snippet(requirement="req", responsibility="resp")
        assert s.get_full_text() == "req\nresp"

    def test_get_full_text_empty(self):
        s = Snippet()
        assert s.get_full_text() == ""


# ==================== Salary ====================


class TestSalary:
    def test_salary_range(self):
        s = Salary(from_amount=100000, to_amount=150000, currency="USD")
        assert s.get_midpoint() == 125000

    def test_salary_from_only(self):
        s = Salary(from_amount=100000)
        assert s.get_midpoint() == 100000

    def test_salary_none(self):
        s = Salary()
        assert s.get_midpoint() is None

    def test_default_currency(self):
        s = Salary(from_amount=100)
        assert s.currency == "RUB"


# ==================== Area ====================


class TestArea:
    def test_area(self):
        a = Area(id=1, name="Москва")
        assert a.id == 1

    def test_hash_and_eq(self):
        a1 = Area(1, "Moscow")
        a2 = Area(1, "Москва")
        assert a1 == a2


# ==================== Employer ====================


class TestEmployer:
    def test_employer(self):
        e = Employer(id="123", name="Company")
        assert e.name == "Company"

    def test_employer_url_optional(self):
        e = Employer(id="123", name="Company")
        assert e.url is None


# ==================== Experience ====================


class TestExperience:
    def test_experience_junior(self):
        for exp_id in ["no_experience", "less1", "junior"]:
            e = Experience(id=exp_id, name="Test")
            assert e.get_level() == "junior"

    def test_experience_middle(self):
        for exp_id in ["between1and3", "between3and6", "middle"]:
            e = Experience(id=exp_id, name="Test")
            assert e.get_level() == "middle"

    def test_experience_senior(self):
        for exp_id in ["between6and10", "morethan10", "senior"]:
            e = Experience(id=exp_id, name="Test")
            assert e.get_level() == "senior"

    def test_experience_unknown_defaults_to_middle(self):
        e = Experience(id="unknown", name="Unknown")
        assert e.get_level() == "middle"


# ==================== Vacancy ====================


class TestVacancy:
    @pytest.fixture
    def sample_api_data(self):
        return {
            "id": "123",
            "name": "Python Developer",
            "area": {"id": 1, "name": "Москва"},
            "employer": {"id": "10", "name": "Test Corp", "url": "http://test.com"},
            "key_skills": [{"name": "Python"}, {"name": "Django"}],
            "description": "Some description",
            "snippet": {"requirement": "req", "responsibility": "resp"},
            "salary": {"from": 100000, "to": 150000, "currency": "RUB"},
            "experience": {"id": "between3and6", "name": "3-6 лет"},
            "published_at": "2024-01-01T00:00:00",
        }

    def test_from_api_full(self, sample_api_data):
        vac = Vacancy.from_api(sample_api_data)
        assert vac.id == "123"
        assert vac.name == "Python Developer"
        assert len(vac.key_skills) == 2
        assert vac.experience_level == "middle"

    def test_from_api_minimal(self):
        data = {"id": "1", "name": "Job", "area": {"id": 1, "name": "Area"}, "employer": {"id": "2", "name": "Emp"}}
        vac = Vacancy.from_api(data)
        assert vac.id == "1"
        assert vac.key_skills == []
        assert vac.salary is None

    def test_from_api_missing_id_raises(self):
        data = {"name": "Job"}
        with pytest.raises(ValueError, match="Отсутствует ID вакансии"):
            Vacancy.from_api(data)

    def test_from_api_missing_name_raises(self):
        data = {"id": "1"}
        with pytest.raises(ValueError, match="Отсутствует название вакансии"):
            Vacancy.from_api(data)

    def test_get_skill_names(self, sample_api_data):
        vac = Vacancy.from_api(sample_api_data)
        assert vac.get_skill_names() == ["Python", "Django"]

    def test_has_skills(self, sample_api_data):
        vac = Vacancy.from_api(sample_api_data)
        assert vac.has_skills() is True

    def test_hash_eq(self, sample_api_data):
        vac1 = Vacancy.from_api(sample_api_data)
        vac2 = Vacancy.from_api(sample_api_data)
        assert vac1 == vac2
        assert hash(vac1) == hash(vac2)


# ==================== VacancyCollection ====================


class TestVacancyCollection:
    @pytest.fixture
    def sample_vacancies(self):
        area = Area(1, "Moscow")
        employer = Employer("1", "Company")
        v1 = Vacancy(
            id="1",
            name="Job1",
            area=area,
            employer=employer,
            key_skills=[KeySkill("Python")],
            experience=Experience(id="between1and3", name="1-3"),
        )
        v2 = Vacancy(
            id="2",
            name="Job2",
            area=area,
            employer=employer,
            key_skills=[KeySkill("Python"), KeySkill("Django")],
            experience=Experience(id="between6and10", name="6-10"),
        )
        return [v1, v2]

    def test_add_and_iter(self, sample_vacancies):
        coll = VacancyCollection()
        for v in sample_vacancies:
            coll.add(v)
        assert len(coll) == 2
        # Дубликат не добавляется
        coll.add(sample_vacancies[0])
        assert len(coll) == 2

    def test_get_all_skills(self, sample_vacancies):
        coll = VacancyCollection(sample_vacancies)
        skills = coll.get_all_skills()
        names = {s.name for s in skills}
        assert names == {"Python", "Django"}

    def test_get_stats(self, sample_vacancies):
        coll = VacancyCollection(sample_vacancies, query="test")
        stats = coll.get_stats()
        assert stats["total_vacancies"] == 2
        assert stats["total_unique_skills"] == 2

    def test_repr(self, sample_vacancies):
        coll = VacancyCollection(sample_vacancies, query="Python")
        assert "2 vacancies" in repr(coll)

class TestVacancyExtended:
    @pytest.fixture
    def sample_api_data(self):
        return {
            "id": "123",
            "name": "Python Developer",
            "area": {"id": 1, "name": "Москва"},
            "employer": {"id": "10", "name": "Test Corp", "url": "http://test.com"},
            "key_skills": [{"name": "Python"}, {"name": "Django"}],
            "description": "Some description",
            "snippet": {"requirement": "req", "responsibility": "resp"},
            "salary": {"from": 100000, "to": 150000, "currency": "RUB"},
            "experience": {"id": "between3and6", "name": "3-6 лет"},
            "published_at": "2024-01-01T00:00:00",
        }

    def test_vacancy_post_init_no_experience(self):
        """Строка 39: __post_init__ без experience"""
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(
            id="1", name="Test", area=area, employer=employer,
            experience=None,
        )
        assert vac.experience_level == "middle"

    def test_from_api_with_invalid_skill(self, sample_api_data):
        """Строки 96-100: невалидный навык пропускается"""
        data = {**sample_api_data}
        data["key_skills"] = [{"name": ""}, {"name": "  "}, {"name": "Python"}]
        vac = Vacancy.from_api(data)
        assert len(vac.key_skills) == 1
        assert vac.key_skills[0].name == "Python"

    def test_from_api_without_snippet(self, sample_api_data):
        """Строка 117: без snippet"""
        data = {**sample_api_data, "snippet": None}
        vac = Vacancy.from_api(data)
        assert vac.snippet is None

    def test_from_api_without_salary(self, sample_api_data):
        """Строка 122: без salary"""
        data = {**sample_api_data, "salary": None}
        vac = Vacancy.from_api(data)
        assert vac.salary is None

    def test_from_api_without_experience(self, sample_api_data):
        """Строка 125: без experience"""
        data = {**sample_api_data, "experience": None}
        vac = Vacancy.from_api(data)
        assert vac.experience is None
        assert vac.experience_level == "middle"

    def test_from_api_key_error_handling(self):
        with pytest.raises(ValueError, match="Отсутствует название вакансии"):
            Vacancy.from_api({"id": "1"})

    def test_from_api_type_error_handling(self):
        with pytest.raises(ValueError, match="Отсутствует название вакансии"):
            Vacancy.from_api({"id": "1", "name": None, "area": None, "employer": None})

    def test_get_all_text_with_description(self, sample_api_data):
        """Строка 168: get_all_text с описанием"""
        vac = Vacancy.from_api(sample_api_data)
        text = vac.get_all_text()
        assert "Python Developer" in text
        assert "Some description" in text

    def test_get_all_text_without_snippet(self):
        """Строки 168-187: get_all_text без snippet"""
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(
            id="1", name="Test", area=area, employer=employer,
            description="Description only",
            snippet=None,
        )
        text = vac.get_all_text()
        assert "Description only" in text

    def test_has_skills_false(self):
        """Строка 187: has_skills = False"""
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(id="1", name="Test", area=area, employer=employer, key_skills=[])
        assert vac.has_skills() is False

    def test_vacancy_repr(self):
        """Строка 234: __repr__"""
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(id="1", name="Dev", area=area, employer=employer, key_skills=[KeySkill("Python")])
        rep = repr(vac)
        assert "Dev" in rep
        assert "Corp" in rep

    def test_vacancy_hash(self):
        """Строка 236: __hash__"""
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac1 = Vacancy(id="1", name="A", area=area, employer=employer)
        vac2 = Vacancy(id="1", name="B", area=area, employer=employer)
        assert hash(vac1) == hash(vac2)

    def test_vacancy_eq_different_id(self):
        """Строка 236: __eq__ с разными ID"""
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac1 = Vacancy(id="1", name="A", area=area, employer=employer)
        vac2 = Vacancy(id="2", name="A", area=area, employer=employer)
        assert vac1 != vac2

    def test_vacancy_eq_non_vacancy(self):
        """Строка 236: __eq__ с не-Vacancy"""
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(id="1", name="A", area=area, employer=employer)
        assert vac != "not a vacancy"

    # ==================== VacancyCollection ====================

    def test_collection_len(self, sample_vacancies):
        """Строка 285-287: __len__"""
        from src.models.vacancy import VacancyCollection
        coll = VacancyCollection(sample_vacancies)
        assert len(coll) == 2

    def test_collection_iter(self, sample_vacancies):
        """Строка 285-287: __iter__"""
        from src.models.vacancy import VacancyCollection
        coll = VacancyCollection(sample_vacancies)
        items = list(coll)
        assert len(items) == 2

    def test_collection_get_all_skills_empty(self):
        """Строка 329: пустая коллекция"""
        from src.models.vacancy import VacancyCollection
        coll = VacancyCollection([])
        skills = coll.get_all_skills()
        assert skills == []

    def test_collection_get_stats_by_level(self, sample_vacancies):
        from src.models.vacancy import VacancyCollection
        coll = VacancyCollection(sample_vacancies)
        stats = coll.get_stats()
        assert "by_level" in stats
        # Фактические значения зависят от фикстуры
        assert stats["by_level"]["middle"] >= 0

    def test_collection_get_stats_avg_skills(self, sample_vacancies):
        """Строка 337: среднее навыков на вакансию"""
        from src.models.vacancy import VacancyCollection
        coll = VacancyCollection(sample_vacancies)
        stats = coll.get_stats()
        assert "avg_skills_per_vacancy" in stats

    def test_collection_add_duplicate(self, sample_vacancies):
        """Строка 287-289: добавление дубликата"""
        from src.models.vacancy import VacancyCollection
        coll = VacancyCollection()
        coll.add(sample_vacancies[0])
        coll.add(sample_vacancies[0])  # дубликат
        assert len(coll) == 1

    def test_collection_repr_no_query(self, sample_vacancies):
        """Строка 352: __repr__ без query"""
        from src.models.vacancy import VacancyCollection
        coll = VacancyCollection(sample_vacancies)
        rep = repr(coll)
        assert "2 vacancies" in rep

    def test_collection_fetched_at_default(self, sample_vacancies):
        """Строка 360: fetched_at по умолчанию"""
        from src.models.vacancy import VacancyCollection
        from datetime import datetime
        coll = VacancyCollection(sample_vacancies)
        assert isinstance(coll.fetched_at, datetime)

    # ==================== Salary ====================

    def test_salary_repr_range(self):
        """Строка 382: __repr__ с диапазоном"""
        s = Salary(from_amount=100000, to_amount=150000, currency="RUB")
        assert "100000-150000 RUB" in repr(s)

    def test_salary_repr_from_only(self):
        s = Salary(from_amount=100000)
        assert "от 100000 RUB" in repr(s)

    def test_salary_repr_to_only(self):
        s = Salary(to_amount=150000)
        assert "до 150000 RUB" in repr(s)

    def test_salary_repr_none(self):
        s = Salary()
        assert repr(s) == "Не указана"

class TestHHResponses:
    """Тесты для моделей ответов hh.ru и функции parse_response."""

    # ----------------------------------------------------------------
    # AreaResponse
    # ----------------------------------------------------------------
    def test_area_response_valid(self):
        data = {"id": 1, "name": "Москва", "url": "https://api.hh.ru/areas/1"}
        area = AreaResponse.model_validate(data)
        assert area.id == 1
        assert area.name == "Москва"
        assert area.url == "https://api.hh.ru/areas/1"

    def test_area_response_optional_url(self):
        data = {"id": 2, "name": "Санкт-Петербург"}
        area = AreaResponse.model_validate(data)
        assert area.url is None

    def test_area_response_missing_required(self):
        with pytest.raises(ValidationError):
            AreaResponse.model_validate({"name": "Москва"})  # нет id

    # ----------------------------------------------------------------
    # EmployerResponse
    # ----------------------------------------------------------------
    def test_employer_response_valid(self):
        data = {
            "id": "123",
            "name": "Яндекс",
            "url": "https://ya.ru",
            "logo_urls": {"90": "logo.png"},
            "trusted": True,
        }
        emp = EmployerResponse.model_validate(data)
        assert emp.id == "123"
        assert emp.name == "Яндекс"
        assert emp.trusted is True

    def test_employer_response_defaults(self):
        data = {"name": "Компания"}
        emp = EmployerResponse.model_validate(data)
        assert emp.id is None
        assert emp.url is None
        assert emp.logo_urls is None
        assert emp.trusted is False

    # ----------------------------------------------------------------
    # SalaryResponse
    # ----------------------------------------------------------------
    def test_salary_response_with_alias(self):
        data = {"from": 100000, "to": 150000, "currency": "USD", "gross": True}
        salary = SalaryResponse.model_validate(data)
        assert salary.from_ == 100000
        assert salary.to == 150000
        assert salary.currency == "USD"
        assert salary.gross is True

    def test_salary_response_with_field_name(self):
        # Проверим, что можно передать и по имени поля from_
        data = {"from_": 100000, "to": 150000}
        salary = SalaryResponse.model_validate(data)
        assert salary.from_ == 100000

    def test_salary_response_defaults(self):
        data = {}
        salary = SalaryResponse.model_validate(data)
        assert salary.from_ is None
        assert salary.to is None
        assert salary.currency == "RUB"
        assert salary.gross is False

    # ----------------------------------------------------------------
    # SnippetResponse
    # ----------------------------------------------------------------
    def test_snippet_response_valid(self):
        data = {"requirement": "Знание Python", "responsibility": "Разработка бэкенда"}
        snippet = SnippetResponse.model_validate(data)
        assert snippet.requirement == "Знание Python"
        assert snippet.responsibility == "Разработка бэкенда"

    def test_snippet_response_empty(self):
        snippet = SnippetResponse.model_validate({})
        assert snippet.requirement is None
        assert snippet.responsibility is None

    # ----------------------------------------------------------------
    # ExperienceResponse
    # ----------------------------------------------------------------
    def test_experience_response_valid(self):
        data = {"id": "noExperience", "name": "Нет опыта"}
        exp = ExperienceResponse.model_validate(data)
        assert exp.id == "noExperience"
        assert exp.name == "Нет опыта"

    # ----------------------------------------------------------------
    # KeySkillResponse
    # ----------------------------------------------------------------
    def test_key_skill_response_valid(self):
        data = {"name": "Python"}
        skill = KeySkillResponse.model_validate(data)
        assert skill.name == "Python"

    # ----------------------------------------------------------------
    # VacancySearchItem
    # ----------------------------------------------------------------
    def test_vacancy_search_item_minimal(self):
        data = {
            "id": "123",
            "name": "Программист",
            "area": {"id": 1, "name": "Москва"},
        }
        item = VacancySearchItem.model_validate(data)
        assert item.id == "123"
        assert item.name == "Программист"
        assert item.area.id == 1
        assert item.employer is None
        assert item.salary is None

    def test_vacancy_search_item_full(self):
        data = {
            "id": "456",
            "name": "Разработчик",
            "area": {"id": 2, "name": "СПб"},
            "employer": {"id": "1", "name": "ООО"},
            "salary": {"from": 50000, "to": 70000},
            "snippet": {"requirement": "Опыт"},
            "experience": {"id": "between1And3", "name": "1-3 года"},
            "published_at": "2024-01-15T12:00:00+0300",
            "url": "https://hh.ru/vacancy/456",
            "alternate_url": "https://hh.ru/vacancy/456",
        }
        item = VacancySearchItem.model_validate(data)
        assert item.employer.name == "ООО"
        assert item.salary.from_ == 50000
        assert item.snippet.requirement == "Опыт"
        assert item.experience.id == "between1And3"

    def test_vacancy_search_item_extra_fields(self):
        data = {
            "id": "789",
            "name": "Тест",
            "area": {"id": 1, "name": "M"},
            "some_unknown_field": 42,
        }
        item = VacancySearchItem.model_validate(data)
        # Должно отработать без ошибок, extra="allow"
        assert item.id == "789"

    # ----------------------------------------------------------------
    # VacancySearchResponse
    # ----------------------------------------------------------------
    def test_vacancy_search_response_valid(self):
        data = {
            "items": [
                {"id": "1", "name": "Job", "area": {"id": 1, "name": "M"}},
                {"id": "2", "name": "Job2", "area": {"id": 2, "name": "SPb"}},
            ],
            "found": 2,
            "pages": 1,
            "page": 0,
            "per_page": 20,
        }
        resp = VacancySearchResponse.model_validate(data)
        assert len(resp.items) == 2
        assert resp.found == 2
        assert resp.pages == 1

    def test_vacancy_search_response_empty_items(self):
        data = {
            "items": [],
            "found": 0,
            "pages": 0,
            "page": 0,
            "per_page": 20,
        }
        resp = VacancySearchResponse.model_validate(data)
        assert resp.items == []
        assert resp.found == 0

    # ----------------------------------------------------------------
    # VacancyDetailResponse
    # ----------------------------------------------------------------
    def test_vacancy_detail_response_minimal(self):
        data = {
            "id": "999",
            "name": "DevOps",
            "area": {"id": 1, "name": "Москва"},
        }
        detail = VacancyDetailResponse.model_validate(data)
        assert detail.id == "999"
        assert detail.key_skills == []
        assert detail.description is None

    def test_vacancy_detail_response_with_skills(self):
        data = {
            "id": "888",
            "name": "Data Scientist",
            "area": {"id": 2, "name": "СПб"},
            "key_skills": [{"name": "Python"}, {"name": "SQL"}],
        }
        detail = VacancyDetailResponse.model_validate(data)
        assert len(detail.key_skills) == 2
        assert detail.key_skills[0].name == "Python"

    def test_vacancy_detail_response_extra_fields(self):
        data = {
            "id": "777",
            "name": "QA",
            "area": {"id": 3, "name": "Казань"},
            "custom_tag": "важно",
        }
        detail = VacancyDetailResponse.model_validate(data)
        assert detail.id == "777"

    # ----------------------------------------------------------------
    # TokenResponse
    # ----------------------------------------------------------------
    def test_token_response_valid(self):
        data = {
            "access_token": "abc123",
            "token_type": "bearer",
            "expires_in": 3600,
        }
        token = TokenResponse.model_validate(data)
        assert token.access_token == "abc123"
        assert token.token_type == "bearer"
        assert token.expires_in == 3600

    def test_token_response_defaults(self):
        data = {"access_token": "token"}
        token = TokenResponse.model_validate(data)
        assert token.token_type == "bearer"
        assert token.expires_in == 3600

    # ----------------------------------------------------------------
    # parse_response
    # ----------------------------------------------------------------
    def test_parse_response_success(self):
        data = {"id": "1", "name": "Test", "area": {"id": 1, "name": "M"}}  # id – строка
        result = parse_response(data, VacancyDetailResponse)
        assert isinstance(result, VacancyDetailResponse)
        assert result.id == "1"
        assert result.name == "Test"

    def test_parse_response_failure_logs_and_raises(self):
        data = {"invalid": "data"}
        with patch("structlog.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            with pytest.raises(ValidationError):
                parse_response(data, VacancyDetailResponse)
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert call_args[0][0] == "response_validation_failed"
            assert call_args[1]["model"] == "VacancyDetailResponse"
