# tests/models/test_models.py
import pytest
from datetime import datetime
import numpy as np
from pydantic import ValidationError

from src.models.student import (
    StudentProfile, ExperienceLevel, ProfileEvaluation,
    ProfileComparison, merge_skills_hierarchically
)
from src.models.comparison import GapResult, ComparisonReport
from src.models.market_metrics import SkillMetrics, DomainMetrics
from src.models.vacancy import (
    KeySkill, Snippet, Salary, Area, Employer, Experience,
    Vacancy, VacancyCollection
)


# ==================== StudentProfile ====================

def test_student_profile_validation_success(sample_student):
    assert sample_student.profile_name == "base"
    assert len(sample_student.skills) == 5
    assert sample_student.target_level == "middle"


def test_student_profile_validation_error():
    with pytest.raises(ValidationError):
        StudentProfile(
            competencies=[],
            skills=[],
            target_level="junior"
        )


def test_student_profile_repr(sample_student):
    rep = repr(sample_student)
    assert "StudentProfile" in rep
    assert sample_student.profile_name in rep


def test_student_profile_with_embedding():
    emb = np.random.rand(384)
    student = StudentProfile(
        profile_name="test",
        skills=["python"],
        embedding=emb
    )
    assert student.embedding is not None
    assert len(student.embedding) == 384


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
        recommendation="Хороший уровень"
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
        readiness_score=58.0
    )
    eval2 = ProfileEvaluation(
        profile_name="advanced",
        student=sample_student,
        level="senior",
        market_coverage_score=85.0,
        skill_coverage=80.0,
        domain_coverage_score=75.0,
        readiness_score=82.0
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
        readiness_score=68.0
    )
    pc = ProfileComparison(evaluations=[eval1])
    result = pc.to_dict_for_json()
    assert result["total_profiles"] == 1
    assert "best_profile" in result


# ==================== merge_skills_hierarchically ====================

def test_merge_skills_no_duplicates():
    result = merge_skills_hierarchically(
        ["python", "sql"],
        ["sql", "docker"],
        ["docker", "git"]
    )
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
            priority="HIGH"
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
            readiness_score=68.0
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
            overall_recommendations=["Учить Docker"]
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
            evaluations=[sample_evaluation]
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
            evaluations=[sample_evaluation]
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
            cluster_relevance=0.7
        )
        score = sm.score(
            level_weights={"junior": 0.2, "middle": 0.5, "senior": 0.3},
            domain_bonus=0.5
        )
        assert 0.0 <= score <= 2.0  # с бонусом может быть >1

    def test_skill_metrics_score_no_bonus(self):
        sm = SkillMetrics(
            skill="python",
            gap_j=0.5,
            demand_j=0.8,
            cluster_relevance=0.7
        )
        score = sm.score(
            level_weights={"junior": 1.0, "middle": 0.0, "senior": 0.0}
        )
        assert 0.0 <= score <= 1.0


# ==================== DomainMetrics ====================

class TestDomainMetrics:
    def test_compute_coverage_full(self):
        dm = DomainMetrics(
            domain="Backend",
            required_skills=["python", "sql", "docker"]
        )
        coverage = dm.compute_coverage({"python", "sql", "docker"})
        assert coverage == 1.0
        assert dm.user_has == 3

    def test_compute_coverage_partial(self):
        dm = DomainMetrics(
            domain="Backend",
            required_skills=["python", "sql", "docker", "k8s"]
        )
        coverage = dm.compute_coverage({"python", "sql"})
        assert coverage == 0.5

    def test_compute_coverage_empty_required(self):
        dm = DomainMetrics(
            domain="Empty",
            required_skills=[]
        )
        coverage = dm.compute_coverage({"python"})
        assert coverage == 0.0

    def test_compute_coverage_case_insensitive(self):
        dm = DomainMetrics(
            domain="Test",
            required_skills=["Python", "SQL"]
        )
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
            "published_at": "2024-01-01T00:00:00"
        }

    def test_from_api_full(self, sample_api_data):
        vac = Vacancy.from_api(sample_api_data)
        assert vac.id == "123"
        assert vac.name == "Python Developer"
        assert len(vac.key_skills) == 2
        assert vac.experience_level == "middle"

    def test_from_api_minimal(self):
        data = {
            "id": "1",
            "name": "Job",
            "area": {"id": 1, "name": "Area"},
            "employer": {"id": "2", "name": "Emp"}
        }
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
        v1 = Vacancy(id="1", name="Job1", area=area, employer=employer,
                     key_skills=[KeySkill("Python")],
                     experience=Experience(id="between1and3", name="1-3"))
        v2 = Vacancy(id="2", name="Job2", area=area, employer=employer,
                     key_skills=[KeySkill("Python"), KeySkill("Django")],
                     experience=Experience(id="between6and10", name="6-10"))
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
        assert stats['total_vacancies'] == 2
        assert stats['total_unique_skills'] == 2

    def test_repr(self, sample_vacancies):
        coll = VacancyCollection(sample_vacancies, query="Python")
        assert "2 vacancies" in repr(coll)