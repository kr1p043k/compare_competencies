# tests/models/test_models.py
import pytest
from src.models.student import StudentProfile
from pydantic import ValidationError
# tests/models/test_vacancy.py (или добавить в test_models.py)
from datetime import datetime
from src.models.vacancy import (
    KeySkill, Snippet, Salary, Area, Employer, Experience,
    Vacancy, VacancyCollection
)

def test_student_profile_validation_success(sample_student):
    assert sample_student.profile_name == "base"
    assert len(sample_student.skills) == 5


def test_student_profile_validation_error():
    """Теперь правильно вызывает ошибку (обязательное поле profile_name)"""
    with pytest.raises(ValidationError):
        StudentProfile(                     # ← без profile_name
            competencies=[],
            skills=[],
            target_level="junior"
        )


def test_vacancy_model_creation():
    area = Area(id=1, name="Москва")
    employer = Employer(id="123", name="Test Corp")
    vac = Vacancy(
        id="test-1",
        name="Test Vacancy",          # ← name, а не title
        area=area,
        employer=employer
    )
    assert vac.name == "Test Vacancy"
from src.models.comparison import GapResult, ComparisonReport


class TestGapResult:
    def test_valid_gap_result(self):
        gap = GapResult(skill="Python", frequency=150, demand_level="high")
        assert gap.skill == "Python"
        assert gap.frequency == 150
        assert gap.demand_level == "high"

    def test_gap_result_invalid_demand_level(self):
        with pytest.raises(ValidationError) as exc_info:
            GapResult(skill="Python", frequency=150, demand_level="critical")
        errors = exc_info.value.errors()
        assert any("demand_level" in err["loc"] for err in errors)

    def test_gap_result_missing_fields(self):
        with pytest.raises(ValidationError):
            GapResult(skill="Python", frequency=150)  # нет demand_level
        with pytest.raises(ValidationError):
            GapResult(skill="Python", demand_level="high")  # нет frequency
        with pytest.raises(ValidationError):
            GapResult(frequency=150, demand_level="high")  # нет skill

    def test_gap_result_type_coercion(self):
        gap = GapResult(skill="Python", frequency="150", demand_level="high")
        assert gap.frequency == 150
        assert isinstance(gap.frequency, int)

    def test_gap_result_serialization(self):
        gap = GapResult(skill="Python", frequency=150, demand_level="high")
        data = gap.model_dump()
        assert data == {"skill": "Python", "frequency": 150, "demand_level": "high"}
        json_str = gap.model_dump_json()
        assert '"skill":"Python"' in json_str


class TestComparisonReport:
    @pytest.fixture
    def valid_gaps(self):
        return {
            "high": [GapResult(skill="Python", frequency=150, demand_level="high")],
            "medium": [GapResult(skill="SQL", frequency=80, demand_level="medium")],
            "low": [GapResult(skill="Git", frequency=30, demand_level="low")],
        }

    @pytest.fixture
    def full_report_data(self, valid_gaps):
        return {
            "student_name": "base",
            "total_competencies": 25,
            "total_mapped_skills": 18,
            "coverage_percent": 72.0,
            "weighted_coverage_percent": 68.5,
            "covered_skills": ["Python", "SQL", "Git"],
            "high_demand_gaps": valid_gaps["high"],
            "medium_demand_gaps": valid_gaps["medium"],
            "low_demand_gaps": valid_gaps["low"],
            "recommendations": ["Изучить FastAPI", "Углубить знания Docker"],
        }

    def test_valid_comparison_report(self, full_report_data):
        report = ComparisonReport(**full_report_data)
        assert report.student_name == "base"
        assert report.coverage_percent == 72.0
        assert len(report.high_demand_gaps) == 1
        assert report.high_demand_gaps[0].skill == "Python"

    def test_comparison_report_missing_required_fields(self):
        with pytest.raises(ValidationError):
            ComparisonReport(student_name="base")  # много отсутствующих

    def test_comparison_report_empty_lists_allowed(self):
        report = ComparisonReport(
            student_name="empty",
            total_competencies=0,
            total_mapped_skills=0,
            coverage_percent=0.0,
            weighted_coverage_percent=0.0,
            covered_skills=[],
            high_demand_gaps=[],
            medium_demand_gaps=[],
            low_demand_gaps=[],
            recommendations=[],
        )
        assert report.covered_skills == []
        assert report.recommendations == []

    def test_comparison_report_type_coercion(self):
        report = ComparisonReport(
            student_name="test",
            total_competencies="30",
            total_mapped_skills="20",
            coverage_percent="65.5",
            weighted_coverage_percent="60.0",
            covered_skills=["Python"],
            high_demand_gaps=[],
            medium_demand_gaps=[],
            low_demand_gaps=[],
            recommendations=[],
        )
        assert isinstance(report.total_competencies, int)
        assert isinstance(report.coverage_percent, float)
        assert report.total_competencies == 30

    def test_comparison_report_nested_validation(self):
        with pytest.raises(ValidationError):
            ComparisonReport(
                student_name="base",
                total_competencies=25,
                total_mapped_skills=18,
                coverage_percent=72.0,
                weighted_coverage_percent=68.5,
                covered_skills=["Python"],
                high_demand_gaps=[{"skill": "Python", "frequency": 150, "demand_level": "INVALID"}],
                medium_demand_gaps=[],
                low_demand_gaps=[],
                recommendations=[],
            )

    def test_comparison_report_serialization(self, full_report_data):
        report = ComparisonReport(**full_report_data)
        data = report.model_dump()
        assert data["student_name"] == "base"
        assert isinstance(data["high_demand_gaps"][0], dict)

        json_str = report.model_dump_json()
        assert "base" in json_str
        assert "Python" in json_str

    def test_comparison_report_from_json(self):
        json_data = """
        {
            "student_name": "dc",
            "total_competencies": 30,
            "total_mapped_skills": 25,
            "coverage_percent": 83.3,
            "weighted_coverage_percent": 79.2,
            "covered_skills": ["Python", "SQL", "Pandas", "Scikit-learn"],
            "high_demand_gaps": [
                {"skill": "PyTorch", "frequency": 120, "demand_level": "high"}
            ],
            "medium_demand_gaps": [
                {"skill": "Docker", "frequency": 85, "demand_level": "medium"}
            ],
            "low_demand_gaps": [],
            "recommendations": ["Освоить PyTorch", "Изучить Docker"]
        }
        """
        report = ComparisonReport.model_validate_json(json_data)
        assert report.student_name == "dc"
        assert report.total_mapped_skills == 25
        assert report.high_demand_gaps[0].skill == "PyTorch"

class TestKeySkill:
    def test_create_valid_skill(self):
        skill = KeySkill("Python")
        assert skill.name == "Python"
        assert skill.id is None

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


class TestSnippet:
    def test_create_snippet(self):
        s = Snippet(requirement="Требования", responsibility="Обязанности")
        assert s.requirement == "Требования"
        assert s.responsibility == "Обязанности"

    def test_has_content_true(self):
        assert Snippet(requirement="req").has_content() is True
        assert Snippet(responsibility="resp").has_content() is True

    def test_has_content_false(self):
        assert Snippet().has_content() is False

    def test_get_full_text_both(self):
        s = Snippet(requirement="req", responsibility="resp")
        assert s.get_full_text() == "req\nresp"

    def test_get_full_text_only_req(self):
        s = Snippet(requirement="req")
        assert s.get_full_text() == "req"

    def test_get_full_text_empty(self):
        s = Snippet()
        assert s.get_full_text() == ""


class TestSalary:
    def test_salary_range(self):
        s = Salary(from_amount=100000, to_amount=150000, currency="USD")
        assert s.get_midpoint() == 125000
        assert repr(s) == "100000-150000 USD"

    def test_salary_from_only(self):
        s = Salary(from_amount=100000)
        assert s.get_midpoint() == 100000
        assert repr(s) == "от 100000 RUB"

    def test_salary_to_only(self):
        s = Salary(to_amount=150000)
        assert s.get_midpoint() == 150000
        assert repr(s) == "до 150000 RUB"

    def test_salary_none(self):
        s = Salary()
        assert s.get_midpoint() is None
        assert repr(s) == "Не указана"

    def test_default_currency(self):
        s = Salary(from_amount=100)
        assert s.currency == "RUB"


class TestArea:
    def test_area(self):
        a = Area(id=1, name="Москва")
        assert a.id == 1
        assert a.name == "Москва"

    def test_hash_and_eq(self):
        a1 = Area(1, "Moscow")
        a2 = Area(1, "Москва")
        assert hash(a1) == hash(a2)
        assert a1 == a2

    def test_repr(self):
        a = Area(1, "Москва")
        assert repr(a) == "Москва (ID 1)"


class TestEmployer:
    def test_employer(self):
        e = Employer(id="123", name="Company", url="http://url")
        assert e.id == "123"
        assert e.name == "Company"
        assert e.url == "http://url"

    def test_employer_url_optional(self):
        e = Employer(id="123", name="Company")
        assert e.url is None

    def test_repr(self):
        e = Employer("123", "Company")
        assert repr(e) == "Company"


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

    def test_experience_empty_id_defaults_to_middle(self):
        e = Experience(id="", name="Empty")
        assert e.get_level() == "middle"

    def test_repr(self):
        e = Experience(id="between1and3", name="1-3 года")
        assert "middle" in repr(e)


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
        assert vac.area.id == 1
        assert vac.employer.name == "Test Corp"
        assert len(vac.key_skills) == 2
        assert vac.description == "Some description"
        assert vac.snippet.requirement == "req"
        assert vac.salary.get_midpoint() == 125000
        assert vac.experience.get_level() == "middle"
        assert vac.experience_level == "middle"
        assert vac.published_at == "2024-01-01T00:00:00"
        assert vac.raw_data == sample_api_data

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
        assert vac.snippet is None
        assert vac.salary is None
        assert vac.experience is None
        assert vac.experience_level == "middle"

    def test_from_api_invalid_skill_skipped(self, caplog):
        data = {
            "id": "1",
            "name": "Job",
            "area": {"id": 1, "name": "Area"},
            "employer": {"id": "2", "name": "Emp"},
            "key_skills": [{"name": "  "}]  # пустой навык
        }
        vac = Vacancy.from_api(data)
        assert len(vac.key_skills) == 0

    def test_from_api_missing_id_raises(self):
        data = {"name": "Job"}
        with pytest.raises(ValueError, match="Отсутствует ID вакансии"):
            Vacancy.from_api(data)

    def test_from_api_missing_name_raises(self):
        data = {"id": "1"}
        with pytest.raises(ValueError, match="Отсутствует название вакансии"):
            Vacancy.from_api(data)

    def test_post_init_validation(self):
        # Успешное создание
        area = Area(1, "Moscow")
        employer = Employer("1", "Company")
        vac = Vacancy(id="1", name="Job", area=area, employer=employer)
        assert vac.experience_level == "middle"

    def test_post_init_invalid_id(self):
        area = Area(1, "Moscow")
        employer = Employer("1", "Company")
        with pytest.raises(ValueError, match="ID вакансии должен быть непустой строкой"):
            Vacancy(id="", name="Job", area=area, employer=employer)

    def test_post_init_invalid_name(self):
        area = Area(1, "Moscow")
        employer = Employer("1", "Company")
        with pytest.raises(ValueError, match="Название вакансии должно быть непустой строкой"):
            Vacancy(id="1", name="", area=area, employer=employer)

    def test_experience_level_from_experience(self):
        area = Area(1, "Moscow")
        employer = Employer("1", "Company")
        exp = Experience(id="between6and10", name="6-10 лет")
        vac = Vacancy(id="1", name="Job", area=area, employer=employer, experience=exp)
        assert vac.experience_level == "senior"

    def test_get_all_text(self, sample_api_data):
        vac = Vacancy.from_api(sample_api_data)
        text = vac.get_all_text()
        assert "Python Developer" in text
        assert "Some description" in text
        assert "req" in text

    def test_get_skill_names(self, sample_api_data):
        vac = Vacancy.from_api(sample_api_data)
        assert vac.get_skill_names() == ["Python", "Django"]

    def test_has_skills(self, sample_api_data):
        vac = Vacancy.from_api(sample_api_data)
        assert vac.has_skills() is True
        vac2 = Vacancy(id="1", name="Job", area=Area(1, "A"), employer=Employer("1", "B"))
        assert vac2.has_skills() is False

    def test_repr(self, sample_api_data):
        vac = Vacancy.from_api(sample_api_data)
        assert "Python Developer" in repr(vac)
        assert "123" in repr(vac)

    def test_hash_eq(self, sample_api_data):
        vac1 = Vacancy.from_api(sample_api_data)
        vac2 = Vacancy.from_api(sample_api_data)
        assert vac1 == vac2
        assert hash(vac1) == hash(vac2)
        vac3 = Vacancy(id="999", name="Other", area=Area(1, "A"), employer=Employer("1", "B"))
        assert vac1 != vac3


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
        v3 = Vacancy(id="3", name="Job3", area=area, employer=employer,
                     key_skills=[], experience=None)
        return [v1, v2, v3]

    def test_add_and_iter(self, sample_vacancies):
        coll = VacancyCollection()
        for v in sample_vacancies:
            coll.add(v)
        assert len(coll) == 3
        # Добавление дубликата
        coll.add(sample_vacancies[0])
        assert len(coll) == 3

    def test_get_all_skills(self, sample_vacancies):
        coll = VacancyCollection(sample_vacancies)
        skills = coll.get_all_skills()
        assert len(skills) == 2  # Python, Django
        names = {s.name for s in skills}
        assert names == {"Python", "Django"}

    def test_get_stats(self, sample_vacancies):
        coll = VacancyCollection(sample_vacancies, query="test")
        stats = coll.get_stats()
        assert stats['total_vacancies'] == 3
        assert stats['vacancies_with_skills'] == 2
        assert stats['total_unique_skills'] == 2
        assert stats['avg_skills_per_vacancy'] == 1.0
        assert stats['by_level'] == {'junior': 0, 'middle': 2, 'senior': 1}

    def test_repr(self, sample_vacancies):
        coll = VacancyCollection(sample_vacancies, query="Python")
        assert "3 vacancies" in repr(coll)
        assert "Python" in repr(coll)