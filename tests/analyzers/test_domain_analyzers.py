# tests/analyzers/test_domain_analyzer.py
import pytest
from src.analyzers.domain_analyzer import DomainAnalyzer


class TestDomainAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return DomainAnalyzer()

    def test_init_default(self, analyzer):
        assert analyzer.domain_map is not None
        assert len(analyzer.domain_map) > 0

    def test_init_custom_map(self):
        custom = {"Custom": ["skill1", "skill2"]}
        da = DomainAnalyzer(domain_map=custom)
        assert da.domain_map == custom

    def test_compute_domain_coverage_empty_skills(self, analyzer):
        result = analyzer.compute_domain_coverage([])
        assert isinstance(result, dict)
        # Все домены должны быть с нулевым покрытием
        for domain_name, metrics in result.items():
            assert metrics.coverage == 0.0

    def test_compute_domain_coverage_partial(self, analyzer):
        user_skills = ["python", "sql", "docker"]
        result = analyzer.compute_domain_coverage(user_skills)

        assert "Backend" in result
        # python + sql + docker — должно быть частичное покрытие Backend
        assert result["Backend"].coverage > 0

        # Backend покрыт не полностью (нужно много навыков)
        assert result["Backend"].coverage < 1.0

    def test_compute_domain_coverage_backend_full(self, analyzer):
        """Проверяем, что покрытие корректно вычисляется"""
        user_skills = ["python", "sql", "docker", "git"]
        result = analyzer.compute_domain_coverage(user_skills)
        backend = result["Backend"]
        assert 0.0 < backend.coverage < 1.0
        # importance по умолчанию 1.0
        assert backend.importance == 1.0

    def test_domain_coverage_case_insensitive(self, analyzer):
        user_skills = ["PYTHON", "Sql", "Docker"]
        result = analyzer.compute_domain_coverage(user_skills)
        backend = result["Backend"]
        assert backend.coverage > 0

    def test_domain_coverage_with_whitespace(self, analyzer):
        user_skills = ["  python  ", "sql", "\tdocker"]
        result = analyzer.compute_domain_coverage(user_skills)
        backend = result["Backend"]
        assert backend.coverage > 0

    def test_all_domains_present(self, analyzer):
        result = analyzer.compute_domain_coverage(["python"])
        expected_domains = {
            "Backend", "Frontend", "Data Science",
            "DevOps & Infrastructure", "Mobile Development",
            "QA / Testing", "1C & ERP", "AI/ML Research & Advanced",
            "Cybersecurity", "Game Development", "Embedded & Systems",
            "GIS & Spatial", "Soft Skills & Management"
        }
        for domain in expected_domains:
            assert domain in result, f"Domain {domain} missing from results"

    def test_domain_coverage_returns_domain_metrics(self, analyzer):
        result = analyzer.compute_domain_coverage(["python"])
        from src.models.market_metrics import DomainMetrics
        for domain_name, metrics in result.items():
            assert isinstance(metrics, DomainMetrics)
            assert metrics.domain == domain_name
            assert hasattr(metrics, 'coverage')
            assert hasattr(metrics, 'importance')
            assert hasattr(metrics, 'required_skills')