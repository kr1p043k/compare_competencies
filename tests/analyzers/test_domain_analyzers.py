# tests/analyzers/test_domain_analyzer.py
import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from src.analyzers.comparison.domain_analyzer import DomainAnalyzer
from src.models.market_metrics import DomainMetrics


@pytest.fixture
def mock_domain_map(tmp_path):
    """Создаёт временный domain_map.json с тестовыми данными."""
    domain_data = {
        "Backend": ["python", "sql", "docker", "git"],
        "Frontend": ["javascript", "react", "html", "css"],
        "Data Science": ["pandas", "numpy", "sklearn"],
        "DevOps & Infrastructure": ["kubernetes", "terraform", "ansible"],
        "Mobile Development": ["swift", "kotlin", "android"],
        "QA / Testing": ["selenium", "pytest", "jmeter"],
        "1C & ERP": ["1c", "erp", "sap"],
        "AI/ML Research & Advanced": ["tensorflow", "pytorch", "transformers"],
        "Cybersecurity": ["owasp", "nmap", "wireshark"],
        "Game Development": ["unity", "unreal", "csharp"],
        "Embedded & Systems": ["c", "cpp", "embedded"],
        "GIS & Spatial": ["qgis", "arcgis", "postgis"],
        "Soft Skills & Management": ["teamwork", "leadership", "agile"],
    }
    domain_file = tmp_path / "domain_map.json"
    domain_file.write_text(json.dumps(domain_data), encoding="utf-8")
    return domain_file


@pytest.fixture
def analyzer(mock_domain_map, monkeypatch):
    """Фикстура с подменой пути к domain_map."""
    monkeypatch.setattr("src.analyzers.comparison.domain_analyzer.config.DOMAIN_MAP_PATH", mock_domain_map)
    return DomainAnalyzer()


def test_init_default(analyzer):
    assert analyzer.domain_map is not None
    assert len(analyzer.domain_map) > 0


def test_init_custom_map():
    custom = {"Custom": ["skill1", "skill2"]}
    da = DomainAnalyzer(domain_map_path=None)  # path не используется, т.к. мы не подменяли config
    # Но для теста лучше передать фиктивный путь и замокать open
    with patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=json.dumps(custom))):
        da = DomainAnalyzer(domain_map_path=Path("/fake/path"))
        assert da.domain_map == custom


def test_compute_domain_coverage_empty_skills(analyzer):
    result = analyzer.compute_domain_coverage([])
    assert isinstance(result, dict)
    for domain_name, metrics in result.items():
        assert metrics.coverage == 0.0


def test_compute_domain_coverage_partial(analyzer):
    user_skills = ["python", "sql", "docker"]
    result = analyzer.compute_domain_coverage(user_skills)
    assert "Backend" in result
    assert result["Backend"].coverage > 0
    assert result["Backend"].coverage < 1.0


def test_compute_domain_coverage_backend_full(analyzer):
    user_skills = ["python", "sql", "docker", "git"]
    result = analyzer.compute_domain_coverage(user_skills)
    backend = result["Backend"]
    assert 0.0 < backend.coverage <= 1.0
    assert backend.importance == 1.0


def test_domain_coverage_case_insensitive(analyzer):
    user_skills = ["PYTHON", "Sql", "Docker"]
    result = analyzer.compute_domain_coverage(user_skills)
    backend = result["Backend"]
    assert backend.coverage > 0


def test_domain_coverage_with_whitespace(analyzer):
    user_skills = ["  python  ", "sql", "\tdocker"]
    result = analyzer.compute_domain_coverage(user_skills)
    backend = result["Backend"]
    assert backend.coverage > 0


def test_all_domains_present(analyzer):
    result = analyzer.compute_domain_coverage(["python"])
    # Проверяем, что все домены из загруженного словаря присутствуют
    expected_domains = set(analyzer.domain_map.keys())
    for domain in expected_domains:
        assert domain in result, f"Domain {domain} missing from results"


def test_domain_coverage_returns_domain_metrics(analyzer):
    result = analyzer.compute_domain_coverage(["python"])
    for domain_name, metrics in result.items():
        assert isinstance(metrics, DomainMetrics)
        assert metrics.domain == domain_name
        assert hasattr(metrics, "coverage")
        assert hasattr(metrics, "importance")
        assert hasattr(metrics, "required_skills")
