# tests/analyzers/test_skill_correlation.py
import numpy as np
import pytest

from src.analyzers.skills.skill_correlation import SkillCorrelationAnalyzer


class TestSkillCorrelationAnalyzer:
    @pytest.fixture
    def sample_vacancies(self):
        return [
            ["python", "sql", "docker"],
            ["python", "fastapi", "docker"],
            ["python", "sql", "git"],
            ["java", "spring", "sql"],
            ["java", "hibernate", "maven"],
            ["javascript", "react", "node.js"],
            ["javascript", "vue", "css"],
            ["python", "tensorflow", "pandas"],
            ["python", "sklearn", "numpy"],
            ["python", "sql", "pytest"],
        ]

    def test_init_default(self):
        """Инициализация с параметрами по умолчанию"""
        analyzer = SkillCorrelationAnalyzer()
        assert analyzer._cooccurrence is not None
        assert analyzer._skill_freq is not None
        assert analyzer._total_vacancies == 0

    def test_fit_builds_cooccurrence(self, sample_vacancies):
        """fit строит матрицу совместной встречаемости"""
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(sample_vacancies)

        assert analyzer._total_vacancies == len(sample_vacancies)
        assert len(analyzer._skill_freq) > 0
        assert len(analyzer._cooccurrence) > 0
        assert "python" in analyzer._skill_freq

    def test_fit_empty_vacancies(self):
        """fit с пустым списком вакансий"""
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit([])
        assert analyzer._total_vacancies == 0
        assert analyzer._skill_freq == {}
        assert analyzer._cooccurrence == {}

    def test_get_top_skills(self, sample_vacancies):
        """get_top_skills возвращает топ-N навыков"""
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(sample_vacancies)

        top = analyzer.get_top_skills(top_n=3)
        assert len(top) <= 3
        assert isinstance(top, list)
        # python должен быть в топе (самый частый)
        assert "python" in top

    def test_get_top_skills_empty(self):
        """get_top_skills без fit"""
        analyzer = SkillCorrelationAnalyzer()
        top = analyzer.get_top_skills(top_n=5)
        assert top == []

    def test_get_correlation_matrix(self, sample_vacancies):
        """get_correlation_matrix возвращает матрицу"""
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(sample_vacancies)

        matrix = analyzer.get_correlation_matrix(top_n=5)
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape[0] <= 5
        assert matrix.shape[1] <= 5
        # Диагональ = 1.0
        if len(matrix) > 0:
            assert np.allclose(np.diag(matrix), 1.0)

    def test_get_correlation_matrix_empty(self):
        """get_correlation_matrix без fit"""
        analyzer = SkillCorrelationAnalyzer()
        matrix = analyzer.get_correlation_matrix(top_n=5)
        assert matrix.shape == (0, 0)

    def test_get_correlation_labeled(self, sample_vacancies):
        """get_correlation_labeled возвращает навыки и матрицу"""
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(sample_vacancies)

        skills, matrix = analyzer.get_correlation_labeled(top_n=5)
        assert isinstance(skills, list)
        assert isinstance(matrix, np.ndarray)
        assert len(skills) == matrix.shape[0]
        assert matrix.shape[0] == matrix.shape[1]

    def test_get_correlation_labeled_custom_skills(self, sample_vacancies):
        """get_correlation_labeled с указанными навыками"""
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(sample_vacancies)

        skills, matrix = analyzer.get_correlation_labeled(skills=["python", "sql", "docker"])
        assert skills == ["python", "sql", "docker"]
        assert matrix.shape == (3, 3)

    def test_get_related_skills(self, sample_vacancies):
        """get_related_skills возвращает связанные навыки"""
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(sample_vacancies)

        related = analyzer.get_related_skills("python", top_k=3, min_cooccurrence=1)
        assert isinstance(related, list)
        if related:
            assert isinstance(related[0], tuple)
            assert len(related[0]) == 2  # (skill, jaccard)

    def test_get_related_skills_unknown(self, sample_vacancies):
        """get_related_skills для неизвестного навыка"""
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(sample_vacancies)

        related = analyzer.get_related_skills("unknown_skill")
        assert related == []

    def test_get_related_skills_not_fitted(self):
        """get_related_skills без fit"""
        analyzer = SkillCorrelationAnalyzer()
        related = analyzer.get_related_skills("python")
        assert related == []

    def test_high_correlation_python_sql(self, sample_vacancies):
        """python и sql имеют высокую корреляцию"""
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(sample_vacancies)

        related = analyzer.get_related_skills("python", top_k=10, min_cooccurrence=1)
        related_skills = [r[0] for r in related]
        # sql или docker часто встречаются с python
        assert "sql" in related_skills or "docker" in related_skills

    def test_correlation_matrix_symmetric(self, sample_vacancies):
        """Матрица корреляции симметрична"""
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(sample_vacancies)

        matrix = analyzer.get_correlation_matrix(top_n=5)
        if len(matrix) > 0:
            assert np.allclose(matrix, matrix.T)

    def test_diagonal_is_one(self, sample_vacancies):
        """Диагональ матрицы = 1.0"""
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(sample_vacancies)

        matrix = analyzer.get_correlation_matrix(top_n=5)
        if len(matrix) > 0:
            assert np.allclose(np.diag(matrix), 1.0)

    def test_jaccard_normalization(self, sample_vacancies):
        """Jaccard-нормализация: значения в [0, 1]"""
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(sample_vacancies)

        matrix = analyzer.get_correlation_matrix(top_n=5)
        if len(matrix) > 0:
            assert np.all(matrix >= 0.0)
            assert np.all(matrix <= 1.0)

    def test_fit_normalizes_skills(self):
        """fit нормализует навыки (убирает дубли в рамках вакансии)"""
        vacancies = [
            ["Python", "python", "PYTHON"],  # дубли
            ["SQL", "sql"],
        ]
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(vacancies)

        # Должен быть 1 python, 1 sql
        assert analyzer._skill_freq.get("python", 0) == 1
        assert analyzer._skill_freq.get("sql", 0) == 1

    def test_large_dataset(self):
        """Тест на большом наборе данных"""
        np.random.seed(42)
        # Используем реальные навыки, которые проходят нормализацию
        skills_pool = [
            "python", "sql", "docker", "java", "javascript", "react", "vue", "angular",
            "django", "flask", "fastapi", "spring", "node.js", "typescript", "go", "rust",
            "postgresql", "mysql", "mongodb", "redis", "kubernetes", "jenkins", "git",
            "aws", "azure", "gcp", "terraform", "ansible", "prometheus", "grafana",
            "pytorch", "tensorflow", "pandas", "numpy", "scikit-learn", "spark",
            "kafka", "rabbitmq", "nginx", "apache", "graphql", "rest api", "grpc",
            "html", "css", "sass", "webpack", "vite", "redux", "mobx"
        ]
        vacancies = []
        for _ in range(200):
            n_skills = np.random.randint(2, 7)
            vac_skills = list(np.random.choice(skills_pool, size=n_skills, replace=False))
            vacancies.append(vac_skills)

        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(vacancies)

        assert analyzer._total_vacancies == 200
        assert len(analyzer._skill_freq) > 0
        assert len(analyzer._cooccurrence) > 0

    def test_fit_single_skill_vacancies(self):
        """fit с вакансиями по одному навыку (нет совместной встречаемости)"""
        vacancies = [
            ["python"],
            ["python"],
            ["sql"],
            ["sql"],
        ]
        analyzer = SkillCorrelationAnalyzer()
        analyzer.fit(vacancies)

        # Навыки есть, но совместная встречаемость = 0
        assert analyzer._skill_freq.get("python", 0) == 2
        assert analyzer._skill_freq.get("sql", 0) == 2
        assert len(analyzer._cooccurrence) == 0  # нет пар
