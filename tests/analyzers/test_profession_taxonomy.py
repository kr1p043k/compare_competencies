import json
import pytest

from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy, TaxonomyValidator
from src.models.market_metrics import DomainMetrics


SAMPLE_TAXONOMY = {
    "professions": {
        "Data Scientist": {
            "domains": ["Data Science"],
            "hh_queries": ["Data Scientist"],
            "aliases": ["data scientist"],
            "competency_codes": [],
        },
        "Python Developer": {
            "domains": ["Backend"],
            "hh_queries": ["Python Developer"],
            "aliases": ["python developer"],
            "competency_codes": ["ППК-Р1", "ППК-Р2", "ППК-Р3"],
        },
        "Fullstack Developer": {
            "domains": ["Backend", "Frontend"],
            "hh_queries": ["Fullstack Developer"],
            "aliases": ["fullstack developer"],
            "competency_codes": ["ППК-Р1", "ППК-Р2"],
        },
    },
    "profile_targets": {
        "base": {
            "target_profession": "Data Scientist",
            "target_domains": ["Data Science"],
            "description": "Test",
        }
    },
}

SAMPLE_DOMAIN_MAP = {
    "Backend": ["python", "sql", "docker", "git"],
    "Frontend": ["javascript", "html", "css"],
    "Data Science": ["python", "sql", "pandas", "numpy"],
}

SAMPLE_KRM = {
    "ППК-Р1": ["programming", "coding", "python", "java"],
    "ППК-Р2": ["testing", "debugging", "pytest", "code review"],
    "ППК-Р3": ["git", "github", "version control"],
}


class TestProfessionTaxonomy:
    @pytest.fixture
    def taxonomy_files(self, tmp_path):
        tax_path = tmp_path / "profession_taxonomy.json"
        tax_path.write_text(json.dumps(SAMPLE_TAXONOMY), encoding="utf-8")
        dom_path = tmp_path / "domain_map.json"
        dom_path.write_text(json.dumps(SAMPLE_DOMAIN_MAP), encoding="utf-8")
        krm_path = tmp_path / "krm_competency_mapping.json"
        krm_path.write_text(json.dumps(SAMPLE_KRM), encoding="utf-8")
        return tax_path, dom_path, krm_path

    @pytest.fixture
    def pt(self, taxonomy_files):
        tax_path, dom_path, krm_path = taxonomy_files
        return ProfessionTaxonomy(
            taxonomy_path=tax_path,
            domain_map_path=dom_path,
            krm_mapping_path=krm_path,
        )

    # ----------------------------------------------------------------
    # INIT & LOADING
    # ----------------------------------------------------------------
    def test_init_with_real_files(self, pt):
        assert pt._taxonomy is not None
        assert pt._domain_map is not None
        assert pt._krm_mapping is not None
        assert "Data Scientist" in pt.professions
        assert "Python Developer" in pt.professions

    def test_init_with_default_paths(self, monkeypatch):
        monkeypatch.setattr(
            "src.config.PROFESSION_TAXONOMY_PATH",
            type("Path", (), {"exists": lambda self: False, "__str__": lambda self: "/fake"})(),
        )
        pt = ProfessionTaxonomy()
        assert pt._taxonomy == {"professions": {}, "profile_targets": {}}
        assert pt.professions == []

    def test_init_missing_taxonomy_file(self, tmp_path):
        pt = ProfessionTaxonomy(
            taxonomy_path=tmp_path / "nonexistent.json",
            domain_map_path=tmp_path / "nonexistent.json",
            krm_mapping_path=tmp_path / "nonexistent.json",
        )
        assert pt._taxonomy == {"professions": {}, "profile_targets": {}}
        assert pt._domain_map == {}
        assert pt._krm_mapping == {}
        assert pt.professions == []

    def test_init_missing_domain_map_only(self, tmp_path, taxonomy_files):
        tax_path, _, krm_path = taxonomy_files
        pt = ProfessionTaxonomy(
            taxonomy_path=tax_path,
            domain_map_path=tmp_path / "missing.json",
            krm_mapping_path=krm_path,
        )
        assert pt._taxonomy is not None
        assert pt._domain_map == {}
        assert pt.get_domain_skills("Backend") == []

    def test_init_missing_krm_only(self, tmp_path, taxonomy_files):
        tax_path, dom_path, _ = taxonomy_files
        pt = ProfessionTaxonomy(
            taxonomy_path=tax_path,
            domain_map_path=dom_path,
            krm_mapping_path=tmp_path / "missing.json",
        )
        assert pt._krm_mapping == {}
        assert pt.get_competency_skills("ППК-Р1") == []

    # ----------------------------------------------------------------
    # professions property
    # ----------------------------------------------------------------
    def test_professions_list(self, pt):
        profs = pt.professions
        assert "Data Scientist" in profs
        assert "Python Developer" in profs
        assert "Fullstack Developer" in profs
        assert len(profs) == 3

    def test_professions_empty_when_no_data(self, tmp_path):
        pt = ProfessionTaxonomy(
            taxonomy_path=tmp_path / "nope.json",
            domain_map_path=tmp_path / "nope.json",
            krm_mapping_path=tmp_path / "nope.json",
        )
        assert pt.professions == []

    # ----------------------------------------------------------------
    # get_profession_info
    # ----------------------------------------------------------------
    def test_get_profession_info_exists(self, pt):
        info = pt.get_profession_info("Python Developer")
        assert info is not None
        assert info["domains"] == ["Backend"]
        assert info["competency_codes"] == ["ППК-Р1", "ППК-Р2", "ППК-Р3"]

    def test_get_profession_info_not_exists(self, pt):
        assert pt.get_profession_info("NonExistent") is None

    def test_get_profession_info_empty_string(self, pt):
        assert pt.get_profession_info("") is None

    # ----------------------------------------------------------------
    # get_domains_for_profession
    # ----------------------------------------------------------------
    def test_get_domains_for_profession(self, pt):
        domains = pt.get_domains_for_profession("Fullstack Developer")
        assert domains == ["Backend", "Frontend"]

    def test_get_domains_for_profession_not_found(self, pt):
        assert pt.get_domains_for_profession("Ghost") == []

    def test_get_domains_for_profession_empty(self, pt):
        assert pt.get_domains_for_profession("") == []

    # ----------------------------------------------------------------
    # get_domain_skills
    # ----------------------------------------------------------------
    def test_get_domain_skills(self, pt):
        skills = pt.get_domain_skills("Backend")
        assert len(skills) == 4
        assert "python" in skills
        assert "sql" in skills

    def test_get_domain_skills_lowercased(self, pt):
        skills = pt.get_domain_skills("Backend")
        assert all(s == s.lower().strip() for s in skills)

    def test_get_domain_skills_unknown(self, pt):
        assert pt.get_domain_skills("Ghost") == []

    def test_get_domain_skills_when_map_empty(self, tmp_path):
        pt = ProfessionTaxonomy(
            taxonomy_path=tmp_path / "nope.json",
            domain_map_path=tmp_path / "nope.json",
            krm_mapping_path=tmp_path / "nope.json",
        )
        assert pt.get_domain_skills("Backend") == []

    # ----------------------------------------------------------------
    # get_profession_skills
    # ----------------------------------------------------------------
    def test_get_profession_skills_single_domain(self, pt):
        skills = pt.get_profession_skills("Python Developer")
        assert skills == {"python", "sql", "docker", "git"}

    def test_get_profession_skills_multi_domain(self, pt):
        skills = pt.get_profession_skills("Fullstack Developer")
        assert skills == {"python", "sql", "docker", "git", "javascript", "html", "css"}

    def test_get_profession_skills_not_found(self, pt):
        assert pt.get_profession_skills("Ghost") == set()

    # ----------------------------------------------------------------
    # get_profile_target
    # ----------------------------------------------------------------
    def test_get_profile_target_exists(self, pt):
        target = pt.get_profile_target("base")
        assert target is not None
        assert target["target_profession"] == "Data Scientist"
        assert target["target_domains"] == ["Data Science"]

    def test_get_profile_target_not_found(self, pt):
        assert pt.get_profile_target("nonexistent") is None

    # ----------------------------------------------------------------
    # get_profession_competency_codes
    # ----------------------------------------------------------------
    def test_get_profession_competency_codes(self, pt):
        codes = pt.get_profession_competency_codes("Python Developer")
        assert codes == ["ППК-Р1", "ППК-Р2", "ППК-Р3"]

    def test_get_profession_competency_codes_empty(self, pt):
        assert pt.get_profession_competency_codes("Data Scientist") == []

    def test_get_profession_competency_codes_not_found(self, pt):
        assert pt.get_profession_competency_codes("Ghost") == []

    # ----------------------------------------------------------------
    # get_competency_skills
    # ----------------------------------------------------------------
    def test_get_competency_skills(self, pt):
        skills = pt.get_competency_skills("ППК-Р1")
        assert "python" in skills
        assert "programming" in skills

    def test_get_competency_skills_not_found(self, pt):
        assert pt.get_competency_skills("NONEXISTENT") == []

    def test_get_competency_skills_when_krm_empty(self, tmp_path):
        pt = ProfessionTaxonomy(
            taxonomy_path=tmp_path / "nope.json",
            domain_map_path=tmp_path / "nope.json",
            krm_mapping_path=tmp_path / "nope.json",
        )
        assert pt.get_competency_skills("ППК-Р1") == []

    # ----------------------------------------------------------------
    # get_profession_competency_skills
    # ----------------------------------------------------------------
    def test_get_profession_competency_skills(self, pt):
        skills = pt.get_profession_competency_skills("Python Developer")
        expected = {"programming", "coding", "python", "java",
                    "testing", "debugging", "pytest", "code review",
                    "git", "github", "version control"}
        assert skills == expected

    def test_get_profession_competency_skills_no_codes(self, pt):
        assert pt.get_profession_competency_skills("Data Scientist") == set()

    def test_get_profession_competency_skills_not_found(self, pt):
        assert pt.get_profession_competency_skills("Ghost") == set()

    # ----------------------------------------------------------------
    # compute_krm_coverage
    # ----------------------------------------------------------------
    def test_compute_krm_coverage_full_match(self, pt):
        result = pt.compute_krm_coverage(
            "Python Developer",
            ["programming", "coding", "python", "java",
             "testing", "debugging", "pytest", "code review",
             "git", "github", "version control"],
        )
        for code, data in result.items():
            assert data["matched"] == data["required"]
            assert data["coverage"] == 1.0

    def test_compute_krm_coverage_partial(self, pt):
        result = pt.compute_krm_coverage(
            "Python Developer",
            ["python", "git"],
        )
        assert result["ППК-Р1"]["matched"] == 1
        assert result["ППК-Р1"]["required"] == 4
        assert result["ППК-Р1"]["coverage"] == 0.25
        assert result["ППК-Р2"]["matched"] == 0
        assert result["ППК-Р3"]["matched"] == 1

    def test_compute_krm_coverage_no_match(self, pt):
        result = pt.compute_krm_coverage("Python Developer", ["xyz"])
        assert result["ППК-Р1"]["matched"] == 0
        assert result["ППК-Р1"]["coverage"] == 0.0

    def test_compute_krm_coverage_empty_user_skills(self, pt):
        result = pt.compute_krm_coverage("Python Developer", [])
        assert all(v["matched"] == 0 for v in result.values())

    def test_compute_krm_coverage_no_competency_codes(self, pt):
        result = pt.compute_krm_coverage("Data Scientist", ["python"])
        assert result == {}

    def test_compute_krm_coverage_not_found(self, pt):
        assert pt.compute_krm_coverage("Ghost", ["python"]) == {}

    def test_compute_krm_coverage_user_skills_case_insensitive(self, pt):
        result = pt.compute_krm_coverage("Python Developer", ["Python", "Git"])
        assert result["ППК-Р1"]["matched"] == 1
        assert result["ППК-Р3"]["matched"] == 1

    # ----------------------------------------------------------------
    # compute_domain_coverage_for_profession
    # ----------------------------------------------------------------
    def test_compute_domain_coverage_single_domain_full(self, pt):
        result = pt.compute_domain_coverage_for_profession(
            "Data Scientist",
            ["python", "sql", "pandas", "numpy"],
        )
        assert "Data Science" in result
        dm = result["Data Science"]
        assert isinstance(dm, DomainMetrics)
        assert dm.coverage == 1.0
        assert dm.total_required == 4
        assert dm.user_has == 4

    def test_compute_domain_coverage_multi_domain(self, pt):
        result = pt.compute_domain_coverage_for_profession(
            "Fullstack Developer",
            ["python", "javascript", "html"],
        )
        assert "Backend" in result
        assert "Frontend" in result
        assert result["Backend"].coverage == 0.25  # 1/4
        assert result["Frontend"].coverage == 2 / 3

    def test_compute_domain_coverage_empty_user_skills(self, pt):
        result = pt.compute_domain_coverage_for_profession("Python Developer", [])
        assert result["Backend"].coverage == 0.0
        assert result["Backend"].user_has == 0

    def test_compute_domain_coverage_not_found(self, pt):
        assert pt.compute_domain_coverage_for_profession("Ghost", ["python"]) == {}

    def test_compute_domain_coverage_case_insensitive(self, pt):
        result = pt.compute_domain_coverage_for_profession(
            "Python Developer",
            ["Python", "SQL", "DOCKER", "GIT"],
        )
        assert result["Backend"].coverage == 1.0

    # ----------------------------------------------------------------
    # compute_weighted_profession_score
    # ----------------------------------------------------------------
    def test_weighted_score_equal_weights(self, pt):
        score, per_domain = pt.compute_weighted_profession_score(
            "Data Scientist",
            ["python", "sql", "pandas", "numpy"],
        )
        assert score == 100.0
        assert per_domain["Data Science"] == 1.0

    def test_weighted_score_partial_equal_weights(self, pt):
        score, per_domain = pt.compute_weighted_profession_score(
            "Fullstack Developer",
            ["python", "javascript"],
        )
        # Backend: 1/4 = 0.25, Frontend: 1/3 ~ 0.3333
        # equal weight: (0.25*0.5 + 0.3333*0.5) / 1.0 * 100
        expected = (0.25 * 0.5 + (1 / 3) * 0.5) * 100
        assert score == pytest.approx(expected, rel=1e-3)
        assert set(per_domain.keys()) == {"Backend", "Frontend"}

    def test_weighted_score_custom_weights(self, pt):
        score, per_domain = pt.compute_weighted_profession_score(
            "Fullstack Developer",
            ["python", "javascript"],
            domain_weights={"Backend": 0.8, "Frontend": 0.2},
        )
        # Backend: 1/4 = 0.25 * 0.8, Frontend: 1/3 ~ 0.3333 * 0.2
        # total_weight = 1.0
        expected = (0.25 * 0.8 + (1 / 3) * 0.2) * 100
        assert score == pytest.approx(expected, rel=1e-3)

    def test_weighted_score_zero_weight(self, pt):
        score, per_domain = pt.compute_weighted_profession_score(
            "Fullstack Developer",
            ["python", "javascript"],
            domain_weights={"Backend": 0.0, "Frontend": 0.0},
        )
        assert score == 0.0

    def test_weighted_score_not_found(self, pt):
        score, per_domain = pt.compute_weighted_profession_score("Ghost", ["python"])
        assert score == 0.0
        assert per_domain == {}

    def test_weighted_score_empty_user_skills(self, pt):
        score, per_domain = pt.compute_weighted_profession_score("Python Developer", [])
        assert score == 0.0
        assert per_domain["Backend"] == 0.0

    # ----------------------------------------------------------------
    # EDGE: domain_weights None (equal weights path)
    # ----------------------------------------------------------------
    def test_weighted_score_fallback_equal_when_weights_none(self, pt):
        score, per_domain = pt.compute_weighted_profession_score(
            "Fullstack Developer",
            ["python", "sql", "docker", "git", "javascript", "html", "css"],
            domain_weights=None,
        )
        assert score == 100.0

    # ----------------------------------------------------------------
    # EDGE: compute_krm_coverage with empty required skills
    # ----------------------------------------------------------------
    def test_krm_coverage_empty_required_skills(self, pt):
        result = pt.compute_krm_coverage("Data Scientist", ["anything"])
        assert result == {}

    def test_krm_coverage_code_with_no_skills(self, tmp_path):
        tax = {
            "professions": {
                "Test Prof": {
                    "domains": [],
                    "competency_codes": ["MISSING-CODE"],
                },
            },
            "profile_targets": {},
        }
        krm = {}
        tax_path = tmp_path / "tax.json"
        tax_path.write_text(json.dumps(tax), encoding="utf-8")
        dom_path = tmp_path / "dom.json"
        dom_path.write_text(json.dumps({}), encoding="utf-8")
        krm_path = tmp_path / "krm.json"
        krm_path.write_text(json.dumps(krm), encoding="utf-8")
        pt = ProfessionTaxonomy(
            taxonomy_path=tax_path,
            domain_map_path=dom_path,
            krm_mapping_path=krm_path,
        )
        result = pt.compute_krm_coverage("Test Prof", ["anything"])
        assert result == {"MISSING-CODE": {"required": 0, "matched": 0, "coverage": 0.0}}

    # ----------------------------------------------------------------
    # EDGE: domain_map empty
    # ----------------------------------------------------------------
    def test_get_domain_skills_empty_map(self, tmp_path):
        tax_path = tmp_path / "tax.json"
        tax_path.write_text(json.dumps(SAMPLE_TAXONOMY), encoding="utf-8")
        dom_path = tmp_path / "empty_domain.json"
        dom_path.write_text(json.dumps({}), encoding="utf-8")
        krm_path = tmp_path / "krm.json"
        krm_path.write_text(json.dumps(SAMPLE_KRM), encoding="utf-8")
        pt = ProfessionTaxonomy(
            taxonomy_path=tax_path,
            domain_map_path=dom_path,
            krm_mapping_path=krm_path,
        )
        assert pt.get_domain_skills("Backend") == []

    # ----------------------------------------------------------------
    # EDGE: json decode error (invalid file)
    # ----------------------------------------------------------------
    def test_init_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{invalid", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            ProfessionTaxonomy(
                taxonomy_path=p,
                domain_map_path=tmp_path / "nope.json",
                krm_mapping_path=tmp_path / "nope.json",
            )


# ========================================================================
# TaxonomyValidator
# ========================================================================

class TestTaxonomyValidator:

    def test_validate_profession_taxonomy_ok(self):
        data = {
            "professions": {
                "Dev": {"domains": ["Backend"], "competency_codes": []},
            },
            "profile_targets": {},
        }
        issues = TaxonomyValidator.validate_profession_taxonomy(data)
        assert issues == []

    def test_validate_profession_taxonomy_bad_type(self):
        issues = TaxonomyValidator.validate_profession_taxonomy("not a dict")
        assert len(issues) == 1

    def test_validate_domain_map_ok(self):
        issues = TaxonomyValidator.validate_domain_map({"Backend": ["py"]}, {"Backend"})
        assert issues == []

    def test_validate_domain_map_not_dict(self):
        issues = TaxonomyValidator.validate_domain_map("bad", set())
        assert len(issues) == 1

    def test_validate_domain_map_skills_not_list(self):
        issues = TaxonomyValidator.validate_domain_map({"Backend": "not a list"}, {"Backend"})
        assert len(issues) == 1

    def test_validate_krm_mapping_ok(self):
        issues = TaxonomyValidator.validate_krm_mapping({"ППК-Р1": ["py"]}, {"ППК-Р1"})
        assert issues == []

    def test_validate_krm_mapping_missing_code(self):
        issues = TaxonomyValidator.validate_krm_mapping({}, {"ППК-Р1"})
        assert any("ППК-Р1" in i for i in issues)

    def test_validate_cross_references_missing_domain(self):
        tax = {"professions": {"Dev": {"domains": ["GhostDomain"], "competency_codes": []}}}
        issues = TaxonomyValidator.validate_cross_references(tax, {}, {})
        assert any("GhostDomain" in i for i in issues)

    def test_validate_cross_references_missing_krm_code(self):
        tax = {"professions": {"Dev": {"domains": [], "competency_codes": ["MISSING"]}}}
        issues = TaxonomyValidator.validate_cross_references(tax, {}, {})
        assert any("MISSING" in i for i in issues)

    def test_validate_cross_references_clean(self):
        tax = {"professions": {"Dev": {"domains": ["Backend"], "competency_codes": ["ППК-Р1"]}}}
        issues = TaxonomyValidator.validate_cross_references(
            tax, {"Backend": ["py"]}, {"ППК-Р1": ["py"]}
        )
        assert issues == []
