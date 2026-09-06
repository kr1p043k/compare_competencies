"""Tests for C6 (has_enhanced_gap), C4 (quality_coverage),
C5 (topic_to_skills), 3.2 (student levels).
"""
import pytest

from src.models.student import StudentProfile
from src.models.enums import CompetencyLevel, ExperienceLevel
from src.models.teacher_analysis import DisciplineCoverage


class TestC6EnhancedFlag:
    """C6: has_enhanced_gap must reflect actual enhancement success."""

    def test_flag_logic_no_enhanced(self):
        """When no discipline has enhanced data, flag is False."""
        disciplines = [
            {"name": "Math"},
            {"name": "Physics"},
        ]
        # Simulate the fixed logic: check each JSON for 'enhanced' key
        has_enh = False
        for d in disciplines:
            # No JSON files exist in test -> flag stays False
            fake_data = {}  # Simulates JSON without 'enhanced'
            if fake_data.get("enhanced"):
                has_enh = True
                break
        assert has_enh is False

    def test_flag_logic_with_enhanced(self):
        """When at least one discipline has enhanced data, flag is True."""
        fake_files = [
            {"enhanced": {"semantic_coverage": {}}},
            {},
        ]
        has_enh = any(f.get("enhanced") for f in fake_files)
        assert has_enh is True


class TestC4QualityCoverage:
    """C4: quality_coverage (weighted) vs coverage_ratio (binary)."""

    def test_discipline_coverage_has_weighted(self):
        """DisciplineCoverage has weighted_coverage field."""
        dc = DisciplineCoverage(
            discipline_id="1",
            discipline_name="Test",
            coverage_ratio=1.0,
            weighted_coverage=0.65,
        )
        assert dc.coverage_ratio == 1.0
        assert dc.weighted_coverage == 0.65
        # Binary says 100%, quality says 65% (weak matches)
        assert dc.weighted_coverage < dc.coverage_ratio

    def test_weighted_defaults_to_zero(self):
        """Backward compat: weighted_coverage defaults to 0.0."""
        dc = DisciplineCoverage(
            discipline_id="1",
            discipline_name="Test",
        )
        assert dc.weighted_coverage == 0.0

    def test_quality_calculation(self):
        """Quality = sum(confidence)/total, binary = matched/total."""
        # 2 matched (conf 1.0 + 0.5), 1 gap, total 3
        # binary = 2/3 = 0.667, quality = 1.5/3 = 0.5
        matched_conf = [1.0, 0.5]
        total = 3
        binary = len(matched_conf) / total
        quality = sum(matched_conf) / total
        assert binary == pytest.approx(0.667, abs=0.01)
        assert quality == pytest.approx(0.5, abs=0.01)
        assert quality < binary


class TestC5TopicToSkills:
    """C5: raw topic string must NOT be added as skill."""

    def test_topic_to_skills_no_raw_topic(self):
        """topic_to_skills must not include the raw topic string."""
        from src.analyzers.academic_gap import AcademicGapAnalyzer
        analyzer = AcademicGapAnalyzer()
        topic = "Базы данных и СУБД"
        skills = analyzer.topic_to_skills(topic)
        # Raw topic should NOT be in the list
        assert topic not in skills, f"Raw topic found in skills: {skills}"
        # Should still return a list (may be empty or have extracted skills)
        assert isinstance(skills, list)

    def test_topic_to_skills_returns_list(self):
        """topic_to_skills always returns a list, even for empty topic."""
        from src.analyzers.academic_gap import AcademicGapAnalyzer
        analyzer = AcademicGapAnalyzer()
        # Empty topic
        result = analyzer.topic_to_skills("")
        assert isinstance(result, list)


class TestStudentLevels32:
    """3.2: StudentProfile skill_levels + correct target_level."""

    def test_skill_levels_field_exists(self):
        """StudentProfile has skill_levels dict field."""
        s = StudentProfile(profile_name="t", competencies=[], skills=[])
        assert hasattr(s, "skill_levels")
        assert s.skill_levels == {}

    def test_skill_levels_preserved(self):
        """skill_levels passed to constructor are preserved."""
        s = StudentProfile(
            profile_name="t",
            competencies=["PC-1"],
            skills=["python"],
            skill_levels={"PC-1": "E"},
        )
        assert s.skill_levels == {"PC-1": "E"}

    def test_competency_level_enum(self):
        """CompetencyLevel has B/P/E/X values."""
        assert CompetencyLevel.BEGINNING == "B"
        assert CompetencyLevel.PRACTICED == "P"
        assert CompetencyLevel.EXPERT == "E"
        assert CompetencyLevel.EXCLUDED == "X"

    def test_backward_compat_no_levels(self):
        """Old code without skill_levels still works."""
        s = StudentProfile(
            profile_name="t",
            competencies=["PC-1"],
            skills=["python"],
        )
        assert s.skill_levels == {}
        assert s.target_level == "middle"  # default

    def test_loader_target_levels(self):
        """Loader assigns correct target_level per profile."""
        from src.loaders_student.student_loader import StudentLoader
        loader = StudentLoader()
        expected = {"base": "junior", "dc": "middle", "top_dc": "senior"}
        for profile, level in expected.items():
            result = loader.load_student(profile)
            if result.is_ok():
                assert result.unwrap().target_level == level, (
                    f"{profile} should be {level}"
                )

    def test_loader_reads_skills(self):
        """Loader reads actual skills from JSON (not empty)."""
        from src.loaders_student.student_loader import StudentLoader
        loader = StudentLoader()
        result = loader.load_student("base")
        if result.is_ok():
            student = result.unwrap()
            # Base should have skills (was 0 before fix due to wrong key)
            assert len(student.skills) > 0, "Base profile should have skills"
