# Regression tests: recommendation noise never reaches teachers (v10).
import pytest

from src.analyzers.skill_cooccurrence import SkillCooccurrence
from src.models.teacher_analysis import (
    CompetencyCoverage, CrossReference, DisciplineCoverage, Recommendation, SkillMatch,
)
from src.predictors.curriculum_recommender import CurriculumRecommender, _classify_skill
from src.text import ru_morph


def _rec():
    return CurriculumRecommender()


def test_fragment_shreds_dropped_but_market_tokens_kept():
    r = _rec()
    assert r._is_fragment("локальных и") is True
    assert r._is_fragment("sql") is False
    assert r._is_fragment("erp") is False
    assert r._is_fragment("linux") is False
    assert r._is_fragment("Hot, Label Encoding") is False


def test_knowledge_description_routes_to_academic():
    r = _rec()
    long_knowledge = (
        "отличительных особенностей фундаментальных и пользовательских типов данных"
        " и объектов программ для хранения и обработки информации различного типа"
    )
    assert _classify_skill(long_knowledge, r.skill_types) == "academic"
    assert _classify_skill("управление знаниями", r.skill_types) != "academic"


def test_lemma_key_declension_insensitive():
    if not ru_morph.available():
        pytest.skip("pymorphy3 unavailable")
    assert ru_morph.lemma_key("определение смысла жизни") == ru_morph.lemma_key(
        "определения смысла жизни"
    )


def test_cooc_link_separates_senses():
    co = SkillCooccurrence().build(
        [
            ["индексы", "sql"],
            ["индексы", "sql"],
            ["индексы", "postgresql"],
            ["теория вероятностей", "математический анализ"],
        ]
    )
    assert co.link("индексы", ["sql", "postgresql"]) > 0.3
    assert co.link("индексы", ["теория вероятностей", "математический анализ"]) == 0.0


def _math_coverage():
    co = SkillCooccurrence().build(
        [
            ["индексы", "sql"],
            ["индексы", "sql"],
            ["python", "django"],
        ]
    )
    cov = DisciplineCoverage(
        discipline_id="t1",
        discipline_name="Тестовая дисциплина",
        top_matched=[SkillMatch("python", 3016)],
        gaps_list=["определение смысла жизни", "определения смысла жизни", "локальных и"],
        truly_missing=[SkillMatch("индексы", 1227)],
        cross_references=[CrossReference("r", 1386, "Другая дисциплина")],
        competencies=[],
        coverage_ratio=0.9,
    )
    return co, cov


def test_weaklink_drops_db_sense_for_math_phrases():
    r = _rec()
    co = SkillCooccurrence().build(
        [["индексы", "sql"], ["индексы", "sql"], ["python", "django"]],
        vocab={"индексы", "sql", "python", "django", "теория вероятностей"},
    )
    cov = DisciplineCoverage(
        discipline_id="t1",
        discipline_name="Тестовая дисциплина",
        top_matched=[SkillMatch("теория вероятностей и основы Python", 65)],
        gaps_list=[],
        truly_missing=[SkillMatch("индексы", 1227)],
        cross_references=[],
        competencies=[],
        coverage_ratio=0.9,
    )
    # refs extracted: python + теория вероятностей; link = 0 -> drop via gate
    from src.predictors.curriculum_recommender import _vocab_refs
    refs = _vocab_refs([m.skill_name for m in cov.top_matched], co.vocab)
    assert refs == {"python", "теория вероятностей"}
    assert co.link("индексы", refs) == 0.0
    recs = r.generate(cov, cooc=co).ok()
    assert [x for x in recs if x.type == "add_new_content"] == []


def test_stronglink_keeps_candidate_with_reason():
    r = _rec()
    co = SkillCooccurrence().build(
        [["индексы", "python"]],
        vocab={"индексы", "python", "sql"},
    )
    cov = DisciplineCoverage(
        discipline_id="t1",
        discipline_name="Тестовая дисциплина",
        top_matched=[SkillMatch("работа с Python и Pandas", 100)],
        gaps_list=[],
        truly_missing=[SkillMatch("индексы", 1227)],
        cross_references=[],
        competencies=[],
        coverage_ratio=0.9,
    )
    recs = r.generate(cov, cooc=co).ok()
    adds = [x for x in recs if x.type == "add_new_content"]
    assert len(adds) == 1 and adds[0].skill_name == "индексы"
    assert "рядом" in adds[0].message and "python" in adds[0].message
    assert "смежно" in adds[0].message

def test_reverse_link_rescues_niche_ref():
    # docker huge (forward < 0.03) but niche tcp loyal to it: P(docker|tcp) = 1.0
    sets = [["docker", "kubernetes"]] * 400 + [["tcp", "docker"]] * 6
    co = SkillCooccurrence().build(sets, vocab={"docker", "kubernetes", "tcp", "linux"})
    assert co.link("docker", ["linux", "tcp"]) < 0.03
    assert co.cond("docker", "tcp") == 1.0
    r = _rec()
    cov = DisciplineCoverage(
        discipline_id="t1",
        discipline_name="Test networks",
        top_matched=[SkillMatch("администрирование linux сетей tcp", 50)],
        gaps_list=[],
        truly_missing=[SkillMatch("docker", 500)],
        cross_references=[],
        competencies=[],
        coverage_ratio=0.9,
    )
    recs = r.generate(cov, cooc=co).ok()
    assert [x.skill_name for x in recs if x.type == "add_new_content"] == ["docker"]


def test_reverse_link_zero_still_drops():
    sets = [["docker", "kubernetes"]] * 400 + [["tcp", "git"]] * 6
    co = SkillCooccurrence().build(
        sets, vocab={"docker", "kubernetes", "tcp", "git", "linux"}
    )
    r = _rec()
    cov = DisciplineCoverage(
        discipline_id="t1",
        discipline_name="Test networks",
        top_matched=[SkillMatch("администрирование linux сетей tcp", 50)],
        gaps_list=[],
        truly_missing=[SkillMatch("docker", 500)],
        cross_references=[],
        competencies=[],
        coverage_ratio=0.9,
    )
    recs = r.generate(cov, cooc=co).ok()
    assert [x for x in recs if x.type == "add_new_content"] == []


def test_self_mentioned_candidate_bypasses_gate():
    # discipline names tcp but does not teach it: mentioned-in-passing -> keep
    co = SkillCooccurrence().build(
        [["docker", "kubernetes"]], vocab={"docker", "kubernetes", "tcp", "linux"}
    )
    r = _rec()
    cov = DisciplineCoverage(
        discipline_id="t1",
        discipline_name="Test networks",
        top_matched=[SkillMatch("администрирование linux сетей tcp", 50)],
        gaps_list=[],
        truly_missing=[SkillMatch("tcp", 100)],
        cross_references=[],
        competencies=[],
        coverage_ratio=0.9,
    )
    recs = r.generate(cov, cooc=co).ok()
    assert [x.skill_name for x in recs if x.type == "add_new_content"] == ["tcp"]


def test_single_char_crossref_dropped():
    r = _rec()
    _, cov = _math_coverage()
    recs = r.generate(cov).ok()
    assert [x for x in recs if x.type == "cross_reference"] == []


def test_declension_dupes_folded():
    if not ru_morph.available():
        pytest.skip("pymorphy3 unavailable")
    r = _rec()
    _, cov = _math_coverage()
    recs = r.generate(cov).ok()
    reviews = [x for x in recs if x.type == "review_content"]
    assert len(reviews) == 1
    assert "похожих формулировок" in reviews[0].message


def test_phantom_empty_skill_never_emerges():
    from src.analyzers.skill_matcher import SkillMatcher
    m = SkillMatcher(market_skills={'': 5442, 'sql': 3519, 'linux': 3020})
    res = m.get_emerging(set(), top_n=10).unwrap()
    assert all(s.strip() for s, _, _ in res)
    assert res[0][0] == 'sql'


def test_validator_invariants():
    r = _rec()
    cov = DisciplineCoverage(
        discipline_id="t",
        discipline_name="D",
        top_matched=[SkillMatch("python", 10)],
    )
    recs = [
        Recommendation(type="add_new_content", priority="medium", skill_name="python",
                       message="m1"),
        Recommendation(type="cross_reference", priority="low", skill_name="r", message="m2"),
        Recommendation(type="review_content", priority="medium", skill_name="", message="m3"),
    ]
    out = r.validate(recs, cov)
    assert [x.message for x in out] == ["m3"]
