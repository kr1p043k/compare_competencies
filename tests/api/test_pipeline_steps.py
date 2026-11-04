"""Тесты шагов конвейера (src/pipeline/pipeline_steps.py).

Стиль как в tests/pipeline/test_stages_extended.py: внешние зависимости
(источник hh.ru, scorer, extractor, cleaner, builder, LTR, GapRunner) и
файловые каталоги подменяются моками. Реальная сеть и БД не трогаются.
"""
from unittest.mock import MagicMock, patch

import pytest

from src import Err, Ok
from src.pipeline.pipeline_steps import (
    GapComputeStep,
    LevelAssignStep,
    LTRTrainStep,
    SkillParseStep,
    SpamFilterStep,
    VacancyFetchStep,
    WeightNormalizeStep,
)


@pytest.fixture
def args():
    a = MagicMock()
    a.excel = False
    a.query = "python"
    a.skip_collection = False
    return a


DICT_VACANCY = {
    "id": "1",
    "name": "Python Developer",
    "area": {"id": 1, "name": "Moscow"},
    "employer": {"id": "1", "name": "Company"},
    "raw_data": {},
}


# --------------------------------------------------------------- VacancyFetch

def test_vacancy_fetch_ok(args):
    s = VacancyFetchStep(args)
    with patch("src.pipeline.pipeline_steps.HhDataSource") as mock_source:
        mock_source.return_value.get_vacancies.return_value = Ok(([1, 2], "parser"))
        result = s.run()
    assert result.is_ok()
    data = result.unwrap()
    assert data == {"vacancies": [1, 2], "parser": "parser", "raw_file": None}


def test_vacancy_fetch_skip_collection_uses_find_file(args):
    s = VacancyFetchStep(args)
    args.skip_collection = True
    with patch("src.pipeline.pipeline_steps.HhDataSource") as mock_source:
        source = mock_source.return_value
        source.get_vacancies.return_value = Ok(([], None))
        source._find_file.return_value = "cached.json"
        result = s.run()
    assert result.is_ok()
    assert result.unwrap()["raw_file"] == "cached.json"
    source._find_file.assert_called_once()


def test_vacancy_fetch_err(args):
    s = VacancyFetchStep(args)
    with patch("src.pipeline.pipeline_steps.HhDataSource") as mock_source:
        mock_source.return_value.get_vacancies.return_value = Err(
            MagicMock(message="api error")
        )
        result = s.run()
    assert result.is_err()


# ---------------------------------------------------------------- SpamFilter

def test_spam_filter_ok_marks_spam(args):
    vacancy = MagicMock()
    vacancy.raw_data = {}
    score = MagicMock()
    score.is_spam = True
    score.flags = [MagicMock(reason="spam")]
    s = SpamFilterStep(args)
    with (
        patch("src.pipeline.pipeline_steps.VacancyQualityScorer") as mock_scorer,
        patch("src.pipeline.helpers.save_detailed_vacancies"),
        patch("builtins.open"),
        patch("src.pipeline.pipeline_steps.config") as cfg,
    ):
        mock_scorer.return_value.score.return_value = Ok(score)
        mock_scorer.return_value._build_report.return_value = {"total": 1}
        cfg.REPORTS_DIR.__truediv__.return_value = MagicMock()
        result = s.run(vacancies=[vacancy], parser=MagicMock())
    assert result.is_ok()
    assert result.unwrap() == {"quality_report": {"total": 1}}
    assert vacancy.raw_data["is_spam"] is True


def test_spam_filter_scorer_err(args):
    vacancy = MagicMock()
    vacancy.raw_data = {}
    s = SpamFilterStep(args)
    with patch("src.pipeline.pipeline_steps.VacancyQualityScorer") as mock_scorer:
        mock_scorer.return_value.score.return_value = Err(MagicMock(message="scoring error"))
        result = s.run(vacancies=[vacancy], parser=MagicMock())
    assert result.is_err()


# ----------------------------------------------------------------- SkillParse

def test_skill_parse_ok_stores_extracted_skills(args):
    vacancy = dict(DICT_VACANCY)
    parser = MagicMock()
    parser.skill_parser.parse_vacancy.return_value = Ok([])
    s = SkillParseStep(args)
    with (
        patch("src.pipeline.pipeline_steps.SkillExtractor") as mock_extractor,
        patch("src.pipeline.helpers.save_detailed_vacancies"),
    ):
        mock_extractor.return_value.extract.return_value = Ok(
            ({"sql": 3}, {"sql": 0.5}, MagicMock())
        )
        result = s.run(vacancies=[vacancy], parser=parser, raw_file=None)
    assert result.is_ok()
    data = result.unwrap()
    assert data["skill_freq"] == {"sql": 3}
    assert data["hybrid_weights_raw"] == {"sql": 0.5}
    assert vacancy["extracted_skills"] == []


def test_skill_parse_err(args):
    s = SkillParseStep(args)
    with patch("src.pipeline.pipeline_steps.SkillExtractor") as mock_extractor:
        mock_extractor.return_value.extract.return_value = Err("extraction failed")
        result = s.run(vacancies=[], parser=MagicMock(), raw_file=None)
    assert result.is_err()


# ------------------------------------------------------------- WeightNormalize

def test_weight_normalize_ok():
    s = WeightNormalizeStep()
    with patch("src.pipeline.pipeline_steps.WeightCleaner") as mock_cleaner:
        mock_cleaner.return_value.clean.return_value = Ok({"python": 0.7})
        result = s.run(hybrid_weights_raw={"python": 1.0})
    assert result.is_ok()
    assert result.unwrap() == {"hybrid_weights": {"python": 0.7}}


def test_weight_normalize_err():
    s = WeightNormalizeStep()
    with patch("src.pipeline.pipeline_steps.WeightCleaner") as mock_cleaner:
        mock_cleaner.return_value.clean.return_value = Err("clean failed")
        result = s.run(hybrid_weights_raw={"python": 1.0})
    assert result.is_err()


# --------------------------------------------------------------- LevelAssign

def test_level_assign_ok():
    s = LevelAssignStep()
    with patch("src.pipeline.pipeline_steps.LevelBuilder") as mock_builder:
        mock_builder.return_value.build.return_value = Ok(({"junior": 1}, []))
        result = s.run(vacancies=[], parser=MagicMock())
    assert result.is_ok()
    assert result.unwrap()["level_data"] == {"junior": 1}


# ---------------------------------------------------------------- LTRTrain

def test_ltr_train_no_vacancy_file():
    s = LTRTrainStep()
    with patch("src.pipeline.pipeline_steps.config") as cfg:
        def mock_truediv(_key):
            m = MagicMock()
            m.exists.return_value = False
            return m

        cfg.DATA_PROCESSED_DIR.__truediv__.side_effect = mock_truediv
        cfg.DATA_RAW_DIR.__truediv__.side_effect = mock_truediv
        result = s.run()
    assert result.is_ok()
    assert result.unwrap() == {"model_trained": False, "reason": "no_vacancy_file"}


# --------------------------------------------------------------- GapCompute

def test_gap_compute_ok_and_err(args):
    s = GapComputeStep({"base": {}}, MagicMock(), args)
    with patch("src.pipeline.pipeline_steps.GapRunner") as mock_runner:
        mock_runner.return_value.run.return_value = Ok(({"base": 1}, {}))
        ok_result = s.run()
    assert ok_result.is_ok()
    assert ok_result.unwrap()["evaluations"] == {"base": 1}

    s_err = GapComputeStep({"base": {}}, MagicMock(), args)
    with patch("src.pipeline.pipeline_steps.GapRunner") as mock_runner:
        mock_runner.return_value.run.return_value = Err("gap failed")
        err_result = s_err.run()
    assert err_result.is_err()
