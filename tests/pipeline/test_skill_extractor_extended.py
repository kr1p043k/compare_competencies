from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

from src import Ok, Err


@pytest.fixture
def args():
    mock = MagicMock()
    mock.no_filter = False
    return mock


@pytest.fixture
def parser():
    return MagicMock()


@pytest.fixture
def vacancies():
    return [{"id": "1", "name": "Python Dev"}]


@pytest.fixture
def extractor(args):
    from src.pipeline.skill_extractor import SkillExtractor
    return SkillExtractor(args)


def _make_mock_config(cache_path):
    cfg = MagicMock()
    cfg.PARSED_SKILLS_CACHE_PATH = cache_path
    return cfg


class TestSkillExtractorExtract:
    def test_extract_cache_hit(self, extractor, vacancies, parser, tmp_path):
        cache_file = tmp_path / "cache" / "parsed_skills_cache.json"
        cache_file.parent.mkdir(parents=True)
        cache_cfg = _make_mock_config(cache_file)

        with patch("src.pipeline.skill_extractor.config", cache_cfg):
            with patch("src.pipeline.skill_extractor.CacheManager") as MockCM:
                cm_instance = MagicMock()
                MockCM.return_value = cm_instance
                cm_instance.load.return_value = Ok({
                    "source_hash": "abc123",
                    "result": {
                        "frequencies": {"python": 10},
                        "hybrid_weights": {"python": 0.5},
                    },
                })
                with patch.object(extractor, "_get_file_hash", return_value="abc123"):
                    with patch("src.pipeline.skill_extractor.load_it_skills", return_value=None):
                        with patch("src.pipeline.skill_extractor.TrendAnalyzer") as MockTA:
                            ta_instance = MagicMock()
                            MockTA.return_value = ta_instance
                            with patch("src.pipeline.skill_extractor.print_top_skills"):
                                with patch("src.pipeline.skill_extractor.parser") as mock_parser_mod:
                                    mock_p = MagicMock()
                                    mock_parser_mod.save_processed_frequencies = mock_p
                                    result = extractor.extract(vacancies, parser, raw_file=Path("test.json"))

        assert result.is_ok()
        freqs, weights, ta = result.unwrap()
        assert freqs == {"python": 10}

    def test_extract_cache_miss(self, extractor, vacancies, parser, tmp_path):
        cache_file = tmp_path / "cache" / "parsed_skills_cache.json"
        cache_cfg = _make_mock_config(cache_file)

        with patch("src.pipeline.skill_extractor.config", cache_cfg):
            with patch("src.pipeline.skill_extractor.CacheManager") as MockCM:
                cm_instance = MagicMock()
                MockCM.return_value = cm_instance
                cm_instance.load.return_value = Err("no cache")
                cm_instance.save.return_value = Ok(None)
                parser.extract_skills_from_vacancies.return_value = Ok({
                    "frequencies": {"python": 5},
                    "hybrid_weights": {"python": 0.3},
                })
                with patch.object(extractor, "_get_file_hash", return_value="def456"):
                    with patch("src.pipeline.skill_extractor.load_it_skills", return_value=None):
                        with patch("src.pipeline.skill_extractor.TrendAnalyzer") as MockTA:
                            ta_instance = MagicMock()
                            MockTA.return_value = ta_instance
                            with patch("src.pipeline.skill_extractor.print_top_skills"):
                                with patch("src.pipeline.skill_extractor.ArtifactManifest") as MockManifest:
                                    manifest_instance = MagicMock()
                                    MockManifest.return_value = manifest_instance
                                    manifest_instance.save.return_value = Ok(None)
                                    result = extractor.extract(vacancies, parser, raw_file=Path("test.json"))

        assert result.is_ok()
        cm_instance.save.assert_called_once()

    def test_extract_no_raw_file(self, extractor, vacancies, parser, tmp_path):
        cache_file = tmp_path / "cache" / "parsed_skills_cache.json"
        cache_cfg = _make_mock_config(cache_file)

        with patch("src.pipeline.skill_extractor.config", cache_cfg):
            with patch("src.pipeline.skill_extractor.CacheManager") as MockCM:
                cm_instance = MagicMock()
                MockCM.return_value = cm_instance
                parser.extract_skills_from_vacancies.return_value = Ok({
                    "frequencies": {"go": 3},
                    "hybrid_weights": {},
                })
                with patch("src.pipeline.skill_extractor.load_it_skills", return_value=None):
                    with patch("src.pipeline.skill_extractor.TrendAnalyzer") as MockTA:
                        ta_instance = MagicMock()
                        MockTA.return_value = ta_instance
                        with patch("src.pipeline.skill_extractor.print_top_skills"):
                            with patch("src.pipeline.skill_extractor.ArtifactManifest") as MockManifest:
                                manifest_instance = MagicMock()
                                MockManifest.return_value = manifest_instance
                                manifest_instance.save.return_value = Ok(None)
                                result = extractor.extract(vacancies, parser, raw_file=None)

        assert result.is_ok()

    def test_extract_whitelist_filter_applied(self, extractor, vacancies, parser, tmp_path):
        cache_file = tmp_path / "cache" / "parsed_skills_cache.json"
        cache_cfg = _make_mock_config(cache_file)

        with patch("src.pipeline.skill_extractor.config", cache_cfg):
            with patch("src.pipeline.skill_extractor.CacheManager") as MockCM:
                cm_instance = MagicMock()
                MockCM.return_value = cm_instance
                parser.extract_skills_from_vacancies.return_value = Ok({
                    "frequencies": {"python": 5, "garbage": 2},
                    "hybrid_weights": {},
                })
                with patch("src.pipeline.skill_extractor.load_it_skills", return_value=["python"]):
                    with patch("src.pipeline.skill_extractor.filter_skills_by_whitelist",
                               return_value={"python": 5}) as mock_filter:
                        with patch("src.pipeline.skill_extractor.TrendAnalyzer") as MockTA:
                            ta_instance = MagicMock()
                            MockTA.return_value = ta_instance
                            with patch("src.pipeline.skill_extractor.print_top_skills"):
                                with patch("src.pipeline.skill_extractor.ArtifactManifest") as MockManifest:
                                    manifest_instance = MagicMock()
                                    MockManifest.return_value = manifest_instance
                                    manifest_instance.save.return_value = Ok(None)
                                    result = extractor.extract(vacancies, parser, raw_file=None)

        assert result.is_ok()

    def test_extract_parser_error(self, extractor, vacancies, parser, tmp_path):
        cache_file = tmp_path / "cache" / "parsed_skills_cache.json"
        cache_cfg = _make_mock_config(cache_file)

        with patch("src.pipeline.skill_extractor.config", cache_cfg):
            with patch("src.pipeline.skill_extractor.CacheManager") as MockCM:
                cm_instance = MagicMock()
                MockCM.return_value = cm_instance
                cm_instance.load.return_value = Err("no cache")
                parser.extract_skills_from_vacancies.return_value = Err("parser fail")
                result = extractor.extract(vacancies, parser, raw_file=None)

        assert result.is_err()

    def test_extract_competency_mapping(self, extractor, vacancies, parser, tmp_path):
        cache_file = tmp_path / "cache" / "parsed_skills_cache.json"
        cache_cfg = _make_mock_config(cache_file)
        cache_cfg.DATA_PROCESSED_DIR = tmp_path / "processed"
        cache_cfg.DATA_PROCESSED_DIR.mkdir(parents=True)

        from collections import Counter

        with patch("src.pipeline.skill_extractor.config", cache_cfg):
            with patch("src.pipeline.skill_extractor.CacheManager") as MockCM:
                cm_instance = MagicMock()
                MockCM.return_value = cm_instance
                parser.extract_skills_from_vacancies.return_value = Ok({
                    "frequencies": {"python": 5, "JAVA": 3},
                    "hybrid_weights": {},
                })
                with patch("src.pipeline.skill_extractor.load_it_skills", return_value=None):
                    with patch("src.pipeline.skill_extractor.load_competency_mapping",
                               return_value={"python": ["PY"]}):
                        with patch("src.pipeline.skill_extractor.TrendAnalyzer") as MockTA:
                            ta_instance = MagicMock()
                            MockTA.return_value = ta_instance
                            with patch("src.pipeline.skill_extractor.print_top_skills"):
                                with patch("src.pipeline.skill_extractor.print_top_competencies"):
                                    with patch("src.pipeline.skill_extractor.ArtifactManifest") as MockManifest:
                                        manifest_instance = MagicMock()
                                        MockManifest.return_value = manifest_instance
                                        manifest_instance.save.return_value = Ok(None)
                                        with patch("src.pipeline.skill_extractor.map_to_competencies",
                                                   return_value=Counter({"python": 5})):
                                            result = extractor.extract(vacancies, parser, raw_file=None)
        assert result.is_ok()

    def test_extract_competency_mapping_generic_filtered(self, extractor, vacancies, parser, tmp_path):
        cache_file = tmp_path / "cache" / "parsed_skills_cache.json"
        cache_cfg = _make_mock_config(cache_file)
        cache_cfg.DATA_PROCESSED_DIR = tmp_path / "processed"
        cache_cfg.DATA_PROCESSED_DIR.mkdir(parents=True)

        from collections import Counter

        with patch("src.pipeline.skill_extractor.config", cache_cfg):
            with patch("src.pipeline.skill_extractor.CacheManager") as MockCM:
                cm_instance = MagicMock()
                MockCM.return_value = cm_instance
                parser.extract_skills_from_vacancies.return_value = Ok({
                    "frequencies": {"python": 5},
                    "hybrid_weights": {},
                })
                with patch("src.pipeline.skill_extractor.load_it_skills", return_value=None):
                    with patch("src.pipeline.skill_extractor.load_competency_mapping",
                               return_value={"python": ["PY"]}):
                        with patch("src.pipeline.skill_extractor.TrendAnalyzer") as MockTA:
                            ta_instance = MagicMock()
                            MockTA.return_value = ta_instance
                            with patch("src.pipeline.skill_extractor.print_top_skills"):
                                with patch("src.pipeline.skill_extractor.print_top_competencies"):
                                    with patch("src.pipeline.skill_extractor.ArtifactManifest") as MockManifest:
                                        manifest_instance = MagicMock()
                                        MockManifest.return_value = manifest_instance
                                        manifest_instance.save.return_value = Ok(None)
                                        with patch("src.pipeline.skill_extractor.map_to_competencies",
                                                   return_value=Counter({"skill": 5})):
                                            result = extractor.extract(vacancies, parser, raw_file=None)
        assert result.is_ok()

    def test_extract_competency_mapping_error(self, extractor, vacancies, parser, tmp_path):
        cache_file = tmp_path / "cache" / "parsed_skills_cache.json"
        cache_cfg = _make_mock_config(cache_file)

        with patch("src.pipeline.skill_extractor.config", cache_cfg):
            with patch("src.pipeline.skill_extractor.CacheManager") as MockCM:
                cm_instance = MagicMock()
                MockCM.return_value = cm_instance
                parser.extract_skills_from_vacancies.return_value = Ok({
                    "frequencies": {"python": 5},
                    "hybrid_weights": {},
                })
                with patch("src.pipeline.skill_extractor.load_it_skills", return_value=None):
                    with patch("src.pipeline.skill_extractor.TrendAnalyzer") as MockTA:
                        ta_instance = MagicMock()
                        MockTA.return_value = ta_instance
                        with patch("src.pipeline.skill_extractor.print_top_skills"):
                            with patch("src.pipeline.skill_extractor.load_competency_mapping",
                                       side_effect=ValueError("mapping error")):
                                with patch("src.pipeline.skill_extractor.ArtifactManifest") as MockManifest:
                                    manifest_instance = MagicMock()
                                    MockManifest.return_value = manifest_instance
                                    manifest_instance.save.return_value = Ok(None)
                                    result = extractor.extract(vacancies, parser, raw_file=None)
        assert result.is_ok()

    def test_extract_top_level_exception(self, extractor, vacancies, parser, tmp_path):
        cache_file = tmp_path / "cache" / "parsed_skills_cache.json"
        cache_cfg = _make_mock_config(cache_file)

        with patch("src.pipeline.skill_extractor.config", cache_cfg):
            with patch("src.pipeline.skill_extractor.CacheManager") as MockCM:
                cm_instance = MagicMock()
                MockCM.return_value = cm_instance
                cm_instance.load.side_effect = ValueError("unexpected boom")
                result = extractor.extract(vacancies, parser, raw_file=Path("test.json"))
        assert result.is_err()

    def test_get_file_hash(self, extractor, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        h = extractor._get_file_hash(f)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_check_manifest_compatible(self, extractor, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache_path.write_text("{}", encoding="utf-8")
        with patch("src.pipeline.skill_extractor.ArtifactManifest.load",
                   return_value=Ok(MagicMock(is_compatible=MagicMock(return_value=Ok(True))))):
            extractor._check_manifest(cache_path)

    def test_check_manifest_incompatible(self, extractor, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache_path.write_text("{}", encoding="utf-8")
        manifest_path = cache_path.with_suffix(".manifest.json")
        manifest_path.write_text("{}", encoding="utf-8")
        incompatible = MagicMock()
        from src import Ok; incompatible.is_compatible.return_value = Ok(False)
        with patch("src.pipeline.skill_extractor.ArtifactManifest.load",
                   return_value=Ok(incompatible)):
            extractor._check_manifest(cache_path)
        assert not cache_path.exists()
        assert not manifest_path.exists()

    def test_check_manifest_load_error(self, extractor, tmp_path):
        cache_path = tmp_path / "cache.json"
        with patch("src.pipeline.skill_extractor.ArtifactManifest.load",
                   return_value=Err("load error")):
            extractor._check_manifest(cache_path)

    def test_check_manifest_no_file(self, extractor, tmp_path):
        cache_path = tmp_path / "cache.json"
        extractor._check_manifest(cache_path)

    def test_console_info(self, extractor, capsys):
        extractor._console_info("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_extract_no_filter_arg(self, extractor, vacancies, parser, tmp_path):
        extractor.args.no_filter = True
        cache_file = tmp_path / "cache" / "parsed_skills_cache.json"
        cache_cfg = _make_mock_config(cache_file)

        with patch("src.pipeline.skill_extractor.config", cache_cfg):
            with patch("src.pipeline.skill_extractor.CacheManager") as MockCM:
                cm_instance = MagicMock()
                MockCM.return_value = cm_instance
                cm_instance.load.return_value = Err("no cache")
                parser.extract_skills_from_vacancies.return_value = Ok({
                    "frequencies": {"python": 5},
                    "hybrid_weights": {},
                })
                with patch("src.pipeline.skill_extractor.load_it_skills", return_value=None):
                    with patch("src.pipeline.skill_extractor.TrendAnalyzer") as MockTA:
                        ta_instance = MagicMock()
                        MockTA.return_value = ta_instance
                        with patch("src.pipeline.skill_extractor.print_top_skills"):
                            with patch("src.pipeline.skill_extractor.ArtifactManifest") as MockManifest:
                                manifest_instance = MagicMock()
                                MockManifest.return_value = manifest_instance
                                manifest_instance.save.return_value = Ok(None)
                                result = extractor.extract(vacancies, parser, raw_file=None)
        assert result.is_ok()
