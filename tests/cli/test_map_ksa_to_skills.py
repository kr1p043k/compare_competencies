"""Tests for map_ksa_to_skills CLI."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


class TestMapKsaToSkills:
    """Test the 3-tier KSA→it_skills mapping."""

    def test_load_json(self, tmp_path):
        from src.cli.map_ksa_to_skills import load_json
        data = {"key": "value"}
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = load_json(f)
        assert result == data

    def test_load_json_missing(self, tmp_path):
        from src.cli.map_ksa_to_skills import load_json
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")

    @pytest.mark.asyncio
    async def test_tier_explicit_json_missing_file(self, tmp_path):
        from src.cli.map_ksa_to_skills import tier_explicit_json
        session = AsyncMock()
        result = await tier_explicit_json(session, tmp_path / "missing.json", {}, {}, {})
        assert result == 0

    @pytest.mark.asyncio
    async def test_tier_explicit_json_creates_links(self, tmp_path):
        from src.cli.map_ksa_to_skills import tier_explicit_json
        from src.models.krm_models import CompetencySkill

        # Create JSON mapping
        mapping = {"Discipline1": {"PK-1": ["python", "sql"]}}
        f = tmp_path / "map.json"
        f.write_text(json.dumps(mapping), encoding="utf-8")

        disc_id = "disc-1"
        comp_id = "comp-1"
        disc_map = {"Discipline1": disc_id}
        comp_map = {(disc_id, "PK-1"): comp_id}
        skill_map = {"python": "sk-1", "sql": "sk-2"}

        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # No existing link
        session.execute = AsyncMock(return_value=mock_result)

        count = await tier_explicit_json(session, f, disc_map, comp_map, skill_map)
        assert count == 2
        assert session.add.call_count == 2

    @pytest.mark.asyncio
    async def test_tier_substring_finds_matches(self):
        from src.cli.map_ksa_to_skills import tier_substring
        from src.models.krm_models import Skill

        krm = {"09.03.02": {"disciplines": {
            "TestDisc": {"ksa": {"PK-1": {"knowledge": ["используя python для анализа"]}}}
        }}}
        disc_id = "disc-1"
        comp_id = "comp-1"
        disc_map = {"TestDisc": disc_id}
        comp_map = {(disc_id, "PK-1"): comp_id}

        mock_skill = MagicMock()
        mock_skill.name = "python"
        mock_skill.id = "sk-1"

        session = AsyncMock()
        # First call returns skills, second returns no existing link
        call_count = [0]
        async def mock_execute(query, *args):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.scalars.return_value.all.return_value = [mock_skill]
            else:
                result.scalar_one_or_none.return_value = None
            return result
        session.execute = mock_execute

        count = await tier_substring(session, krm, disc_map, comp_map)
        assert count >= 1

    @pytest.mark.asyncio
    async def test_tier_semantic_unavailable(self):
        from src.cli.map_ksa_to_skills import tier_semantic
        krm = {"09.03.02": {"disciplines": {}}}
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=mock_result)
        count = await tier_semantic(session, krm, {}, {})
        assert count == 0

    @pytest.mark.asyncio
    async def test_run_mapping_no_direction(self):
        from src.cli.map_ksa_to_skills import run_mapping
        from src.models.krm_models import Direction

        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)

        with patch("src.cli.map_ksa_to_skills.async_session_factory") as mock_factory:
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await run_mapping()
            assert result == {}
