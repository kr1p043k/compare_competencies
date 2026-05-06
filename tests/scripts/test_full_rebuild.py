# tests/scripts/test_full_rebuild.py
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestFullRebuild:
    def test_import_module(self):
        """Модуль импортируется"""
        import scripts.full_rebuild
        assert hasattr(scripts.full_rebuild, 'BASE')
        assert hasattr(scripts.full_rebuild, 'to_remove')

    def test_base_path_is_correct(self):
        """BASE указывает на корень проекта"""
        import scripts.full_rebuild
        base = scripts.full_rebuild.BASE
        assert base.exists()
        assert (base / "src").exists()

    def test_to_remove_contains_expected_files(self):
        """Список удаляемых файлов содержит ключевые файлы"""
        import scripts.full_rebuild
        to_remove = scripts.full_rebuild.to_remove
        paths = [str(p) for p in to_remove]

        assert any("processed" in p for p in paths)
        assert any("models" in p for p in paths)
        assert any("embeddings" in p for p in paths)

    @patch("subprocess.run")
    @patch("shutil.rmtree")
    def test_full_rebuild_execution(self, mock_rmtree, mock_run, tmp_path, monkeypatch):
        """Полный цикл перестроения (все шаги выполняются)"""
        mock_run.return_value = MagicMock(returncode=0)

        # Подменяем BASE на временную директорию
        monkeypatch.setattr("scripts.full_rebuild.BASE", tmp_path)
        monkeypatch.setattr("scripts.full_rebuild.DATA", tmp_path / "data")

        # Создаём файлы для удаления
        processed = tmp_path / "data" / "processed"
        processed.mkdir(parents=True)
        (processed / "parsed_skills.pkl").touch()
        (processed / "skill_weights.json").touch()

        models = tmp_path / "data" / "models"
        models.mkdir(parents=True)
        (models / "ltr_ranker_xgb_regressor.joblib").touch()

        cache = tmp_path / "data" / "embeddings" / "cache"
        cache.mkdir(parents=True)
        (cache / "skill_embeddings.json").touch()

        # Выполняем скрипт
        import scripts.full_rebuild as fr

        # Проверяем, что файлы удалены
        assert not (processed / "parsed_skills.pkl").exists()
        assert not (models / "ltr_ranker_xgb_regressor.joblib").exists()
        assert not cache.exists()

        # Проверяем, что subprocess вызван для каждой команды
        assert mock_run.call_count == 4

    @patch("subprocess.run")
    @patch("shutil.rmtree")
    def test_full_rebuild_command_failure(self, mock_rmtree, mock_run, tmp_path, monkeypatch):
        """Ошибка при выполнении команды"""
        monkeypatch.setattr("scripts.full_rebuild.BASE", tmp_path)
        monkeypatch.setattr("scripts.full_rebuild.DATA", tmp_path / "data")

        # Создаём пустые директории
        (tmp_path / "data" / "processed").mkdir(parents=True)
        (tmp_path / "data" / "embeddings" / "cache").mkdir(parents=True)

        mock_run.return_value = MagicMock(returncode=1)

        with pytest.raises(SystemExit):
            import scripts.full_rebuild

    def test_commands_list(self):
        """Список команд содержит ожидаемые скрипты"""
        import scripts.full_rebuild
        commands = scripts.full_rebuild.commands

        command_strs = [" ".join(cmd) for cmd in commands]
        assert any("main.py" in s for s in command_strs)
        assert any("train_clusters.py" in s for s in command_strs)
