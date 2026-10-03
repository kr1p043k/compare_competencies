"""Ручные пометки foundational: нормализация, фильтр выдачи, персистентность."""
from src.api_pkg.routers import teacher as t


class TestNormSkill:
    def test_basic(self):
        assert t._norm_skill("  Отладка  ") == "отладка"

    def test_collapse_ws(self):
        assert t._norm_skill("создавать  программный\nкод") == "создавать программный код"

    def test_empty(self):
        assert t._norm_skill("") == ""
        assert t._norm_skill(None) == ""


class TestApplyFilter:
    def _recs(self):
        return [
            {"type": "review_content", "priority": "medium", "message": "m1",
             "skill": "Отлаживать программы на языке С/С++"},
            {"type": "foundational", "priority": "low", "message": "m2",
             "skill": "Синтаксис языка"},
            {"type": "review_content", "priority": "medium", "message": "m3",
             "skill": "Другой навык"},
            {"type": "major_revision", "priority": "high", "message": "m4"},
        ]

    def test_flagged_removed_case_insensitive(self):
        vis, hid = t._apply_foundational_filter(
            self._recs(), ["отлаживать программы на языке с/с++"])
        assert len(vis) == 2
        assert len(hid) == 2
        by_skill = {h["skill"]: h for h in hid}
        assert by_skill["Отлаживать программы на языке С/С++"]["manual"] is True
        assert by_skill["Синтаксис языка"]["manual"] is False

    def test_auto_foundational_hidden_without_flags(self):
        vis, hid = t._apply_foundational_filter(self._recs(), [])
        assert len(vis) == 3 and len(hid) == 1
        assert hid[0]["skill"] == "Синтаксис языка"
        assert hid[0]["manual"] is False

    def test_no_flags_keeps_non_foundational(self):
        recs = [r for r in self._recs() if r["type"] != "foundational"]
        vis, hid = t._apply_foundational_filter(recs, [])
        assert len(vis) == 3 and hid == []

    def test_rec_without_skill_kept(self):
        vis, hid = t._apply_foundational_filter(
            [{"type": "review_content", "message": "m"}], ["x"])
        assert len(vis) == 1 and hid == []


class TestPersistence:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(t, "_foundational_path", lambda: tmp_path / "flags.json")
        assert t._load_foundational() == []
        t._save_foundational(["a", "b", "a"])
        # _save не дедуплицирует, _load — да
        assert t._load_foundational() == ["a", "b"]

    def test_load_legacy_list_format(self, tmp_path, monkeypatch):
        p = tmp_path / "flags.json"
        p.write_text('["x", "X ", ""]', encoding="utf-8")
        monkeypatch.setattr(t, "_foundational_path", lambda: p)
        assert t._load_foundational() == ["x"]

    def test_load_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(t, "_foundational_path", lambda: tmp_path / "nope.json")
        assert t._load_foundational() == []
