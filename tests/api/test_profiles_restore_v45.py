"""Unit (v45): custom profiles restore on startup scan."""
from src.api_pkg import deps as _deps
from src.api_pkg.startup import register_custom_student_profiles


def _map(codes):
    return [c + "-skill" for c in codes]


def test_restore_custom(tmp_path):
    import json
    sd = tmp_path / 'students'
    sd.mkdir()
    (sd / 'my_ds_competency.json').write_text(json.dumps(
        {"компетенции": ["UK-1"], "навыки": [], "target_level": "senior"}), encoding="utf-8")
    (sd / 'en_ds_competency.json').write_text(json.dumps(
        {"competencies": ["UK-2"], "skills": ["sql"]}), encoding="utf-8")
    (sd / 'bad_competency.json').write_text("{not json", encoding="utf-8")
    (sd / 'base_competency.json').write_text(json.dumps({"codes": ["UK-9"]}), encoding="utf-8")
    already = {'base': object()}
    got = register_custom_student_profiles(sd, already, _map)
    assert got == ['en_ds', 'my_ds']
    assert already['my_ds'].target_level == 'senior'
    assert already['my_ds'].skills == ['UK-1-skill']
    assert already['base'] is not None and len(already) == 3


def test_restore_empty_dir(tmp_path):
    assert register_custom_student_profiles(tmp_path / 'nope', {}, _map) == []
