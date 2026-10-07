"""P-length: 2-символьные whitelist-навыки минуют TOO_SHORT; 1-символьные — нет."""

from src.parsing.skills.skill_validator import SkillValidator


class TestWhitelistShortSkills:
    def test_qa_valid(self):
        assert SkillValidator().validate("qa", 1.0).unwrap().is_valid

    def test_go_ui_ux_1c_valid(self):
        v = SkillValidator()
        for s in ("go", "ui", "ux", "1c"):
            assert v.validate(s, 1.0).unwrap().is_valid, s

    def test_single_char_still_rejected(self):
        v = SkillValidator()
        assert not v.validate("r", 1.0).unwrap().is_valid
        assert not v.validate("x", 1.0).unwrap().is_valid

    def test_nonwhitelist_short_still_rejected(self):
        assert not SkillValidator().validate("zz", 1.0).unwrap().is_valid
