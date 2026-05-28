# tests/scoring/test_vacancy_quality_scorer.py
import pytest
from src import Ok
from src.scoring.vacancy_quality_scorer import VacancyQualityScorer, SpamFlag, QualityScore, _count_urls
from src.models.vacancy import Vacancy, Salary, Snippet, Area, Employer, KeySkill


@pytest.fixture
def scorer():
    return VacancyQualityScorer(min_skills=2, min_description_chars=100, spam_threshold=0.5)


def _make_vac(overrides: dict | None = None) -> Vacancy:
    defaults = dict(
        id='1',
        name='Python Developer',
        area=Area(id=1, name='Moscow'),
        employer=Employer(id='123', name='Tech Corp'),
        salary=None,
        description='Real description ' * 20,
        key_skills=[KeySkill(name='python'), KeySkill(name='sql')],
        snippet=Snippet(requirement=None, responsibility=None),
    )
    if overrides:
        defaults.update(overrides)
    return Vacancy(**defaults)


class TestSpamFlag:
    def test_repr(self):
        f = SpamFlag('NO_DESCRIPTION', 'No description')
        assert 'NO_DESCRIPTION' in repr(f)
        assert 'No description' in repr(f)

    def test_reasons_property(self):
        s = QualityScore('1', 't', 'e', 0.3, True, [SpamFlag('NO_SKILLS', ''), SpamFlag('GENERIC_NAME', '')])
        assert s.reasons == ['NO_SKILLS', 'GENERIC_NAME']

    def test_quality_score_repr_spam(self):
        s = QualityScore('1', 'n', 'e', 0.3, True)
        assert '[SPAM]' in repr(s)

    def test_quality_score_repr_ok(self):
        s = QualityScore('1', 'n', 'e', 0.7, False)
        assert '[OK]' in repr(s)


class TestCountUrls:
    def test_no_urls(self):
        assert _count_urls('hello world') == 0

    def test_single_url(self):
        assert _count_urls('visit https://example.com') == 1

    def test_multiple_urls(self):
        assert _count_urls('a http://a.com b https://b.com/c c https://c.com') == 3

    def test_empty_string(self):
        assert _count_urls('') == 0


class TestVacancyQualityScorer:
    def test_clean_vacancy(self, scorer):
        v = _make_vac()
        s = scorer.score(v).unwrap()
        assert not s.is_spam
        assert s.score >= 0.5

    def test_no_description(self, scorer):
        v = _make_vac({'description': ''})
        s = scorer.score(v).unwrap()
        assert any(f.reason == 'NO_DESCRIPTION' for f in s.flags)

    def test_too_short_description(self, scorer):
        v = _make_vac({'description': 'Short'})
        s = scorer.score(v).unwrap()
        assert any(f.reason == 'TOO_SHORT_DESCRIPTION' for f in s.flags)

    def test_no_skills(self, scorer):
        v = _make_vac({'key_skills': []})
        s = scorer.score(v).unwrap()
        assert any(f.reason == 'нет навыков' for f in s.flags)

    def test_too_few_skills(self, scorer):
        v = _make_vac({'key_skills': [KeySkill(name='python')]})
        s = scorer.score(v).unwrap()
        assert any(f.reason == 'TOO_FEW_KEY_SKILLS' for f in s.flags)

    def test_suspicious_employer(self, scorer):
        v = _make_vac({'employer': Employer(id='999', name='Кадровое агентство')})
        s = scorer.score(v).unwrap()
        assert any(f.reason == 'SUSPICIOUS_EMPLOYER' for f in s.flags)

    def test_generic_name(self, scorer):
        v = _make_vac({'name': 'водитель'})
        s = scorer.score(v).unwrap()
        assert any(f.reason == 'GENERIC_NAME' for f in s.flags)

    def test_generic_name_pattern(self, scorer):
        v = _make_vac({'name': 'вакансия менеджер'})
        s = scorer.score(v).unwrap()
        assert any(f.reason == 'GENERIC_NAME' for f in s.flags)

    def test_promo_description(self, scorer):
        v = _make_vac({'description': 'самая высокая зарплата на рынке!', 'snippet': Snippet()})
        s = scorer.score(v).unwrap()
        assert any(f.reason == 'PROMO_DESCRIPTION' for f in s.flags)

    def test_excessive_urls(self, scorer):
        urls = ' '.join([f'https://site{i}.com' for i in range(5)])
        v = _make_vac({'description': urls, 'snippet': Snippet()})
        s = scorer.score(v).unwrap()
        assert any(f.reason == 'EXCESSIVE_URLS' for f in s.flags)

    def test_salary_anomaly(self, scorer):
        v = _make_vac({'salary': Salary(from_amount=2_000_000, to_amount=3_000_000, currency='RUR')})
        s = scorer.score(v).unwrap()
        assert any(f.reason == 'SALARY_ANOMALY' for f in s.flags)

    def test_score_below_threshold_is_spam(self):
        strict = VacancyQualityScorer(spam_threshold=0.8)
        v = _make_vac({'description': '', 'key_skills': []})
        s = strict.score(v).unwrap()
        assert s.is_spam

    def test_non_spam_salary_ok(self, scorer):
        v = _make_vac({'salary': Salary(from_amount=100_000, to_amount=150_000, currency='RUR')})
        s = scorer.score(v).unwrap()
        assert not any(f.reason == 'SALARY_ANOMALY' for f in s.flags)

    def test_non_spam_urls_ok(self, scorer):
        urls = ' '.join([f'https://site{i}.com' for i in range(2)])
        v = _make_vac({'description': urls, 'snippet': Snippet()})
        s = scorer.score(v).unwrap()
        assert not any(f.reason == 'EXCESSIVE_URLS' for f in s.flags)

    def test_filter_vacancies_all_clean(self, scorer):
        v1 = _make_vac({'id': '1'})
        v2 = _make_vac({'id': '2'})
        clean, spam, report = scorer.filter_vacancies([v1, v2]).unwrap()
        assert len(clean) == 2
        assert len(spam) == 0
        assert report['spam_count'] == 0

    def test_filter_vacancies_mixed(self, scorer):
        clean_v = _make_vac({'id': '1'})
        spam_v = _make_vac({
            'id': '2', 'description': '', 'key_skills': [],
            'name': 'водитель',
            'employer': Employer(id='999', name='Кадровое агентство'),
        })
        clean, spam, report = scorer.filter_vacancies([clean_v, spam_v]).unwrap()
        assert len(clean) == 1
        assert len(spam) == 1
        assert report['spam_count'] == 1
        assert report['spam_rate'] > 0

    def test_build_report_empty(self, scorer):
        report = scorer._build_report([], 0)
        assert report['total_vacancies'] == 0
        assert report['avg_quality_score'] == 1.0

    def test_build_report_with_spam(self, scorer):
        scores = [
            QualityScore('1', 'n', 'e', 0.3, True, [SpamFlag('NO_SKILLS', ''), SpamFlag('GENERIC_NAME', '')]),
            QualityScore('2', 'n', 'e', 0.9, False),
        ]
        report = scorer._build_report(scores, 2)
        assert report['spam_count'] == 1
        assert report['spam_reasons']['NO_SKILLS'] == 1
        assert report['spam_reasons']['GENERIC_NAME'] == 1
        assert len(report['spam_vacancies']) == 1

    def test_print_report(self, scorer, capsys):
        report = {
            'total_vacancies': 10, 'clean_count': 8, 'spam_count': 2, 'spam_rate': 0.2,
            'avg_quality_score': 0.6, 'spam_reasons': {'NO_SKILLS': 1, 'GENERIC_NAME': 1},
            'spam_vacancies': [{'id': '1', 'name': 'Test', 'employer': 'E', 'score': 0.3,
                                'flags': [{'reason': 'NO_SKILLS', 'detail': ''}]}],
        }
        scorer.print_report(report)
        out = capsys.readouterr().out
        assert 'VACANCY QUALITY REPORT' in out
        assert 'NO_SKILLS' in out

    def test_snippet_both_texts_checked(self, scorer):
        v = _make_vac({'description': 'normal job', 'snippet': Snippet(requirement='работа без опыта', responsibility='')})
        s = scorer.score(v).unwrap()
        assert any(f.reason == 'PROMO_DESCRIPTION' for f in s.flags)

    def test_score_never_below_zero(self, scorer):
        v = _make_vac({'description': '', 'key_skills': []})
        s = scorer.score(v).unwrap()
        assert s.score >= 0.0

    def test_snippet_is_none(self, scorer):
        v = _make_vac({'snippet': None})
        s = scorer.score(v).unwrap()
        assert not s.is_spam

    def test_get_midpoint_from_only_from(self, scorer):
        v = _make_vac({'salary': Salary(from_amount=2_000_000, to_amount=None, currency='RUR')})
        s = scorer.score(v).unwrap()
        assert any(f.reason == 'SALARY_ANOMALY' for f in s.flags)

    def test_get_midpoint_from_only_to(self, scorer):
        v = _make_vac({'salary': Salary(from_amount=None, to_amount=50_000, currency='RUR')})
        s = scorer.score(v).unwrap()
        assert not any(f.reason == 'SALARY_ANOMALY' for f in s.flags)
