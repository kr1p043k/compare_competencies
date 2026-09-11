# Hash-seed determinism: same logical inputs built via set iteration must
# produce byte-identical outputs under different PYTHONHASHSEED (v15).
import json
import os
import subprocess
import sys

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SNIPPET = "import json, sys\nsys.path.insert(0, '.')\nfrom src.analyzers.skill_matcher import SkillMatcher\nfrom src.models.teacher_analysis import DisciplineCoverage, SkillMatch\nfrom src.predictors.curriculum_recommender import CurriculumRecommender\n# market built from SET iteration: insertion order varies with hash seed\nkeys = {f'skill_{i:02d}' for i in range(30)}\nmarket = {k: (50 if int(k[-2:]) % 3 == 0 else 100) for k in keys}\nm = SkillMatcher(market_skills=market)\nem = m.get_emerging(set(), top_n=10).unwrap()\ncov = DisciplineCoverage(discipline_id='t', discipline_name='D',\n    top_matched=[SkillMatch('python', 10)],\n    gaps_list=['определение смысла жизни', 'определения смысла жизни'],\n    truly_missing=[SkillMatch(n, 100) for n in ['docker','bash','git','linux','javascript','zabbix','terraform','ansible']],\n    cross_references=[], competencies=[], coverage_ratio=0.9)\nrecs = CurriculumRecommender().generate(cov).ok()\nprint(json.dumps({'emerging': em,\n    'recs': [(r.type, r.skill_name, r.priority, r.message) for r in recs]},\n    ensure_ascii=False, sort_keys=True))"


def _run(seed):
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = str(seed)
    p = subprocess.run(
        [sys.executable, '-c', SNIPPET],
        cwd=str(REPO), capture_output=True, text=True, env=env, timeout=300,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    lines = [l for l in p.stdout.splitlines() if l.startswith('{')]
    assert len(lines) == 1, p.stdout[-500:]
    return lines[0]


def test_deterministic_across_hash_seeds():
    out0 = _run(0)
    out1 = _run(1)
    assert out0 == out1


def test_deterministic_twice_same_seed():
    assert _run(0) == _run(0)
