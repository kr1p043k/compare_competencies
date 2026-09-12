"""Adjudicate e2 vs e3 into gold standard + round-2 review file.

Rules (documented, reproducible):
- agree -> keep
- e2=0/e3=2 (extreme) -> explicit EXTREME map
- e2=1/e3=2 -> 1 (e3 inflation band, no core found)
- e2=0/e3=1 -> 1, except DEMOTE0 (closed ecosystems) -> 0
- e2=2/e3=1 -> 1
Outputs: labeling/gold.csv, labeling/round_2.csv
"""
import csv
import glob

EXTREME = {
    119: 1, 133: 1, 198: 0, 208: 0, 209: 0, 211: 1, 212: 1,
    232: 0, 235: 0, 278: 0, 338: 0, 340: 0, 409: 1, 412: 0,
    461: 0, 463: 0, 476: 0,
}
EXTREME_WHY = {
    119: "hadoop устарел, но смежен с DS",
    133: "cmake смежен (C++-сборки)",
    198: "netlify — фронтенд-хостинг, не бэкенд",
    208: "appium — мобильное тестирование, не бэкенд",
    209: "юзабилити — не задача бэкенда",
    211: "jest смежен (Node-тесты)",
    212: "mocha смежен (Node-тесты)",
    232: "риск-менеджмент — не задача бэкенда",
    235: "управление рисками ИИ — не задача бэкенда",
    278: "data vault — моделирование, не DevOps-ядро",
    338: "a/b — аналитика, не DevOps",
    340: "бизнес-анализ — не DevOps",
    409: "безопасность алгоритмов смежна с анализом требований",
    412: "fairness — ML-этика, не СА",
    461: "playwright — не задача аналитика",
    463: "testng — не задача аналитика",
    476: "shell — не задача аналитика",
}
DEMOTE0 = {242, 454, 455, 456, 457, 458, 471}  # game engines/graphics, material-ui
DEMOTE_WHY = "замкнутая экосистема без точек касания с ролью"


def load(pattern, has_argtype=False):
    rows = {}
    for f in glob.glob(pattern):
        with open(f, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh, delimiter=";"):
                i = int(r["id"])
                rows[i] = {"role": r["role"], "skill": r["skill"], "vote": int(r["vote"])}
    return rows


def main():
    e2 = load("labeling/votes_e2_*.csv")
    e3 = load("labeling/votes_e3_*.csv")
    ids = sorted(set(e2) & set(e3))
    assert len(ids) == 500, len(ids)
    gold, review = [], []
    for i in ids:
        a, b = e2[i]["vote"], e3[i]["vote"]
        role, skill = e2[i]["role"], e2[i]["skill"]
        assert (role, skill) == (e3[i]["role"], e3[i]["skill"]), f"row mismatch at {i}"
        if a == b:
            g, rule, comment = a, "agree", ""
        elif a == 0 and b == 2:
            g, rule = EXTREME[i], "extreme: adjudicated"
            comment = EXTREME_WHY.get(i, "")
        elif a == 1 and b == 2:
            g, rule, comment = 1, "band 1v2: e3 inflation", "смежный, не ядро: ежедневной работы роли не требует"
        elif a == 0 and b == 1:
            if i in DEMOTE0:
                g, rule, comment = 0, "closed ecosystem", DEMOTE_WHY
            else:
                g, rule, comment = 1, "band 0v1: e2 strictness", "смежный: встречается в задачах роли эпизодически"
        else:  # a == 2 and b == 1
            g, rule, comment = 1, "band 2v1: e2 soft inflation", "процессный/софтовый навык — смежный, не ядро"
        gold.append([i, role, skill, g, rule])
        if a != b:
            review.append([i, role, skill, a, b, g, comment, ""])
    with open("labeling/gold.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "role", "skill", "gold", "rule"])
        w.writerows(gold)
    with open("labeling/round_2.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "role", "skill", "expert2", "expert3", "adjudicated", "comment", "confirmed"])
        w.writerows(review)
    from collections import Counter
    print("gold marginals:", dict(sorted(Counter(g for _, _, _, g, _ in gold).items())))
    print("review rows:", len(review))


if __name__ == "__main__":
    main()
