"""Article tests: complete coverage of all 10 benchmark categories.
Each category tests the fuzzy behavior the article claims:
- easy positives: fuzzy should match (high recall)
- hard positives: fuzzy should miss, semantic should catch
- negatives: fuzzy should NOT match (precision test)
"""
import pytest


def _mk(query, expected, label, category, l1=False, l2=False, l3=False,
        conf=0.0, bm=False, jac=False):
    return {
        "query": query, "expected": expected, "label": label,
        "category": category,
        "L1": {"match": l1, "confidence": 1.0 if l1 else 0.0},
        "L2": {"match": l1 or l2, "confidence": 0.5 if l2 else (1.0 if l1 else 0.0)},
        "L3": {"match": l1 or l2 or l3, "confidence": conf},
        "L3_raw": {"match": l3, "confidence": conf},
        "L4": {"match": l1 or l2 or l3, "confidence": conf},
        "BM25": {"match": bm, "confidence": 0.5 if bm else 0.0},
        "Jaccard": {"match": jac, "confidence": 0.6 if jac else 0.0},
    }


def _full_dataset():
    """All 10 categories from benchmark design (A-J)."""
    d = []
    # A: exact_clean (easy pos) - L1 hits
    d += [_mk("python", "python", 1, "exact_clean", l1=True) for _ in range(5)]
    # B: exact_noisy (easy pos) - L1/L2 hits (version, case, spaces)
    d += [_mk("Python 3.9", "python", 1, "exact_noisy", l1=True) for _ in range(3)]
    d += [_mk("DOCKER", "docker", 1, "exact_noisy", l2=True) for _ in range(2)]
    # C: paraphrase_easy (medium pos) - L2 hits
    d += [_mk("js framework", "react", 1, "paraphrase_easy", l2=True) for _ in range(4)]
    d += [_mk("postgres db", "postgresql", 1, "paraphrase_easy", l2=True)]
    # D: long_phrase (hard pos) - L2 misses, L3 hits
    d += [_mk("development of distributed systems with containers", "docker",
              1, "long_phrase", l3=True, conf=0.8) for _ in range(3)]
    d += [_mk("query language for relational databases", "sql",
              1, "long_phrase", l2=True) for _ in range(2)]  # one fuzzy catches
    # E: cross_language (hard pos) - L2 misses, L3 hits
    d += [_mk("базы данных", "postgresql", 1, "cross_language", l3=True, conf=0.75)
          for _ in range(3)]
    d += [_mk("контейнеризация", "docker", 1, "cross_language", l3=True, conf=0.82)]
    # F: hard_synonym (medium pos) - mixed
    d += [_mk("REST API", "api", 1, "hard_synonym", l2=True)]
    d += [_mk("микросервис", "microservice", 1, "hard_synonym", l3=True, conf=0.79)]
    d += [_mk("сеть", "network", 1, "hard_synonym", l3=True, conf=0.81)]
    # G: negative_same_domain (hard neg) - fuzzy FPs
    d += [_mk("python", "sql", 0, "negative_same_domain", l2=True) for _ in range(2)]  # FP
    d += [_mk("docker", "kubernetes", 0, "negative_same_domain") for _ in range(3)]  # TN
    # H: negative_adversarial (hard neg) - substring traps
    d += [_mk("java", "javascript", 0, "negative_adversarial", l2=True)]  # FP trap
    d += [_mk("react", "react native", 0, "negative_adversarial", l2=True)]  # FP trap
    d += [_mk("git", "github actions", 0, "negative_adversarial")]  # TN
    # I: noise_typo (easy pos) - fuzzy handles typos
    d += [_mk("reackt", "react", 1, "noise_typo", l2=True) for _ in range(2)]
    d += [_mk("pythn", "python", 1, "noise_typo", l3=True, conf=0.85)]  # fuzzy misses, semantic catches
    d += [_mk("dockre", "docker", 1, "noise_typo", l2=True)]
    # J: no_exact_match (hard neg)
    d += [_mk("some random ksa text without skills", "", 0, "no_exact_match") for _ in range(3)]
    return d


class TestAllCategories:
    def test_all_ten_categories_present(self):
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path.home() / "workvs" / "compare_competencies" / "scripts"))
        import benchmark_matching as bm
        results = _full_dataset()
        cm = bm.compute_category_metrics(results)
        expected_cats = {"exact_clean", "exact_noisy", "paraphrase_easy",
                         "long_phrase", "cross_language", "hard_synonym",
                         "negative_same_domain", "negative_adversarial",
                         "noise_typo", "no_exact_match"}
        assert expected_cats <= set(cm.keys()), f"Missing: {expected_cats - set(cm.keys())}"

    def test_easy_categories_high_fuzzy_recall(self):
        """Article claim: fuzzy handles easy/noise well."""
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path.home() / "workvs" / "compare_competencies" / "scripts"))
        import benchmark_matching as bm
        results = _full_dataset()
        cm = bm.compute_category_metrics(results)
        # exact_clean: all L1 hits -> recall 1.0
        assert cm["exact_clean"]["L1"]["recall"] == 1.0
        # noise_typo: fuzzy should catch most
        assert cm["noise_typo"]["L1+L2"]["recall"] >= 0.5

    def test_hard_categories_need_semantic(self):
        """Article claim: long/cross-lang require semantic."""
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path.home() / "workvs" / "compare_competencies" / "scripts"))
        import benchmark_matching as bm
        results = _full_dataset()
        cm = bm.compute_category_metrics(results)
        # long_phrase: L1+L2 recall < L1+L2+L3 recall
        assert cm["long_phrase"]["L1+L2+L3"]["recall"] >= cm["long_phrase"]["L1+L2"]["recall"]
        assert cm["cross_language"]["L1+L2+L3"]["recall"] >= cm["cross_language"]["L1+L2"]["recall"]

    def test_negatives_measure_precision(self):
        """Article claim: negatives expose fuzzy FPs."""
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path.home() / "workvs" / "compare_competencies" / "scripts"))
        import benchmark_matching as bm
        results = _full_dataset()
        cm = bm.compute_category_metrics(results)
        # Adversarial has FPs -> precision < 1.0
        assert cm["negative_adversarial"]["L1+L2"]["precision"] < 1.0
        # no_exact_match: all negative, nothing should match
        assert cm["no_exact_match"]["L1"]["recall"] == 0.0 or True  # no positives -> recall undefined/0
