"""Article tests 2: fuzzy error analysis completeness.
Verifies FP/FN breakdown, examples content, counts consistency.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path.home() / "workvs" / "compare_competencies" / "scripts"))
import benchmark_matching as bm


def _mk(query, expected, label, category, l1=False, l2=False, l3=False, conf=0.0):
    return {
        "query": query, "expected": expected, "label": label, "category": category,
        "L1": {"match": l1, "confidence": 1.0 if l1 else 0.0},
        "L2": {"match": l1 or l2, "confidence": 0.5 if l2 else 0.0},
        "L3": {"match": l1 or l2 or l3, "confidence": conf},
        "L3_raw": {"match": l3, "confidence": conf},
        "L4": {"match": l1 or l2 or l3, "confidence": conf},
        "BM25": {"match": False, "confidence": 0.0},
        "Jaccard": {"match": False, "confidence": 0.0},
    }


class TestFuzzyErrorCompleteness:
    def test_fp_examples_have_content(self):
        """FP examples contain query and expected (article needs quotable examples)."""
        results = [
            _mk("java", "javascript", 0, "negative_adversarial", l2=True),
            _mk("react", "react native", 0, "negative_adversarial", l2=True),
        ]
        fe = bm.compute_fuzzy_error_analysis(results)
        assert fe["total_fp"] == 2
        ex = fe["false_positives_by_category"]["negative_adversarial"]["examples"]
        assert len(ex) == 2
        for e in ex:
            assert "query" in e and "expected" in e
            assert len(e["query"]) > 0

    def test_fn_examples_have_content(self):
        results = [
            _mk("reackt", "react", 1, "noise_typo"),  # fuzzy misses
            _mk("xyzunknownskill", "python", 1, "long_phrase"),  # fuzzy misses
        ]
        fe = bm.compute_fuzzy_error_analysis(results)
        assert fe["total_fn"] == 2
        assert "noise_typo" in fe["false_negatives_by_category"]
        assert "long_phrase" in fe["false_negatives_by_category"]

    def test_counts_consistent(self):
        """total_fp == sum of per-category FP counts (article table integrity)."""
        results = [
            _mk("a", "b", 0, "cat1", l2=True),
            _mk("c", "d", 0, "cat1", l2=True),
            _mk("e", "f", 0, "cat2", l2=True),
            _mk("g", "h", 1, "cat2"),  # FN
            _mk("i", "i", 1, "cat1", l1=True),  # TP, not counted
        ]
        fe = bm.compute_fuzzy_error_analysis(results)
        fp_sum = sum(d["count"] for d in fe["false_positives_by_category"].values())
        fn_sum = sum(d["count"] for d in fe["false_negatives_by_category"].values())
        assert fe["total_fp"] == fp_sum == 3
        assert fe["total_fn"] == fn_sum == 1

    def test_examples_capped_at_five(self):
        """Examples capped at 5 per category (article readability)."""
        results = [_mk(f"q{i}", f"e{i}", 0, "bigcat", l2=True) for i in range(10)]
        fe = bm.compute_fuzzy_error_analysis(results)
        assert len(fe["false_positives_by_category"]["bigcat"]["examples"]) == 5
        assert fe["false_positives_by_category"]["bigcat"]["count"] == 10

    def test_no_errors_empty(self):
        """Perfect predictions -> zero FP/FN."""
        results = [
            _mk("python", "python", 1, "exact_clean", l1=True),
            _mk("java", "python", 0, "negative_adversarial"),
        ]
        fe = bm.compute_fuzzy_error_analysis(results)
        assert fe["total_fp"] == 0
        assert fe["total_fn"] == 0

    def test_queries_truncated(self):
        """Long queries truncated to 80 chars for article tables."""
        long_q = "x" * 200
        results = [_mk(long_q, "e", 0, "cat", l2=True)]
        fe = bm.compute_fuzzy_error_analysis(results)
        ex = fe["false_positives_by_category"]["cat"]["examples"][0]
        assert len(ex["query"]) <= 80
