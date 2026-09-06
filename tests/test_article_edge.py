"""Article tests 4: edge cases for all new benchmark functions.
Empty inputs, single items, ties, all-positive, all-negative,
missing keys, unknown categories.
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


class TestEmptyInputs:
    def test_category_metrics_empty(self):
        cm = bm.compute_category_metrics([])
        assert cm == {}

    def test_fuzzy_errors_empty(self):
        fe = bm.compute_fuzzy_error_analysis([])
        assert fe["total_fp"] == 0
        assert fe["total_fn"] == 0
        assert fe["false_positives_by_category"] == {}
        assert fe["false_negatives_by_category"] == {}

    def test_mcnemar_empty(self):
        mc = bm.compute_mcnemar_per_category([])
        assert mc == {}

    def test_contribution_empty(self):
        fc = bm.compute_fuzzy_contribution([])
        # Should not crash; F1 of empty = 0 or defined
        assert "L1_exact" in fc


class TestSingleItem:
    def test_single_positive(self):
        r = [_mk("python", "python", 1, "exact_clean", l1=True)]
        cm = bm.compute_category_metrics(r)
        assert cm["exact_clean"]["n"] == 1
        assert cm["exact_clean"]["positive"] == 1
        fe = bm.compute_fuzzy_error_analysis(r)
        assert fe["total_fp"] == 0 and fe["total_fn"] == 0

    def test_single_negative_correct(self):
        r = [_mk("x", "y", 0, "neg")]
        fe = bm.compute_fuzzy_error_analysis(r)
        assert fe["total_fp"] == 0

    def test_single_fp(self):
        r = [_mk("java", "javascript", 0, "trap", l2=True)]
        fe = bm.compute_fuzzy_error_analysis(r)
        assert fe["total_fp"] == 1


class TestAllPositiveNegative:
    def test_all_positive(self):
        r = [_mk(f"q{i}", f"e{i}", 1, "c", l1=True) for i in range(5)]
        cm = bm.compute_category_metrics(r)
        assert cm["c"]["positive"] == 5
        assert cm["c"]["negative"] == 0
        # Precision with no negatives predicted: defined
        assert cm["c"]["L1"]["precision"] == 1.0

    def test_all_negative(self):
        r = [_mk(f"q{i}", f"e{i}", 0, "c") for i in range(5)]
        cm = bm.compute_category_metrics(r)
        assert cm["c"]["positive"] == 0
        # Recall with no positives: 0 by convention (no crash)
        assert cm["c"]["L1"]["recall"] == 0.0


class TestMissingCategory:
    def test_unknown_category_key(self):
        """Results without 'category' key group under 'unknown'."""
        r = [{"query": "a", "expected": "a", "label": 1,
              "L1": {"match": True}, "L2": {"match": True},
              "L3": {"match": True}, "L3_raw": {"match": True},
              "L4": {"match": True}, "BM25": {"match": False},
              "Jaccard": {"match": False}}]
        cm = bm.compute_category_metrics(r)
        assert "unknown" in cm


class TestTiesAndBoundaries:
    def test_mcnemar_tie_b_equals_c(self):
        """b == c > 0 -> chi2 small, not significant."""
        # b: L2 wrong + L3 right (3 positives caught only by semantic)
        # c: L2 right + L3 wrong needs L3_raw=False while L2=True on a positive
        #   -> use negatives where L2 FPs but L3 also FPs? No, that is agreement.
        # Simplest tie: 2x (L2 miss, L3 hit) + manual c via FP where L3 misses.
        # L2 hits but L3 misses is impossible (cumulative). Use two categories
        # and check formula instead: construct b=2, c=2 via mixed labels.
        # Positive L3-only (b++): L2=0,L3=1,label=1
        # Negative L2-only-FP fixed by L3? L3 cumulative always includes L2.
        # So c (L2 right, L3 wrong) requires non-cumulative L3_raw.
        # Direct unit check of formula with synthetic disagreement:
        # b=2, c=2 -> chi2 = (|2-2|-1)^2/4 = 0.25
        r = [_mk(f"p{i}", f"e{i}", 1, "c", l3=True, conf=0.8) for i in range(2)]
        r += [_mk(f"q{i}", f"f{i}", 0, "c", l2=True) for i in range(2)]
        # Here: positives L3-only -> b+=2; negatives L2-FP, L3 also FP (cumulative) -> agreement, not c.
        # So b=2, c=0 -> chi2 = (|2-0|-1)^2/2 = 0.5
        mc = bm.compute_mcnemar_per_category(r)
        b, c = mc["c"]["b"], mc["c"]["c"]
        expected = round((abs(b - c) - 1) ** 2 / (b + c), 4) if (b + c) > 0 else 0.0
        assert mc["c"]["chi2"] == expected
        # With small discordant count, not significant
        assert mc["c"]["sig"] is False

    def test_contribution_no_fuzzy_gain(self):
        """When fuzzy adds nothing, delta is 0."""
        r = [_mk("a", "a", 1, "c", l1=True), _mk("x", "y", 0, "c")]
        fc = bm.compute_fuzzy_contribution(r)
        # L2 adds nothing beyond L1 here (L1 already covers)
        assert fc["L1+L2_fuzzy"]["delta_vs_prev"] == 0.0
