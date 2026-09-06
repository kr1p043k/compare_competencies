"""Article tests 3: McNemar per-category + fuzzy contribution.
Verifies statistical correctness and contribution accounting.
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


class TestMcNemarPerCategory:
    def test_identical_predictions_no_significance(self):
        """L2 == L3 everywhere -> chi2=0, p=1.0, not significant."""
        results = [
            _mk("a", "a", 1, "cat", l1=True),
            _mk("b", "c", 0, "cat"),
        ]
        mc = bm.compute_mcnemar_per_category(results)
        assert mc["cat"]["chi2"] == 0.0
        assert mc["cat"]["p"] == 1.0
        assert mc["cat"]["sig"] is False

    def test_semantic_wins_significant(self):
        """Many b (fuzzy wrong, semantic right), few c -> significant."""
        results = []
        # 10 cases where fuzzy misses but semantic catches (positives)
        for i in range(10):
            results.append(_mk(f"long query {i}", f"skill{i}", 1, "hard", l3=True, conf=0.8))
        # 1 case where fuzzy right, semantic wrong
        results.append(_mk("x", "y", 0, "hard", l2=True, l3=True, conf=0.9))
        mc = bm.compute_mcnemar_per_category(results)
        # b=10 (L2 wrong: FN on 10 positives; L3 right), c>=0
        assert mc["hard"]["b"] >= 9
        assert mc["hard"]["chi2"] > 3.84  # significant at 0.05
        assert mc["hard"]["sig"] is True

    def test_structure_complete(self):
        """Each category entry has b, c, chi2, p, sig, n."""
        results = [_mk("a", "a", 1, "c1", l1=True), _mk("b", "c", 0, "c2", l2=True)]
        mc = bm.compute_mcnemar_per_category(results)
        for cat, s in mc.items():
            for key in ["b", "c", "chi2", "p", "sig", "n"]:
                assert key in s, f"Missing {key} in {cat}"
            assert 0.0 <= s["p"] <= 1.0
            assert isinstance(s["sig"], bool)

    def test_continuity_correction(self):
        """McNemar uses continuity correction: (|b-c|-1)^2/(b+c)."""
        # b=10, c=1: (|10-1|-1)^2/11 = 64/11 = 5.82
        results = [_mk(f"q{i}", f"e{i}", 1, "c", l3=True, conf=0.8) for i in range(10)]
        results.append(_mk("fp", "x", 0, "c", l2=True, l3=True, conf=0.9))
        mc = bm.compute_mcnemar_per_category(results)
        # b counts L2-wrong-L3-right; verify chi2 formula holds
        b, c = mc["c"]["b"], mc["c"]["c"]
        expected_chi2 = round((abs(b - c) - 1) ** 2 / (b + c), 4) if (b + c) > 0 else 0.0
        assert mc["c"]["chi2"] == expected_chi2


class TestFuzzyContribution:
    def test_monotonic_f1(self):
        """F1 non-decreasing as stages added (cumulative matching)."""
        results = [
            _mk("python", "python", 1, "c", l1=True),
            _mk("reackt", "react", 1, "c", l2=True),
            _mk("query", "skill", 1, "c", l3=True, conf=0.8),
            _mk("x", "y", 0, "c"),
        ]
        fc = bm.compute_fuzzy_contribution(results)
        assert fc["L1+L2_fuzzy"]["f1"] >= fc["L1_exact"]["f1"]
        assert fc["L1+L2+L3_semantic"]["f1"] >= fc["L1+L2_fuzzy"]["f1"]

    def test_delta_accounting(self):
        """delta_vs_prev sums correctly: L1 + d12 + d23 = L123."""
        results = [
            _mk("a", "a", 1, "c", l1=True),
            _mk("b", "b", 1, "c", l2=True),
            _mk("c", "d", 0, "c"),
        ]
        fc = bm.compute_fuzzy_contribution(results)
        total = fc["L1_exact"]["f1"] + fc["L1+L2_fuzzy"]["delta_vs_prev"] \
            + fc["L1+L2+L3_semantic"]["delta_vs_prev"]
        assert abs(total - fc["L1+L2+L3_semantic"]["f1"]) < 1e-6

    def test_first_delta_zero(self):
        """First stage has delta 0 and prev None."""
        results = [_mk("a", "a", 1, "c", l1=True)]
        fc = bm.compute_fuzzy_contribution(results)
        assert fc["L1_exact"]["delta_vs_prev"] == 0.0
        assert fc["L1_exact"]["prev"] is None

    def test_fuzzy_adds_recall(self):
        """Article core claim: fuzzy adds recall over exact alone."""
        results = [
            _mk("python", "python", 1, "c", l1=True),
            _mk("reackt", "react", 1, "c", l2=True),  # only fuzzy catches
            _mk("dockre", "docker", 1, "c", l2=True),  # only fuzzy catches
        ]
        fc = bm.compute_fuzzy_contribution(results)
        assert fc["L1+L2_fuzzy"]["delta_vs_prev"] > 0
