import pytest

from src.evaluation.metrics import (
    RetrievalMetrics,
    RegressionMetrics,
    ClassificationMetrics,
    ClusteringMetrics,
)
import numpy as np


class TestRetrievalMetrics:
    def test_precision_at_k(self):
        rel = {"a", "b"}
        ret = ["a", "c", "b", "d"]
        assert RetrievalMetrics.precision_at_k(rel, ret, 1) == 1.0
        assert RetrievalMetrics.precision_at_k(rel, ret, 2) == 0.5
        assert RetrievalMetrics.precision_at_k(rel, ret, 4) == 0.5

    def test_precision_at_k_zero(self):
        assert RetrievalMetrics.precision_at_k(set(), ["a"], 1) == 0.0
        assert RetrievalMetrics.precision_at_k({"a"}, [], 1) == 0.0
        assert RetrievalMetrics.precision_at_k({"a"}, ["b"], 0) == 0.0

    def test_recall_at_k(self):
        rel = {"a", "b"}
        ret = ["a", "c"]
        assert RetrievalMetrics.recall_at_k(rel, ret, 1) == 0.5
        assert RetrievalMetrics.recall_at_k(rel, ret, 2) == 0.5

    def test_recall_at_k_empty_relevant(self):
        assert RetrievalMetrics.recall_at_k(set(), ["a"], 1) == 1.0

    def test_recall_at_k_empty_retrieved(self):
        assert RetrievalMetrics.recall_at_k({"a"}, [], 1) == 0.0

    def test_f1_at_k(self):
        rel = {"a"}
        ret = ["a", "b"]
        f1 = RetrievalMetrics.f1_at_k(rel, ret, 1)
        assert f1 == pytest.approx(1.0)
        f1_2 = RetrievalMetrics.f1_at_k(rel, ret, 2)
        assert f1_2 == pytest.approx(2 * 0.5 * 1.0 / 1.5)

    def test_f1_at_k_zero(self):
        assert RetrievalMetrics.f1_at_k(set(), ["a"], 1) == 0.0

    def test_average_precision(self):
        rel = {"a", "c"}
        ret = ["a", "b", "c", "d"]
        ap = RetrievalMetrics.average_precision(rel, ret)
        assert ap == pytest.approx((1.0 + 2 / 3) / 2)

    def test_average_precision_no_hits(self):
        assert RetrievalMetrics.average_precision({"x"}, ["a", "b"]) == 0.0

    def test_map(self):
        queries = [
            ({"a"}, ["a", "b"]),
            ({"c"}, ["a", "b", "c"]),
        ]
        map_val = RetrievalMetrics.mean_average_precision(queries)
        assert map_val == pytest.approx((1.0 + 1 / 3) / 2)

    def test_map_empty(self):
        assert RetrievalMetrics.mean_average_precision([]) == 0.0

    def test_mrr(self):
        queries = [
            ({"a"}, ["b", "a"]),
            ({"c"}, ["a", "b"]),
        ]
        assert RetrievalMetrics.mean_reciprocal_rank(queries) == pytest.approx((0.5 + 0.0) / 2)

    def test_mrr_empty(self):
        assert RetrievalMetrics.mean_reciprocal_rank([]) == 0.0

    def test_ndcg_at_k(self):
        rel = {"a", "b"}
        ret = ["a", "c", "b"]
        ndcg = RetrievalMetrics.ndcg_at_k(rel, ret, 3)
        assert ndcg > 0 and ndcg <= 1.0

    def test_ndcg_empty(self):
        assert RetrievalMetrics.ndcg_at_k(set(), [], 1) == 0.0

    def test_hit_rate_at_k(self):
        assert RetrievalMetrics.hit_rate_at_k({"a"}, ["a", "b"], 1) == 1.0
        assert RetrievalMetrics.hit_rate_at_k({"a"}, ["b"], 1) == 0.0
        assert RetrievalMetrics.hit_rate_at_k(set(), ["a"], 1) == 0.0

    def test_report(self):
        rel = {"a", "b"}
        ret = ["a", "c", "b"]
        r = RetrievalMetrics.report(rel, ret, ks=[1, 2])
        assert "P@1" in r
        assert "R@2" in r
        assert "F1@1" in r
        assert "NDCG@2" in r
        assert "AP" in r
        assert "MRR" in r


class TestRegressionMetrics:
    def test_mae(self):
        assert RegressionMetrics.mean_absolute_error([1, 2], [1, 3]) == 0.5

    def test_mae_empty(self):
        assert RegressionMetrics.mean_absolute_error([], []) == 0.0

    def test_mse(self):
        assert RegressionMetrics.mean_squared_error([1, 2], [1, 3]) == 0.5

    def test_rmse(self):
        assert RegressionMetrics.root_mean_squared_error([1, 2], [1, 3]) == pytest.approx(0.7071, rel=1e-3)

    def test_r2(self):
        r2 = RegressionMetrics.r2_score([1, 2, 3], [1, 2, 3])
        assert r2 == pytest.approx(1.0)
        r2_poor = RegressionMetrics.r2_score([1, 2, 3], [3, 2, 1])
        assert r2_poor < 0

    def test_r2_zero_variance(self):
        assert RegressionMetrics.r2_score([5, 5, 5], [5, 5, 5]) == 0.0

    def test_mape(self):
        mape = RegressionMetrics.mean_absolute_percentage_error([100, 200], [110, 180])
        assert mape == pytest.approx((0.1 + 0.1) / 2)

    def test_mape_all_zero(self):
        assert RegressionMetrics.mean_absolute_percentage_error([0, 0], [1, 2]) == 0.0

    def test_regression_report(self):
        r = RegressionMetrics.report([1, 2], [1, 3])
        assert "MAE" in r
        assert "RMSE" in r
        assert "R2" in r
        assert "MAPE" in r


class TestClassificationMetrics:
    def test_accuracy(self):
        assert ClassificationMetrics.accuracy([1, 0, 1], [1, 0, 1]) == 1.0
        assert ClassificationMetrics.accuracy([1, 0], [0, 1]) == 0.0

    def test_accuracy_empty(self):
        assert ClassificationMetrics.accuracy([], []) == 0.0

    def test_precision(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 1, 0]
        assert ClassificationMetrics.precision(y_true, y_pred) == pytest.approx(1.0)

    def test_precision_no_positives(self):
        assert ClassificationMetrics.precision([0, 0], [0, 0]) == 0.0

    def test_recall(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 1, 0]
        assert ClassificationMetrics.recall(y_true, y_pred) == pytest.approx(2 / 3)

    def test_f1(self):
        f1 = ClassificationMetrics.f1([1, 0, 1], [1, 0, 0])
        assert f1 == pytest.approx(2 * 1.0 * 0.5 / 1.5)

    def test_f1_zero(self):
        assert ClassificationMetrics.f1([0, 0], [0, 0]) == 0.0

    def test_classification_report(self):
        r = ClassificationMetrics.report([1, 0], [1, 0])
        assert "accuracy" in r and r["accuracy"] == 1.0
        assert "f1" in r


class TestClusteringMetrics:
    def test_silhouette_single_cluster(self):
        x = np.array([[0], [1]])
        labels = np.array([0, 0])
        assert ClusteringMetrics.silhouette_score(x, labels) == 0.0

    def test_silhouette_two_clusters(self):
        x = np.array([[0, 0], [0, 0.1], [10, 10], [10, 10.1]])
        labels = np.array([0, 0, 1, 1])
        score = ClusteringMetrics.silhouette_score(x, labels)
        assert score > 0

    def test_davies_bouldin_single(self):
        x = np.array([[0], [1]])
        labels = np.array([0, 0])
        assert ClusteringMetrics.davies_bouldin_score(x, labels) == 0.0

    def test_calinski_harabasz_single(self):
        x = np.array([[0], [1]])
        labels = np.array([0, 0])
        assert ClusteringMetrics.calinski_harabasz_score(x, labels) == 0.0

    def test_intra_cluster_distance_single(self):
        x = np.array([[0], [1]])
        labels = np.array([0, 0])
        d = ClusteringMetrics.intra_cluster_distance(x, labels)
        assert "mean_intra" in d
        assert "max_intra" in d

    def test_intra_cluster_noise_only(self):
        x = np.array([[0], [1]])
        labels = np.array([-1, -1])
        d = ClusteringMetrics.intra_cluster_distance(x, labels)
        assert d["mean_intra"] == 0.0
        assert d["max_intra"] == 0.0

    def test_intra_empty_after_noise(self):
        x = np.array([[0], [1], [2]])
        labels = np.array([0, -1, 0])
        d = ClusteringMetrics.intra_cluster_distance(x, labels)
        assert d["mean_intra"] >= 0

    def test_clustering_report(self):
        x = np.array([[0, 0], [0, 0.1], [10, 10], [10, 10.1]])
        labels = np.array([0, 0, 1, 1])
        r = ClusteringMetrics.report(x, labels)
        assert "n_clusters" in r
        assert r["n_clusters"] == 2
        assert "silhouette" in r
        assert "davies_bouldin" in r
        assert "calinski_harabasz" in r
        assert "mean_intra" in r
