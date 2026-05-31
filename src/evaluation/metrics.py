from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class RetrievalMetrics:
    @staticmethod
    def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
        if k <= 0:
            return 0.0
        top = retrieved[:k]
        if not top:
            return 0.0
        return len([x for x in top if x in relevant]) / k

    @staticmethod
    def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
        if not relevant:
            return 1.0
        top = retrieved[:k]
        if not top:
            return 0.0
        return len([x for x in top if x in relevant]) / len(relevant)

    @staticmethod
    def f1_at_k(relevant: set, retrieved: list, k: int) -> float:
        p = RetrievalMetrics.precision_at_k(relevant, retrieved, k)
        r = RetrievalMetrics.recall_at_k(relevant, retrieved, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def average_precision(relevant: set, retrieved: list) -> float:
        hits = 0
        sum_prec = 0.0
        for i, item in enumerate(retrieved, 1):
            if item in relevant:
                hits += 1
                sum_prec += hits / i
        if hits == 0:
            return 0.0
        return sum_prec / hits

    @staticmethod
    def mean_average_precision(queries: list[tuple[set, list]]) -> float:
        if not queries:
            return 0.0
        aps = [RetrievalMetrics.average_precision(rel, ret) for rel, ret in queries]
        return sum(aps) / len(aps)

    @staticmethod
    def mean_reciprocal_rank(queries: list[tuple[set, list]]) -> float:
        if not queries:
            return 0.0
        rrs = []
        for relevant, retrieved in queries:
            for i, item in enumerate(retrieved, 1):
                if item in relevant:
                    rrs.append(1.0 / i)
                    break
            else:
                rrs.append(0.0)
        return sum(rrs) / len(rrs)

    @staticmethod
    def ndcg_at_k(relevant: set, retrieved: list, k: int) -> float:
        top = retrieved[:k]
        if not top:
            return 0.0
        dcg = 0.0
        for i, item in enumerate(top):
            if item in relevant:
                dcg += 1.0 / math.log2(i + 2)
        ideal = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def hit_rate_at_k(relevant: set, retrieved: list, k: int) -> float:
        top = retrieved[:k]
        if not top:
            return 0.0
        return 1.0 if any(x in relevant for x in top) else 0.0

    @staticmethod
    def report(relevant: set, retrieved: list, ks: list[int] | None = None) -> dict[str, float]:
        if ks is None:
            ks = [1, 3, 5, 10, 20]
        result = {}
        for k in ks:
            result[f"P@{k}"] = RetrievalMetrics.precision_at_k(relevant, retrieved, k)
            result[f"R@{k}"] = RetrievalMetrics.recall_at_k(relevant, retrieved, k)
            result[f"F1@{k}"] = RetrievalMetrics.f1_at_k(relevant, retrieved, k)
            result[f"NDCG@{k}"] = RetrievalMetrics.ndcg_at_k(relevant, retrieved, k)
        result["AP"] = RetrievalMetrics.average_precision(relevant, retrieved)
        result["MAP"] = result["AP"]
        result["MRR"] = RetrievalMetrics.mean_reciprocal_rank([(relevant, retrieved)])
        return result


class RegressionMetrics:
    @staticmethod
    def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        if not y_true or not y_pred:
            return 0.0
        return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

    @staticmethod
    def mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        if not y_true or not y_pred:
            return 0.0
        return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

    @staticmethod
    def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        return float(math.sqrt(RegressionMetrics.mean_squared_error(y_true, y_pred)))

    @staticmethod
    def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        y_true_a = np.array(y_true)
        y_pred_a = np.array(y_pred)
        ss_res = np.sum((y_true_a - y_pred_a) ** 2)
        ss_tot = np.sum((y_true_a - np.mean(y_true_a)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    @staticmethod
    def mean_absolute_percentage_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        y_true_a = np.array(y_true, dtype=float)
        y_pred_a = np.array(y_pred, dtype=float)
        mask = y_true_a != 0
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs((y_true_a[mask] - y_pred_a[mask]) / y_true_a[mask])))

    @staticmethod
    def report(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
        return {
            "MAE": RegressionMetrics.mean_absolute_error(y_true, y_pred),
            "RMSE": RegressionMetrics.root_mean_squared_error(y_true, y_pred),
            "R2": RegressionMetrics.r2_score(y_true, y_pred),
            "MAPE": RegressionMetrics.mean_absolute_percentage_error(y_true, y_pred),
        }


class ClassificationMetrics:
    @staticmethod
    def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        if not y_true:
            return 0.0
        return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    @staticmethod
    def precision(y_true: Sequence[int], y_pred: Sequence[int], pos_label: int = 1) -> float:
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == pos_label and t == pos_label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == pos_label and t != pos_label)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def recall(y_true: Sequence[int], y_pred: Sequence[int], pos_label: int = 1) -> float:
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == pos_label and t == pos_label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if p != pos_label and t == pos_label)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def f1(y_true: Sequence[int], y_pred: Sequence[int], pos_label: int = 1) -> float:
        p = ClassificationMetrics.precision(y_true, y_pred, pos_label)
        r = ClassificationMetrics.recall(y_true, y_pred, pos_label)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def report(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
        return {
            "accuracy": ClassificationMetrics.accuracy(y_true, y_pred),
            "precision": ClassificationMetrics.precision(y_true, y_pred),
            "recall": ClassificationMetrics.recall(y_true, y_pred),
            "f1": ClassificationMetrics.f1(y_true, y_pred),
        }


class ClusteringMetrics:
    @staticmethod
    def silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
        from sklearn.metrics import silhouette_score as sk_silhouette
        if len(set(labels)) < 2:
            return 0.0
        try:
            return float(sk_silhouette(x, labels, metric="cosine"))
        except Exception:
            return 0.0

    @staticmethod
    def davies_bouldin_score(x: np.ndarray, labels: np.ndarray) -> float:
        from sklearn.metrics import davies_bouldin_score as sk_db
        if len(set(labels)) < 2:
            return 0.0
        try:
            return float(sk_db(x, labels))
        except Exception:
            return 0.0

    @staticmethod
    def calinski_harabasz_score(x: np.ndarray, labels: np.ndarray) -> float:
        from sklearn.metrics import calinski_harabasz_score as sk_ch
        if len(set(labels)) < 2:
            return 0.0
        try:
            return float(sk_ch(x, labels))
        except Exception:
            return 0.0

    @staticmethod
    def intra_cluster_distance(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        unique = set(labels)
        if -1 in unique:
            unique.discard(-1)
        if not unique:
            return {"mean_intra": 0.0, "max_intra": 0.0}
        distances = []
        for label in unique:
            mask = labels == label
            cluster = x[mask]
            if len(cluster) < 2:
                continue
            center = np.mean(cluster, axis=0)
            cluster_dists = np.linalg.norm(cluster - center, axis=1)
            distances.extend(cluster_dists.tolist())
        if not distances:
            return {"mean_intra": 0.0, "max_intra": 0.0}
        return {"mean_intra": float(np.mean(distances)), "max_intra": float(np.max(distances))}

    @staticmethod
    def report(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        result = {}
        result["n_clusters"] = len(set(labels)) - (1 if -1 in labels else 0)
        result["silhouette"] = ClusteringMetrics.silhouette_score(x, labels)
        result["davies_bouldin"] = ClusteringMetrics.davies_bouldin_score(x, labels)
        result["calinski_harabasz"] = ClusteringMetrics.calinski_harabasz_score(x, labels)
        result.update(ClusteringMetrics.intra_cluster_distance(x, labels))
        return result
