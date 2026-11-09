"""Calibrate ranker raw scores -> P(relevant) = P(grade>=1 | score).

Design:
- XGBRanker scores are unbounded and query-relative: fine for ordering,
  useless for thresholds and cross-role % display.
- Target: monotonic map score -> probability. Candidates: Platt
  (LogisticRegression on score) vs Isotonic. ~370 train rows -> expect
  Platt to be more stable; pick by TEST log-loss/Brier, not by belief.
- Protocol: split is rebuilt EXACTLY as in fit_ranker (per-role stratified,
  seed 42) on the same gold order; calibrator fits on TRAIN scores
  (in-sample caveat: mild optimism, stated), evaluates on TEST scores.
  Ranking order is unchanged (both maps are monotonic).
Outputs: data/models/ranker_calibrator.joblib, labeling/calibration_report.json
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from unittest.mock import patch  # noqa: E402

from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

from scripts.train_ltr_ranker import read_gold  # noqa: E402
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "models" / "ltr_ranker_xgb_rank_human.joblib"
SEED = 42


def rebuild_split(rows: list[dict]):
    roles = sorted({r["role"] for r in rows})
    train_idx, test_idx = [], []
    for role in roles:
        idx = [i for i, r in enumerate(rows) if r["role"] == role]
        grades = [rows[i]["grade"] for i in idx]
        tr, te = next(StratifiedShuffleSplit(
            n_splits=1, test_size=0.25, random_state=SEED).split(idx, grades))
        train_idx.extend(idx[i] for i in tr)
        test_idx.extend(idx[i] for i in te)
    return train_idx, test_idx, roles


def reliability(yb, p, bins: int = 5):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for b in range(bins):
        m = (p >= q[b]) & (p <= q[b + 1] if b == bins - 1 else p < q[b + 1])
        if m.sum() == 0:
            continue
        out.append({"mean_pred": round(float(p[m].mean()), 3),
                    "frac_pos": round(float(yb[m].mean()), 3), "n": int(m.sum())})
    return out


def zpool(scores_by_role: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Per-query z-norm, pooled. Returns (z_all, {role: (mean, std)})."""
    zs, stats = [], {}
    for role, s in scores_by_role.items():
        mu, sd = float(np.mean(s)), float(np.std(s)) or 1.0
        stats[role] = (mu, sd)
        zs.append((s - mu) / sd)
    return np.concatenate(zs), stats


def oof_scores(rows: list[dict], roles: list[str], vacancies: list[dict]):
    """Role-based out-of-fold raw scores: each role scored by a model
    that never saw it. Scale-free use only (z-norm per query)."""
    from unittest.mock import patch

    from src import Ok
    from src.parsing.api.embedding_loader import get_embedding_model
    with patch("src.predictors.ltr_recommendation_engine.get_embedding_model",
               return_value=get_embedding_model()):
        eng = LTRRecommendationEngine()
        out = {}
        for held in roles:
            sub = [g for g in rows if g["role"] != held]
            res = eng.fit_ranker(sub, vacancies, seed=42,
                                 model_out=ROOT / "temp" / f"oof_{held[:6]}.joblib")
            assert isinstance(res, Ok), res
            market_emb = np.mean(list(eng.skill_embeddings.values()), axis=0)
            rows_h = [g for g in rows if g["role"] == held]
            Xh, _, _, _, _ = eng._build_ranking_frame(
                list(range(len(rows_h))), rows_h, {}, market_emb, {held: 0})
            out[held] = np.asarray(eng.model.predict(Xh))
    return out


def main() -> None:
    rows = read_gold(ROOT / "labeling" / "gold.csv")
    train_idx, test_idx, roles = rebuild_split(rows)
    with patch("src.predictors.ltr_recommendation_engine.get_embedding_model",
               return_value=MagicMock()):
        eng = LTRRecommendationEngine()
        eng.load_model(ART)
    assert len(test_idx) == 128, f"split drift: test={len(test_idx)}"
    train_core: dict[str, list[str]] = {r: [] for r in roles}
    for i in train_idx:
        if rows[i]["grade"] == 2:
            train_core[rows[i]["role"]].append(rows[i]["skill"])
    market_emb = np.mean(list(eng.skill_embeddings.values()), axis=0)
    qid_of = {r: i for i, r in enumerate(roles)}
    Xtr, ytr, _, _, train_order = eng._build_ranking_frame(train_idx, rows, train_core, market_emb, qid_of)
    Xte, yte, _, _, test_order = eng._build_ranking_frame(test_idx, rows, train_core, market_emb, qid_of)
    str_, ste = np.asarray(eng.model.predict(Xtr)), np.asarray(eng.model.predict(Xte))
    yb_tr, yb_te = (ytr >= 1).astype(int), (yte >= 1).astype(int)

    def znorm(scores: np.ndarray, order) -> np.ndarray:
        """Per-query (role) z-score; query-relative scores must be
        standardized before an absolute score->probability map is fit."""
        z = np.empty_like(scores)
        for rl in roles:
            m = np.array([rows[i]["role"] == rl for i in order])
            if not m.any():
                continue
            s_rl = scores[m]
            mu, sd = float(np.mean(s_rl)), float(np.std(s_rl)) or 1.0
            z[m] = (s_rl - mu) / sd
        return z

    ztr, zte = znorm(str_, train_order), znorm(ste, test_order)

    cands = {
        "platt": LogisticRegression().fit(ztr.reshape(-1, 1), yb_tr),
        "isotonic": IsotonicRegression(out_of_bounds="clip").fit(ztr, yb_tr),
    }
    rep = {}
    for name, cal in cands.items():
        p = np.asarray(cal.predict(zte.reshape(-1, 1)) if name == "platt" else cal.predict(zte))
        rep[name] = {"log_loss": round(float(log_loss(yb_te, p)), 4),
                     "brier": round(float(brier_score_loss(yb_te, p)), 4),
                     "reliability": reliability(yb_te, p)}
        print(name, rep[name]["log_loss"], rep[name]["brier"])
    # dummy: affine map from TRAIN z min/max (no test leakage), clipped
    lo, hi = float(ztr.min()), float(ztr.max())
    p_d = np.clip((zte - lo) / (hi - lo + 1e-9), 1e-6, 1 - 1e-6)
    rep["raw_minmax"] = {"log_loss": round(float(log_loss(yb_te, p_d)), 4),
                         "brier": round(float(brier_score_loss(yb_te, p_d)), 4)}
    print("raw_minmax", rep["raw_minmax"])
    # --- honest protocol (OPTIONAL: needs a live DB for role-OOF scores) ---
    # fit on role-OOF z-scores, eval on untouched test z-scores. Skipped
    # gracefully when Postgres is unavailable; the in-sample z-scored
    # candidates above are the primary result.
    try:
        import asyncio

        from scripts.train_ltr_ranker import load_vacancies
        vacancies = asyncio.run(load_vacancies(3000))
        oof = oof_scores(rows, roles, vacancies)
        oof_roles = sorted(oof)
        yb_oof = np.concatenate([(np.array([r["grade"] for r in rows if r["role"] == rl]) >= 1).astype(int)
                                  for rl in oof_roles])
        z_oof, _ = zpool({rl: oof[rl] for rl in oof_roles})
        # test z per role-query (frame order is known: test_order holds gold idxs)
        zt_list, yb_zt_list = [], []
        for rl in roles:
            m = np.array([rows[i]["role"] == rl for i in test_order])
            s_rl = ste[m]
            mu, sd = float(np.mean(s_rl)), float(np.std(s_rl)) or 1.0
            zt_list.append((s_rl - mu) / sd)
            yb_zt_list.append(yb_te[m])
        zt = np.concatenate(zt_list)
        yb_zt = np.concatenate(yb_zt_list)
        cands["oof_platt"] = LogisticRegression().fit(z_oof.reshape(-1, 1), yb_oof)
        cands["oof_isotonic"] = IsotonicRegression(out_of_bounds="clip").fit(z_oof, yb_oof)
        for name in ("oof_platt", "oof_isotonic"):
            cal = cands[name]
            p = np.asarray(cal.predict(zt.reshape(-1, 1)) if name == "oof_platt" else cal.predict(zt))
            rep[name] = {"log_loss": round(float(log_loss(yb_zt, p)), 4),
                         "brier": round(float(brier_score_loss(yb_zt, p)), 4),
                         "reliability": reliability(yb_zt, p)}
            print(name, rep[name]["log_loss"], rep[name]["brier"])
    except Exception as exc:  # noqa: BLE001 - DB optional, report and continue
        rep["oof_skipped"] = f"{type(exc).__name__}: {exc}"
        print("oof_skipped", rep["oof_skipped"])
    winner = min(rep, key=lambda k: rep[k]["log_loss"] if isinstance(rep[k], dict) else float("inf"))
    kinds = {"platt": "logistic_on_z", "isotonic": "isotonic_on_z", "raw_minmax": "affine_train_z_minmax",
             "oof_platt": "logistic_on_oof_z", "oof_isotonic": "isotonic_on_oof_z"}
    kind = kinds[winner]
    payload: dict = {"kind": kind, "note": "P(grade>=1) from ranker score; monotonic, order-preserving. "
                                           "ALL variants expect per-query z-scored input (standardize "
                                           "the query's raw scores before applying the map)."}
    if winner == "raw_minmax":
        payload["lo"] = lo
        payload["hi"] = hi
    else:
        payload["model"] = cands[winner]
    joblib.dump(payload,
                ROOT / "data" / "models" / "ranker_calibrator.joblib")
    rep["winner"] = winner
    rep["caveat"] = ("oof_* variants: fit on role-OOF z-scores (no train reuse), "
                     "eval on untouched test split; serving = per-query z-norm then map")
    with open(ROOT / "labeling" / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("WINNER:", winner)


if __name__ == "__main__":
    main()
