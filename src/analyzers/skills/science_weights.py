"""Роль-зависимое смешивание рыночных и научных весов навыков.

Рынок - частоты навыков в вакансиях (``data/processed/skill_weights.json``).
Наука - частоты навыков в корпусе публикаций
(``data/processed/article_weights.json``, артефакт ``build-article-weights``).

Смесь по роли::

    w = market_max * ((1 - lambda) * w_market / market_max + lambda * w_science)

Обе половины живут в шкале 0..1 (наука нормирована на максимум роли при сборке),
результат возвращается в шкале рыночных весов, поэтому его можно отдавать в
``GapAnalyzer`` без пересчёта порогов.

Навык без научного свидетельства сохраняет рыночный вес как есть: нулевое
научное значение означало бы «в публикациях не встречается», но это не факт,
а отсутствие данных. При ``lambda = 0`` смешивание выключено и результат
побитово равен входу - это гарантирует, что фича не меняет прежнее поведение,
пока её не настроили.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from src import config

logger = structlog.get_logger(__name__)

_KEY_JUNK = re.compile(r"[^0-9a-zа-яё]+")


def _norm_key(text: str | None) -> str:
    """Ключ роли без регистра, эмодзи и пунктуации: '📊 Data Science и ML' -> 'data science и ml'."""
    return " ".join(_KEY_JUNK.sub(" ", (text or "").casefold()).split())


def _token_match(a: str, b: str) -> bool:
    """Совпадение слов роли: точное или по общему началу (science ~ scientist)."""
    if a == b:
        return True
    if len(a) < 5 or len(b) < 5:
        return False
    common = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        common += 1
    return common >= 5


def _resolve_role_key(role: str | None, keys) -> str | None:
    """Сопоставить свободное имя роли с известным ключом.

    Порядок: точное совпадение, вхождение подстроки, затем покрытие всех слов
    ключа словами роли (с учётом словоформ: «Data Scientist» -> «data science»,
    «QA Automation Engineer» -> «qa engineer»). Из нескольких кандидатов берётся
    самый длинный ключ - он специфичнее.
    """
    n = _norm_key(role)
    if not n:
        return None
    if n in keys:
        return n
    role_tokens = n.split()
    best: str | None = None
    for k in keys:
        if not k:
            continue
        if k in n or n in k:
            if best is None or len(k) > len(best):
                best = k
            continue
        key_tokens = k.split()
        if not key_tokens:
            continue
        tokens_covered = all(any(_token_match(kt, rt) for rt in role_tokens) for kt in key_tokens)
        if tokens_covered and (best is None or len(k) > len(best)):
            best = k
    return best


def _normalize_science(raw: dict) -> dict[str, float]:
    """Привести веса науки к 0..1: отрицательные срезать, максимум роли = 1."""
    clean: dict[str, float] = {}
    for skill, value in (raw or {}).items():
        if not isinstance(skill, str) or not skill.strip():
            continue
        try:
            w = float(value)
        except (TypeError, ValueError):
            continue
        if w <= 0:
            continue
        clean[skill.strip().casefold()] = w
    peak = max(clean.values(), default=0.0)
    if peak > 0:
        clean = {s: min(1.0, w / peak) for s, w in clean.items()}
    return clean


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("role_source_mix_read_failed", path=str(path), error=str(exc))
        return {}
    return data if isinstance(data, dict) else {}


@dataclass(frozen=True)
class RoleSourceMix:
    """Смесь рыночных и научных весов, разрешённая по роли."""

    article_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    global_weights: dict[str, float] = field(default_factory=dict)
    role_lambdas: dict[str, float] = field(default_factory=dict)
    default_lambda: float = 0.0
    source: str = "disabled"

    # ---------------------------------------------------------------- загрузка

    @classmethod
    def disabled(cls) -> RoleSourceMix:
        return cls()

    @classmethod
    def from_files(
        cls,
        article_path: Path | None = None,
        mix_path: Path | None = None,
    ) -> RoleSourceMix:
        """Прочитать оба артефакта. Любой отсутствующий = фича инертна."""
        article_path = Path(article_path or config.ARTICLE_WEIGHTS_PATH)
        mix_path = Path(mix_path or config.ROLE_SOURCE_MIX_PATH)

        article_raw = _read_json(article_path)
        mix_raw = _read_json(mix_path)
        if not article_raw or not mix_raw:
            logger.info(
                "role_source_mix_inactive",
                article=str(article_path),
                mix=str(mix_path),
                article_found=bool(article_raw),
                mix_found=bool(mix_raw),
            )
            return cls()

        roles_raw = article_raw.get("roles")
        roles_raw = roles_raw if isinstance(roles_raw, dict) else {}
        article: dict[str, dict[str, float]] = {}
        for role, weights in roles_raw.items():
            if isinstance(weights, dict):
                normalized = _normalize_science(weights)
                if normalized:
                    article[_norm_key(str(role))] = normalized
        global_weights = _normalize_science(
            article_raw.get("global") if isinstance(article_raw.get("global"), dict) else {}
        )

        lambdas: dict[str, float] = {}
        roles_mix = mix_raw.get("roles")
        mix_items = roles_mix.items() if isinstance(roles_mix, dict) else ()
        for role, value in mix_items:
            try:
                lam = float(value)
            except (TypeError, ValueError):
                logger.warning("role_source_mix_bad_lambda", role=str(role), value=repr(value))
                continue
            lambdas[_norm_key(str(role))] = max(0.0, min(1.0, lam))
        try:
            default_lambda = float(mix_raw.get("default", 0.0))
        except (TypeError, ValueError):
            default_lambda = 0.0
        default_lambda = max(0.0, min(1.0, default_lambda))

        mix = cls(
            article_weights=article,
            global_weights=global_weights,
            role_lambdas={k: v for k, v in lambdas.items() if k},
            default_lambda=default_lambda,
            source=f"{article_path.name}+{mix_path.name}",
        )
        logger.info(
            "role_source_mix_loaded",
            roles_with_science=len(article),
            science_skills=len(global_weights),
            mix_roles=len(mix.role_lambdas),
            default_lambda=default_lambda,
            active=mix.active,
        )
        return mix

    # ---------------------------------------------------------------- запросы

    @property
    def active(self) -> bool:
        """Есть научные веса и хотя бы одна ненулевая lambda."""
        has_science = bool(self.article_weights or self.global_weights)
        has_lambda = self.default_lambda > 0.0 or any(v > 0.0 for v in self.role_lambdas.values())
        return has_science and has_lambda

    def lambda_for(self, role: str | None) -> float:
        key = _resolve_role_key(role, self.role_lambdas.keys())
        if key is None:
            return self.default_lambda
        return self.role_lambdas[key]

    def science_for(self, role: str | None) -> dict[str, float]:
        key = _resolve_role_key(role, self.article_weights.keys())
        if key is not None:
            return self.article_weights[key]
        return self.global_weights

    # ---------------------------------------------------------------- смесь

    def blend(self, market_weights: dict[str, float], role: str | None = None) -> dict[str, float]:
        """Смешать веса рынка с научными весами роли. lambda=0 - вход без изменений."""
        lam = self.lambda_for(role)
        if lam <= 0.0 or not market_weights:
            return dict(market_weights)
        science = self.science_for(role)
        if not science:
            return dict(market_weights)
        market_max = max(market_weights.values(), default=0.0)
        if market_max <= 0.0:
            return dict(market_weights)

        blended: dict[str, float] = {}
        for skill, weight in market_weights.items():
            sci = science.get(skill.strip().casefold())
            if sci is None:
                blended[skill] = weight
                continue
            market_norm = max(0.0, min(1.0, weight / market_max))
            blended[skill] = market_max * ((1.0 - lam) * market_norm + lam * sci)
        return blended

    def blend_by_level(
        self,
        weights_by_level: dict[str, dict[str, float]],
        role: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """Смешать веса каждого уровня. Неактивная смесь возвращает вход как есть."""
        if not self.active or self.lambda_for(role) <= 0.0:
            return weights_by_level
        return {level: self.blend(weights, role) for level, weights in weights_by_level.items()}


_cached: tuple[tuple[float, float], RoleSourceMix] | None = None


def default_mix() -> RoleSourceMix:
    """Синглтон с перечитыванием по mtime: файлы правятся без рестарта API."""
    global _cached
    if not config.ROLE_SOURCE_MIX_ENABLED:
        return RoleSourceMix.disabled()
    try:
        stamps = (
            Path(config.ARTICLE_WEIGHTS_PATH).stat().st_mtime,
            Path(config.ROLE_SOURCE_MIX_PATH).stat().st_mtime,
        )
    except OSError:
        stamps = (0.0, 0.0)
    if _cached is not None and _cached[0] == stamps:
        return _cached[1]
    mix = RoleSourceMix.from_files()
    _cached = (stamps, mix)
    return mix
