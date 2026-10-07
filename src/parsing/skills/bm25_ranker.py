"""BM25 ранкер с предфильтрацией и кэшированием."""

import json
import random
import re

import numpy as np
import pymorphy3
import structlog
from rank_bm25 import BM25Okapi

from src import Err, Ok, Result, config
from src.errors import DomainError
from src.parsing.skills.skill_normalizer import SkillNormalizer
from src.parsing.utils import load_it_skills

logger = structlog.get_logger(__name__)


_DEFAULT_STOP_LEMMAS = {
    "в","без","до","из","к","на","по","о","от","перед","при","через","с","у","за","над","об","под","про","для",
    "и","да","или","либо","не","ни","как","так","то","что","чтобы","если","хотя","пока","когда","где","который",
    "этот","тот","мой","твой","свой","наш","ваш","весь","всякий","любой",
    "the","be","to","of","and","a","in","that","have","i","it","for","not","on","with","he","as","you","do","at",
    "this","but","his","by","from","they","we","say","her","she","or","an","will","my","one","all","would","there",
    "their","what","so","up","out","if","about","who","get","which","go","me","when","make","can","like","time",
    "no","just","him","know","take","person","into","year","your","good","some","could","them","see","other","than",
    "then","now","look","only","come","its","over","think","also","back","after","use","two","how","our","work",
    "first","well","way","even","new","want","because","any","these","give","day","most","us","is","was","are",
    "been","has","had","were","said","did","may","am",
}


class BM25Ranker:
    def __init__(self):
        self._morph = pymorphy3.MorphAnalyzer()
        self._cached_corpus = None
        self._corpus_hash = None
        self._stop_lemmas = self._load_stop_lemmas()

    def _load_stop_lemmas(self) -> set[str]:
        path = getattr(config, "STOP_LEMMAS_PATH", None) or config.REFERENCE_DIR / "stop_lemmas.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning("stop_lemmas_load_failed_using_defaults", path=str(path), error=str(e))
        return _DEFAULT_STOP_LEMMAS

    def _compute_corpus_hash(self, vacancies: list) -> str:
        """Хеш ID + длин + содержимого (одни ID протухают при правках)."""
        import hashlib
        import json
        parts = []
        for v in vacancies:
            if not v:
                continue
            vid = v.get("id", "") if isinstance(v, dict) else v.id
            try:
                vtext = self._extract_vacancy_text(v)
            except Exception:
                vtext = ""
            parts.append((str(vid), len(vtext),
                          hashlib.sha256(vtext.encode("utf-8", errors="replace")).hexdigest()[:16]))
        parts.sort()
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:16]

    def _extract_vacancy_text(self, vac) -> str:
        parts = []
        if isinstance(vac, dict):
            desc = vac.get("description") or ""
            if not isinstance(desc, str):
                desc = ""
            if desc:
                parts.append(re.sub(r"<[^>]+>", " ", desc))
            snippet = vac.get("snippet") or {}
            req = snippet.get("requirement") or ""
            resp = snippet.get("responsibility") or ""
            if req:
                parts.append(re.sub(r"<[^>]+>", " ", req))
            if resp:
                parts.append(re.sub(r"<[^>]+>", " ", resp))
            key_skills = " ".join(str(s.get("name", "")) for s in vac.get("key_skills", []) if isinstance(s, dict))
            if key_skills:
                parts.append(key_skills)
        else:
            if hasattr(vac, "description") and vac.description:
                parts.append(re.sub(r"<[^>]+>", " ", vac.description))
            # Кэш-файлы без description: добираем текст из сниппета (как в dict-ветке),
            # иначе корпус пуст и hybrid_weights всегда {}.
            sn = getattr(vac, "snippet", None)
            if sn is not None:
                req = getattr(sn, "requirement", None) or (
                    sn.get("requirement") if isinstance(sn, dict) else None
                ) or ""
                resp = getattr(sn, "responsibility", None) or (
                    sn.get("responsibility") if isinstance(sn, dict) else None
                ) or ""
                if not isinstance(req, str):
                    req = ""
                if not isinstance(resp, str):
                    resp = ""
                if req:
                    parts.append(re.sub(r"<[^>]+>", " ", req))
                if resp:
                    parts.append(re.sub(r"<[^>]+>", " ", resp))
            key_skills = " ".join(
                str(getattr(s, "name", "")) for s in (vac.key_skills if hasattr(vac, "key_skills") else [])
            )
            if key_skills:
                parts.append(key_skills)
        return " ".join(p.strip() for p in parts if p and p.strip())

    def _lemmatize(self, text: str) -> list[str]:
        words = re.findall(r"(?u)\b\w[\w\+\-\#\.]+\b", text.lower())
        if not words:
            return []
        lemmas = []
        for w in words:
            if any("а" <= c <= "я" or c == "ё" for c in w):
                try:
                    lemmas.append(self._morph.parse(w)[0].normal_form)
                except Exception:
                    logger.warning("lemmatization_failed", word=w)
                    lemmas.append(w)
            else:
                lemmas.append(w)
        return lemmas

    def calculate_weights(self, vacancies: list) -> Result[dict[str, float], DomainError]:
        """BM25-веса навыков по корпусу вакансий (IDF-корпус семплируется).

        P2 (Exp-прототип: IDX-FULL F1 0.981 vs IDX-SKILL 0.936):
        доки корпуса — ПОЛНЫЕ леммы вакансии (минус стопы), а не
        whitelist-фильтрованные n-граммы. Чинит два дефекта старого дизайна:
        (1) мультисловные навыки (`get_scores(["machine learning"])` по
        однотокенным докам не матчились никогда → системный невесом);
        (2) 27% вакансий давали пустые skill-доки и выпадали из IDF.
        Запрос навыка — его леммы-униграммы; вес — средний скор по докам.
        Скоринг только навыков, чьи леммы есть хотя бы в одном доке
        (иначе отсутствующие навыки получали бы веса).
        Сэмпл — random(seed=7), не stride. k1/b — дефолты (сетка плоская).
        """
        try:
            ch = self._compute_corpus_hash(vacancies)
            if self._cached_corpus is not None and self._corpus_hash == ch:
                logger.info("BM25 из кэша")
                return Ok(self._cached_corpus)

            whitelist_raw = load_it_skills()
            whitelist = set()
            for skill in whitelist_raw:
                r = SkillNormalizer.normalize(skill)
                norm = r.unwrap() if r.is_ok() else ""
                if norm:
                    whitelist.add(norm)

            corpus_docs: list[list[str]] = []
            for vac in vacancies:
                text = self._extract_vacancy_text(vac)
                if not text:
                    continue
                lemmas = self._lemmatize(text)
                doc = [t for t in lemmas if t and t not in self._stop_lemmas]
                if doc:
                    corpus_docs.append(doc)

            if not corpus_docs:
                logger.warning("Нет валидных документов для BM25")
                return Ok({})

            # Bound IDF corpus (config BM25_MAX_CORPUS_DOCS); random, не stride
            # (stride даёт систематическое смещение по порядку вакансий).
            # Presence-проверка — по ВСЕМ докам (иначе сэмпл режет хвост навыков),
            # скоринг — по сэмплу.
            _max_docs = getattr(config, "BM25_MAX_CORPUS_DOCS", 200) or 200
            if len(corpus_docs) > _max_docs:
                _corpus_sample = random.Random(7).sample(corpus_docs, _max_docs)
            else:
                _corpus_sample = corpus_docs
            bm25 = BM25Okapi(_corpus_sample)
            doc_sets_all = [set(d) for d in corpus_docs]

            weights = {}
            for term in sorted(whitelist):
                try:
                    q = [t for t in self._lemmatize(term)
                         if t and t not in self._stop_lemmas]
                    if not q:
                        continue
                    qset = set(q)
                    if not any(qset <= ds for ds in doc_sets_all):
                        continue
                    scores = bm25.get_scores(q)
                    if len(scores) == 0:
                        continue
                    avg = float(sum(scores) / len(scores))
                    if avg > config.BM25_MIN_SCORE:
                        weights[term] = round(avg, 4)
                except ZeroDivisionError:
                    continue

            if weights:
                vals = np.array(list(weights.values()))
                vmin, vmax = vals.min(), vals.max()
                if vmax > vmin:
                    for k in weights:
                        weights[k] = round((weights[k] - vmin) / (vmax - vmin), 4)

            logger.info("BM25 готов", skill_count=len(weights))
            self._cached_corpus = weights
            self._corpus_hash = ch
            return Ok(weights)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="calculate_weights"))
