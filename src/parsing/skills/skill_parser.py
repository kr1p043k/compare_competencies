"""
Типизированный парсер навыков с полным логированием.
Отвечает ТОЛЬКО за извлечение навыков, не за валидацию!
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum

import structlog

from src import Err, Ok, Result
from src.errors import DomainError
from src.models.vacancy import Vacancy
from src.parsing.skills.skill_normalizer import SkillNormalizer

logger = structlog.get_logger(__name__)

# Cyrillic ↔ Latin homoglyph map for skill matching
_CYR_TO_LAT = str.maketrans({
    'а': 'a', 'А': 'a',
    'е': 'e', 'Е': 'e',
    'о': 'o', 'О': 'o',
    'р': 'p', 'Р': 'p',
    'с': 'c', 'С': 'c',
    'у': 'y', 'У': 'y',
    'х': 'x', 'Х': 'x',
    'к': 'k', 'К': 'k',
    'м': 'm', 'М': 'm',
    'н': 'n', 'Н': 'n',
    'в': 'b', 'В': 'b',
})

_LAT_TO_CYR = str.maketrans({
    'a': 'а', 'A': 'а',
    'e': 'е', 'E': 'е',
    'o': 'о', 'O': 'о',
    'p': 'р', 'P': 'р',
    'c': 'с', 'C': 'с',
    'y': 'у', 'Y': 'у',
    'x': 'х', 'X': 'х',
    'k': 'к', 'K': 'к',
    'm': 'м', 'M': 'м',
    'n': 'н', 'N': 'н',
    'b': 'в', 'B': 'в',
})


def _normalize_for_matching(text: str) -> str:
    """Lowercase + normalize both Cyrillic and Latin homoglyphs to Latin.
    Applies LAT→CYR first (to catch mixed-script words like "cиcтeмнoe"),
    then CYR→LAT (to canonicalize everything to Latin)."""
    t = text.lower()
    t = t.translate(_LAT_TO_CYR)
    t = t.translate(_CYR_TO_LAT)
    return t


def _slice_base(text: str, text_norm: str) -> str:
    """Строка, по индексам которой вырезается оригинал навыка.

    `_normalize_for_matching` свёртывает регистр и гомоглифы посимвольно, поэтому
    позиции в `text_norm` совпадают с позициями в `text.lower()`. Для обычного
    текста `lower()` длину не меняет, тогда режем прямо из исходной строки и
    сохраняем точное написание; иначе (редкий случай смены длины при lower)
    откатываемся на lower-версию, где соответствие позиций гарантировано.
    """
    return text if len(text) == len(text_norm) else text.lower()


def _iter_tokens(segment: str, splitter: re.Pattern):
    """Разбить `segment` по `splitter`, отдавая (токен, смещение_в_segment).

    Нужно, чтобы не терять позиции: `re.split` их не сохраняет, а нам нужно
    вырезать оригинальный фрагмент по абсолютным индексам исходного текста.
    """
    pos = 0
    for m in splitter.finditer(segment):
        if m.start() > pos:
            yield segment[pos:m.start()], pos
        pos = m.end()
    if pos < len(segment):
        yield segment[pos:], pos


_VERSION_RE = re.compile(r"[-\s]*v?\d+(?:\.\d+)*")


# Module-level patterns for location/abbreviation filtering.
# Moved from _marker_search locals (C1 fix: _regex_search referenced them as undefined).
LOCATION_PATTERNS = [
    re.compile(r"метро\s", re.IGNORECASE),
    re.compile(r"м\.\s*[А-ЯЁа-яё]", re.IGNORECASE),
    re.compile(r"ул\.\s", re.IGNORECASE),
    re.compile(r"д\.\s*\d", re.IGNORECASE),
    re.compile(r"г\.\s*[А-ЯЁ]", re.IGNORECASE),
    re.compile(r"район\s", re.IGNORECASE),
    re.compile(r"станци[яию]\s", re.IGNORECASE),
]
# 2-5 letter abbreviations are likely noise, not skills
ABBREV_RE = re.compile(r"^[А-ЯЁ]{2,5}$")

# Homoglyph map for parse_vacancy dedup (F6 fix: module-level, was recreated per-call)
_HOMOGLYPH_MAP = str.maketrans("саорехуеСАОРЕХУЕ", "caopexyeCAOPEXYE")



def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


# Soft skills: их не бывает в it_skills.json (там только технологии), поэтому
# «коммуникативные навыки» и подобные требования из описания терялись.
# Канон -> список триггерных формулировок (ищутся по границе слова).
SOFT_SKILLS: dict[str, tuple[str, ...]] = {
    "коммуникация": (
        "коммуникативные навыки", "коммуникабельность", "коммуникация",
        "навыки коммуникации", "эффективная коммуникация", "умеет общаться",
        "грамотная речь", "грамотная письменная речь", "письменная речь",
        "устная речь", "навыки деловой переписки", "деловая переписка",
    ),
    "работа в команде": (
        "работа в команде", "работать в команде", "командная работа",
        "командообразование", "team player", "тимворк",
    ),
    "клиентоориентированность": (
        "клиентоориентированность", "клиентоориентированный",
        "ориентация на клиента", "работа с клиентами",
    ),
    "внимательность": (
        "внимательность", "внимательность к деталям", "усидчивость",
    ),
    "стрессоустойчивость": (
        "стрессоустойчивость", "стрессоустойчивый", "работа под давлением",
    ),
    "самоорганизация": (
        "самоорганизация", "самостоятельность", "тайм-менеджмент",
        "управление временем", "умение планировать",
    ),
    "аналитическое мышление": (
        "аналитическое мышление", "аналитические навыки", "системное мышление",
        "критическое мышление",
    ),
    "многозадачность": (
        "многозадачность", "большим объёмом информации", "большим объемом информации",
        "большим объёмом данных", "большим объемом данных",
    ),
    "обучаемость": (
        "обучаемость", "быстро обучаем", "готовность к обучению",
    ),
    "инициативность": (
        "инициативность", "проактивность",
    ),
    "ответственность": (
        "ответственность", "исполнительность", "пунктуальность",
    ),
    "презентационные навыки": (
        "презентационные навыки", "навыки презентации", "публичные выступления",
    ),
    "переговоры": (
        "навыки переговоров", "ведение переговоров", "переговоры",
    ),
    "наставничество": (
        "наставничество", "менторство", "обучение сотрудников",
    ),
}

# Одно слово -> канон, для быстрого поиска
_SOFT_INDEX: dict[str, str] = {}
for _canon, _phrases in SOFT_SKILLS.items():
    for _ph in _phrases:
        _SOFT_INDEX[_ph] = _canon


def _load_it_skills() -> set[str]:
    """Load skill list from it_skills.json, normalized for matching."""
    from src import config as _cfg
    path = _cfg.IT_SKILLS_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        skills = {_normalize_for_matching(s.strip()) for s in raw if s.strip()}
        logger.info("it_skills_loaded", count=len(skills), path=str(path))
        return skills
    except Exception as exc:
        logger.warning("it_skills_load_failed", error=str(exc), path=str(path))
        return set()


class SkillSource(Enum):
    """Источник извлечения навыка"""

    KEY_SKILLS = "key_skills"
    DESCRIPTION = "description"
    SNIPPET_REQUIREMENT = "snippet_req"
    SNIPPET_RESPONSIBILITY = "snippet_resp"
    MARKER = "marker"
    REGEX = "regex"
    NGRAM = "ngram"
    DIRECT = "direct"


@dataclass
class ExtractedSkill:
    """Извлечённый из текста навык с метаданными"""

    text: str
    source: SkillSource
    raw_match: str = ""
    confidence: float = 1.0

    def __hash__(self):
        return hash(self.text.lower())

    def __eq__(self, other):
        if isinstance(other, ExtractedSkill):
            return self.text.lower() == other.text.lower()
        return False

    def __repr__(self):
        return f"ExtractedSkill('{self.text}', source={self.source.value}, conf={self.confidence:.2f})"


@dataclass
class ParsingStats:
    """Статистика парсинга"""

    total_extracted: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    text_length_processed: int = 0

    def __repr__(self):
        sources_str = ", ".join(f"{k}={v}" for k, v in sorted(self.by_source.items()))
        return f"ParsingStats(total={self.total_extracted}, sources=[{sources_str}])"


class SkillParser:
    """
    Парсер для извлечения навыков из вакансий.
    Фокусируется на ИЗВЛЕЧЕНИИ, не на фильтрации.
    """

    # populated in __init__ from it_skills.json; kept as fallback
    TECH_SKILLS: set[str] = set()

    # (frozenset_of_skills, index) - index: first-char -> skills, cached once
    _DIRECT_INDEX: tuple[frozenset, dict] | None = None

    # Скомпилированный alternation-паттерн маркеров (1110 термов собирать
    # на каждую вакансию дорого) - пересборка при смене TECH_SKILLS.
    _MARKER_PATTERN: tuple[frozenset, re.Pattern] | None = None

    SKILL_MARKERS = [
        "ключевые навыки",
        "ключевые компетенции",
        "навыки",
        "требования",
        "требования к кандидату",
        "необходимые навыки",
        "мы ждем",
        "ожидаем от вас",
        "что нужно знать",
        "профессиональные навыки",
        "опыт работы с",
        "знание",
        "владение",
        "умение",
        "должен знать",
        "должен уметь",
        "stack",
        "технологии",
        "инструменты",
    ]

    # Слова, которые указывают на отрицание
    NEGATION_WORDS = ["не требуется", "не нужно", "не обязательно", "не критично", "не важно"]

    def __init__(self):
        self.stats = ParsingStats()
        if not SkillParser.TECH_SKILLS:
            SkillParser.TECH_SKILLS = _load_it_skills()

    def parse_vacancy(self, vacancy: Vacancy) -> Result[list[ExtractedSkill], DomainError]:
        """Извлекает все навыки из вакансии"""
        try:
            skills = []

            match self._extract_from_key_skills(vacancy):
                case Ok(ks):
                    skills.extend(ks)
                case Err(e):
                    logger.debug("key_skills_extract_failed", error=str(e))

            text_sources = []
            if vacancy.snippet:
                if vacancy.snippet.requirement:
                    text_sources.append((vacancy.snippet.requirement, SkillSource.SNIPPET_REQUIREMENT, 2000))
                if vacancy.snippet.responsibility:
                    text_sources.append((vacancy.snippet.responsibility, SkillSource.SNIPPET_RESPONSIBILITY, 2000))
            if vacancy.description:
                text_sources.append((vacancy.description, SkillSource.DESCRIPTION, 10000))

            for text, source, max_len in text_sources:
                match self._extract_from_text(text, source=source, max_text_length=max_len):
                    case Ok(extracted):
                        skills.extend(extracted)
                    case Err(e):
                        logger.debug("text_extract_failed", source=source, error=str(e))

                match self._extract_soft_skills(text, source=source):
                    case Ok(soft):
                        skills.extend(soft)
                    case Err(e):
                        logger.debug("soft_skill_extract_failed", source=source, error=str(e))

            self.stats.total_extracted += len(skills)

            # Normalize cyrillic homoglyphs and deduplicate (case-insensitive)
            # _HOMOGLYPH_MAP now module-level (F6 fix).
            # Пустые имена отбрасываем здесь, чтобы "" не оседали в parsed_skills
            # и не рисовались пустыми бейджами в карточках.
            # Дедуп - по свёрнутому ключу, но ХРАНИМ ОРИГИНАЛ: свёртка необратима
            # (несколько кириллических букв → одна латинская), запись norm в text
            # портила parsed_skills в БД смешанно-алфавитной кашей.
            deduped = []
            seen: set[str] = set()
            for s in skills:
                if not (s.text or "").strip():
                    continue
                norm = s.text.lower().translate(_HOMOGLYPH_MAP)
                if norm not in seen:
                    seen.add(norm)
                    s.text = s.text.strip()
                    deduped.append(s)
            return Ok(deduped)
        except Exception as e:
            logger.exception("parse_vacancy_failed", vacancy_id=vacancy.id)
            return Err(DomainError(message=f"Failed to parse vacancy {vacancy.id}", detail=str(e)))

    def _extract_from_key_skills(self, vacancy: Vacancy) -> Result[list[ExtractedSkill], DomainError]:
        """Извлекает из официального поля key_skills.

        Структурный фильтр мусора (цифры/пунктуация/1 символ): whitelist-гейт
        здесь НЕ ставим - key_skills главный источник discovery новых навыков.
        Исключение/allowlist как в SkillMatcher._MARKET_SINGLE_ALLOW: языки
        R и C односимвольны, но реальны.
        """
        try:
            skills = []
            for key_skill in vacancy.key_skills:
                name = (key_skill.name or "").strip()
                if not name:
                    continue
                if len(name) < 2 and name.lower() not in ("r", "c"):
                    continue
                if re.fullmatch(r"[\d\s.\-,/]+", name):
                    continue
                if re.fullmatch(r"[^\w\s]+", name):
                    continue
                skill = ExtractedSkill(
                    text=name, source=SkillSource.KEY_SKILLS, raw_match=name, confidence=1.0
                )
                skills.append(skill)
                self._update_stats(SkillSource.KEY_SKILLS)
            return Ok(skills)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="_extract_from_key_skills"))

    def _extract_from_text(
        self, text: str, source: SkillSource, max_text_length: int = 1000
    ) -> Result[list[ExtractedSkill], DomainError]:
        """Извлекает навыки из текста несколькими методами"""
        try:
            if not text:
                return Ok([])

            text = text[:max_text_length]
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            self.stats.text_length_processed += len(text)

            skills = []
            for method in (self._direct_search, self._marker_search, self._regex_search):
                match method(text, source):
                    case Ok(extracted):
                        skills.extend(extracted)
                    case Err(e):
                        logger.debug("skill_extract_method_failed", method=method.__name__, error=str(e))

            return Ok(skills)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="_extract_from_text"))

    def _direct_search(self, text: str, source: SkillSource) -> Result[list[ExtractedSkill], DomainError]:
        """Прямой поиск по списку TECH_SKILLS с контролем контекста"""
        try:
            skills = []
            text_norm = _normalize_for_matching(text)
            # P5: отрицания свернуть тем же маппингом, что текст
            # («не» → «ne», иначе сравнение мёртвое).
            _negation_norms = [_normalize_for_matching(w) for w in self.NEGATION_WORDS]

            extended_skills = [
                "machine learning", "deep learning", "data science",
                "full stack", "rest api", "sql database",
                "spring boot", "scikit-learn",
            ]

            all_skills = set(self.TECH_SKILLS) | set(extended_skills)
            index = self._direct_index(all_skills)

            i = 0
            n = len(text_norm)
            while i < n:
                candidates = index.get(text_norm[i])
                if candidates:
                    for skill in candidates:
                        end = self._match_skill_at(text_norm, i, skill)
                        if end is None:
                            continue

                        if len(skill) == 1 and re.search(r'[а-яё]', text[i:end], re.IGNORECASE):
                            after = text_norm[end:end + 3]
                            if after.startswith("++"):
                                pass
                            elif re.search(r'[а-яё]', text_norm[max(0, i - 5):i], re.IGNORECASE):
                                continue

                        start = max(0, i - 50)
                        context = text_norm[start:end + 50]
                        if any(neg in context for neg in _negation_norms):
                            continue

                        original = text[i:end].strip()
                        skills.append(ExtractedSkill(
                            text=original or skill, source=source, raw_match=skill, confidence=0.95
                        ))
                        self._update_stats(source)
                i += 1

            return Ok(skills)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="_direct_search"))

    @classmethod
    def _direct_index(cls, all_skills: set[str]) -> dict[str, list[str]]:
        """Build/cache index: first char -> skills (longest first)."""
        key = frozenset(all_skills)
        if cls._DIRECT_INDEX is not None and cls._DIRECT_INDEX[0] == key:
            return cls._DIRECT_INDEX[1]
        index: dict[str, list[str]] = {}
        for skill in all_skills:
            index.setdefault(skill[0], []).append(skill)
        for lst in index.values():
            lst.sort(key=len, reverse=True)
        cls._DIRECT_INDEX = (key, index)
        return index

    def _match_skill_at(self, text_norm: str, pos: int, skill: str) -> int | None:
        """Match `skill` at `pos` in normalized text; return end index (incl. version) or None."""
        if not text_norm.startswith(skill, pos):
            return None
        if skill[0].isalnum() and pos > 0 and _is_word_char(text_norm[pos - 1]):
            return None
        base_end = pos + len(skill)
        ends = [base_end]
        if skill[-1].isalnum() and len(skill) >= 2:
            m = _VERSION_RE.match(text_norm, base_end)
            if m:
                ends.insert(0, m.end())
        for end in ends:
            if skill[-1].isalnum() and end < len(text_norm) and _is_word_char(text_norm[end]):
                continue
            return end
        return None

    def _extract_soft_skills(
        self, text: str, source: SkillSource
    ) -> Result[list[ExtractedSkill], DomainError]:
        """Soft skills по словарю формулировок.

        Технический whitelist их не содержит, поэтому без этого прохода
        требования вроде «коммуникативные навыки» не попадали в parsed_skills
        и не показывались в карточке вакансии.
        """
        try:
            if not text:
                return Ok([])

            text_norm = _normalize_for_matching(re.sub(r"<[^>]+>", " ", text))
            found: list[ExtractedSkill] = []
            seen: set[str] = set()

            for phrase, canon in _SOFT_INDEX.items():
                if canon in seen:
                    continue
                phrase_norm = _normalize_for_matching(phrase)
                if len(phrase_norm) < 4:
                    continue
                if re.search(
                    r"(?<!\w)" + re.escape(phrase_norm) + r"(?!\w)", text_norm
                ):
                    seen.add(canon)
                    found.append(
                        ExtractedSkill(
                            text=canon,
                            source=source,
                            raw_match=phrase,
                            confidence=0.7,
                        )
                    )
            return Ok(found)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="_extract_soft_skills"))

    def _marker_search(self, text: str, source: SkillSource) -> Result[list[ExtractedSkill], DomainError]:
        """Поиск навыков после маркеров.

        P0: один проход alternation longest-first С границами слов
        (было `tech in line` - 'go' срабатывал внутри 'django').
        Односимвольные навыки пропускаем: в маркерных строках это шум.
        P5: negation-контекст ±50 как в _direct_search.
        """
        try:
            skills = []
            text_norm = _normalize_for_matching(text)
            _negation_norms = [_normalize_for_matching(w) for w in self.NEGATION_WORDS]

            # Паттерны для фильтрации не-навыков
            # LOCATION_PATTERNS and ABBREV_RE now module-level (C1 fix)

            techs = sorted(
                (t for t in self.TECH_SKILLS if len(t) >= 2),
                key=len, reverse=True,
            )
            if not techs:
                return Ok([])
            key = frozenset(techs)
            if SkillParser._MARKER_PATTERN is not None and SkillParser._MARKER_PATTERN[0] == key:
                marker_pat = SkillParser._MARKER_PATTERN[1]
            else:
                marker_pat = re.compile(
                    r"(?<!\w)(?:" + "|".join(re.escape(t) for t in techs) + r")(?!\w)")
                SkillParser._MARKER_PATTERN = (key, marker_pat)

            # Все маркеры в одном alternation (longest-first): берём регион
            # после КАЖДОГО вхождения маркера (C2 fix: было `parts[1][:600]` -
            # только первый сегмент после первого вхождения + обрезка до 600),
            # до следующего маркера или конца текста.
            marker_norms = sorted(
                {_normalize_for_matching(marker) for marker in self.SKILL_MARKERS},
                key=len, reverse=True,
            )
            heading_pat = re.compile("|".join(re.escape(m) for m in marker_norms))
            # Разделители: `·` (U+00B7) наравне с `•` (U+2022) - был пропущен.
            separators = re.compile(r"[\n,•·\-*;]")

            base = _slice_base(text, text_norm)
            for heading in heading_pat.finditer(text_norm):
                region_start = heading.end()
                next_heading = heading_pat.search(text_norm, region_start)
                region_end = next_heading.start() if next_heading else len(text_norm)
                region = text_norm[region_start:region_end]
                for raw_token, tok_off in _iter_tokens(region, separators):
                    lead = len(raw_token) - len(raw_token.lstrip())
                    token = raw_token.strip()
                    # min 2 символа: выровнено с фильтром `len(t) >= 2` в techs
                    # (было <= 3 - терялись `sql`, `git`, `go`).
                    if len(token) < 2 or len(token) >= 100:
                        continue
                    # Фильтрация локаций
                    if any(p.search(token) for p in LOCATION_PATTERNS):
                        continue
                    # Фильтрация аббревиатур
                    if ABBREV_RE.match(token):
                        continue
                    # Извлекаем только отдельные навыки из строки, не всю строку
                    for m in marker_pat.finditer(token):
                        tech = m.group(0)
                        start = max(0, m.start() - 50)
                        context = token[start:m.end() + 50]
                        if any(neg in context for neg in _negation_norms):
                            continue
                        # В text пишем ОРИГИНАЛ (точное написание из описания),
                        # а не свёрнутую латиницу: свёртка нужна только для
                        # сопоставления. Позиции text_norm == позиции base.
                        abs_start = region_start + tok_off + lead + m.start()
                        abs_end = region_start + tok_off + lead + m.end()
                        original = base[abs_start:abs_end].strip() or tech
                        skills.append(ExtractedSkill(text=original, source=source, raw_match=tech, confidence=0.8))
                        self._update_stats(source)

            return Ok(skills)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="_marker_search"))

    def _regex_search(self, text: str, source: SkillSource) -> Result[list[ExtractedSkill], DomainError]:
        """Поиск по regex паттернам.

        P0: токен обязан нормализоваться в непустой whitelist-навык
        (было: свободный текст шёл дальше и лишь частично резался валидатором).
        P5: negation-контекст ±50 (finditer вместо findall ради позиций).
        """
        try:
            skills = []
            text_norm = _normalize_for_matching(text)
            _negation_norms = [_normalize_for_matching(w) for w in self.NEGATION_WORDS]

            # Pattern prefixes normalized to match homoglyph-normalized text (C1 fix)
            _rx_p1 = _normalize_for_matching("опыт работы с|опыт с|работа с|знание|владение|умение")
            _rx_p2 = _normalize_for_matching("должен")
            _rx_p3 = _normalize_for_matching("знать|уметь")
            _rx_p4 = _normalize_for_matching("требуется|требуется знание")
            patterns = [
                (r"(?:" + _rx_p1 + r")\s+([a-zа-яё0-9\s\+\#\-]+)", 0.85),
                (r"(?:" + _rx_p2 + " (?:" + _rx_p3 + r"))\s+([a-zа-яё0-9\s\+\#\-]+)", 0.75),
                (r"(?:" + _rx_p4 + r")\s+([a-zа-яё0-9\s\+\#\-]+)", 0.80),
            ]

            base = _slice_base(text, text_norm)
            token_split = re.compile(r"[,;]\s*|\s+и\s+|\s+или\s+")
            for pattern, confidence in patterns:
                for m in re.finditer(pattern, text_norm):
                    group = m.group(1)
                    if not (3 < len(group.strip()) < 100):
                        continue
                    group_start = m.start(1)
                    # Split multi-skill phrases into individual tokens (C1 fix),
                    # сохраняя смещения внутри группы для вырезания оригинала.
                    for raw_token, tok_off in _iter_tokens(group, token_split):
                        lead = len(raw_token) - len(raw_token.lstrip())
                        token = raw_token.strip()
                        if 2 < len(token) < 50:
                            # Filter locations/abbreviations per-token (C1 fix)
                            if any(p.search(token) for p in LOCATION_PATTERNS):
                                continue
                            if ABBREV_RE.match(token):
                                continue
                            # P0: только whitelist-навыки (свёртка+нормализация)
                            match SkillNormalizer.normalize(token):
                                case Ok(norm) if norm and _normalize_for_matching(norm) in self.TECH_SKILLS:
                                    pass
                                case _:
                                    continue
                            # P5: отрицание рядом - не навык
                            start = max(0, m.start() - 50)
                            context = text_norm[start:m.end() + 50]
                            if any(neg in context for neg in _negation_norms):
                                continue
                            # В text - ОРИГИНАЛ (точное написание), не свёртка; `norm`
                            # остаётся только ключом whitelist-проверки.
                            abs_start = group_start + tok_off + lead
                            abs_end = abs_start + len(token)
                            original = base[abs_start:abs_end].strip() or norm
                            skills.append(ExtractedSkill(
                                text=original, source=source, raw_match=group.strip(),
                                confidence=confidence
                            ))
                            self._update_stats(source)

            return Ok(skills)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="_regex_search"))

    def _update_stats(self, source: SkillSource) -> None:
        """Обновляет статистику"""
        key = source.value
        self.stats.by_source[key] = self.stats.by_source.get(key, 0) + 1

    def get_stats(self) -> ParsingStats:
        """Возвращает статистику парсинга"""
        return self.stats

    def reset_stats(self) -> None:
        """Сбрасывает статистику"""
        self.stats = ParsingStats()
