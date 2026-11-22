"""RPD document loader - extract competencies and KSA from discipline PDFs.

Usage:
    loader = RPDLoader("temp/rpd_pdfs")
    data = loader.load_all()          # all PDFs
    disc = loader.load_discipline("Операционные системы")  # one discipline
"""

import json
import os
import re
import sys
import unicodedata

import structlog
from pypdf import PdfReader

logger = structlog.get_logger(__name__)

HAS_EASYOCR = False
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    pass

HAS_PDF2IMAGE = False
try:
    import pdf2image
    HAS_PDF2IMAGE = True
except ImportError:
    pass

# ─── Patterns ───────────────────────────────────────────────────────────────

#: Заголовок раздела РПД: «III. ТРЕБОВАНИЯ …» или «III. Требования …».
#: Класс символов БЕЗ \n: прежний [А-ЯЁA-Z\s]+ с IGNORECASE съедал перевод
#: строки и склеивал «IV. Содержание\nIII. Практические задания» в один
#: заголовок, из-за чего границы секций уезжали.
SEC_PATTERN = re.compile(
    r"(?:^|\n)[ \t]*(?:I{1,3}|IV|V|VI{1,3}|IX|X)\.?[ \t]+"
    r"[А-ЯЁA-Z][А-ЯЁA-Z \t\-–—,.:]{0,90}",
    re.IGNORECASE,
)

#: Разделитель кода: до трёх пробелов/дефисов всех видов/минуса.
#: В реальных РПД встречается «ОПК -3.1» (пробел + дефис) и «УК−8» (минус):
#: одиночный [-\s]? такие коды не ловил, и компетенция терялась целиком.
_COMP_SEP = r"[\s\-–—−]{0,3}"

#: ВПК обязателен: в 9 РПД 2025 (Java, ИИ-приложения, ЖЦ веб-приложений,
#: прикладные архитектуры ГНС, компьютерное зрение) ВСЕ компетенции - ВПК.
#: Без него строка «ВПК-2. Способен…» не становилась компетенцией, и вся
#: дисциплина давала 0 кодов и 0 ЗУН. ВПК идёт первым в альтернативе, чтобы
#: не разбираться как ПК.
COMP_PATTERN = re.compile(
    r"(ВПК" + _COMP_SEP + r"\d+(?:[.]\d+)?"
    r"|ОПК" + _COMP_SEP + r"\d+(?:[.]\d+)?|ПК" + _COMP_SEP + r"\d+(?:[.]\d+)?"
    r"|УК" + _COMP_SEP + r"\d+(?:[.]\d+)?|ППК" + _COMP_SEP + r"[РС]\d+(?:[.]\d+)?"
    r"|ИП" + _COMP_SEP + r"\d+(?:[.]\d+)?)"
)

#: Любой KSA-маркер с двоеточием (RU или EN).
KSA_ANY_RE = re.compile(
    r"(Знания|Умения|Навыки|Знать|Уметь|Владеть|Знает|Умеет|Владеет"
    r"|Knowledge|Abilities|Skills|Be able|To know|To be able|Knows|Can)(?:\s+to)?\s*:",
    re.IGNORECASE,
)

#: Слова-маркеры KSA. Английские формы нужны для англоязычных РПД
#: («To know:», «To be able to:»), где русских заголовков нет вовсе.
_KSA_WORDS = (
    r"Знания|Умения|Навыки|Знать|Уметь|Владеть|Знает|Умеет|Владеет"
    r"|Knowledge|Abilities|Skills|Be able|To know|To be able|Knows|Can"
)

#: Маркер с двоеточием - в любом месте строки («Знания: основы…»).
#: «To be able to:» - отдельный вариант английской формы.
KSA_MARKER_RE = re.compile(
    r"(" + _KSA_WORDS + r")(?:\s+to)?\s*:", re.IGNORECASE
)

#: Маркер без двоеточия - только если он занимает строку целиком.
KSA_MARKER_BARE_RE = re.compile(r"^(" + _KSA_WORDS + r")\s*$", re.IGNORECASE)


def count_ksa_markers(text: str) -> int:
    """Число KSA-маркеров: с двоеточием или отдельной строкой без него."""
    with_colon = len(KSA_ANY_RE.findall(text))
    bare = sum(1 for ln in text.split("\n") if KSA_MARKER_BARE_RE.match(ln.strip()))
    return with_colon + bare

#: Шапка таблицы компетенций, повторяющаяся на каждой странице. В KSA-текст
#: она попадать не должна (в KRM оседала как «знание»).
TABLE_HEADER_RE = re.compile(
    r"^(?:компетенци[яи]|индикаторы|результаты\s+обучения)\b", re.IGNORECASE
)

RUS_KSA_MAP = {
    "Знания": "knowledge", "Знать": "knowledge", "Знает": "knowledge",
    "Умения": "abilities", "Уметь": "abilities", "Умеет": "abilities",
    "Навыки": "skills", "Владеть": "skills", "Владеет": "skills",
}
RUS_KSA_PATTERN = re.compile(
    r"(Знания|Умения|Навыки|Знать|Уметь|Владеть|Знает|Умеет|Владеет)\s*:",
    re.IGNORECASE
)

ENG_KSA_MAP = {
    "Knowledge": "knowledge", "Know": "knowledge", "Knows": "knowledge",
    "To know": "knowledge",
    "Abilities": "abilities", "Be able": "abilities", "Able": "abilities",
    "To be able": "abilities", "Can": "abilities",
    "Skills": "skills",
}
ENG_KSA_PATTERN = re.compile(
    r"(Knowledge|Abilities|Skills|Be able)\s*:",
    re.IGNORECASE
)

BULLET_RE = re.compile(r"[\u2012\u2013\-\u2022\u2023\u25E6\u2043\u2219•‣⁃]\s")
CONT_PREFIXES = ("(", ")", ",", ";")

KSA_PREFIXES = [
    "Навыками ", "Владеет ", "Владеет навыком ", "Владеет навыками ",
    "Знает ", "Умеет ", "Знания: ", "Умения: ", "Навыки: ",
    "Знать: ", "Уметь: ", "Владеть: ",
    "Практическими приемами ", "Практическим опытом ",
    "Техниками ", "Методами ", "Инструментами ", "Опытом ",
    "Skills: ", "Knowledge: ", "Abilities: ",
]


# ─── Helpers ────────────────────────────────────────────────────────────────

def is_continuation(line: str) -> bool:
    if not line:
        return False
    return line.startswith(CONT_PREFIXES) or (line[0].islower() and not line[0].isdigit())


#: Вводная фраза, а не пункт перечня: «Знания, умения и навыки, формируемые…».
PROSE_KSA_RE = re.compile(
    r"^(?:знания|умения|навыки)[,\s]+(?:умения|навыки|знания)", re.IGNORECASE
)

#: Конец предложения - признак, что перенос строки уже не продолжение.
_SENTENCE_END = (".", ";", ":", "!", "?")


def _ends_sentence(text: str) -> bool:
    return (text or "").rstrip().endswith(_SENTENCE_END)


def _is_wrap_line(line: str) -> bool:
    """Строка похожа на перенос предыдущего пункта.

    Это либо продолжение со строчной буквы/скобки, либо строка, начавшаяся
    с аббревиатуры («ОС на основе…», «GPU в своих проектах…»). Обычная фраза
    с заглавного слова переносом не считается - она станет новым пунктом.
    """
    if is_continuation(line):
        return True
    # Аббревиатура целиком в верхнем регистре: «ОС», «GPU», «ПО», «СУБД».
    # «Знания» сюда не попадает - после «Зн» идёт строчная.
    return bool(re.match(r"^[А-ЯЁA-Z]{2,}(?![А-Яа-яЁёa-z])", line))


def clean_skill(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^[\u2012\u2013\-\u2022\u2023\u25E6\u2043\u2219•‣⁃]\s*", "", t)
    for prefix in KSA_PREFIXES:
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    for prefix in ["навыками ", "владеет ", "знает ", "умеет ",
                    "знания: ", "умения: ", "навыки: "]:
        if t.lower().startswith(prefix):
            t = t[len(prefix):]
            break
    t = re.sub(r"\s+", " ", t).strip(" ,;.:-")
    return t


def dedup_children(comp_ksa: dict) -> dict:
    cleaned = {}
    for comp in comp_ksa:
        parent = comp.rsplit(".", 1)[0]
        if "." in comp and parent in comp_ksa:
            child_flat = set(comp_ksa[comp].get("flat", []))
            parent_flat = set(comp_ksa[parent].get("flat", []))
            if child_flat.issubset(parent_flat):
                cleaned[comp] = None
                continue
        cleaned[comp] = comp_ksa[comp]
    for comp in list(cleaned.keys()):
        if cleaned[comp] is None:
            del cleaned[comp]
    return cleaned


def normalize_comp_code(code: str) -> str:
    """«ОПК -3.1» / «ПК 3» / «УК−8» -> «ОПК-3.1» / «ПК-3» / «УК-8».

    Раньше здесь было `upper().replace(" ", "-")`, и код с пробелом перед
    дефисом превращался в «ОПК--3.1». Такой код не совпадал с KRM, а в БД
    давал отдельную компетенцию-двойник.
    """
    return re.sub(r"[\s\-–—−]+", "-", (code or "").strip()).upper()


# ─── Text extraction ────────────────────────────────────────────────────────

def extract_text_pypdf(fpath: str) -> str | None:
    try:
        reader = PdfReader(fpath)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text if text.strip() else None
    except Exception:
        return None



# F13 fix: module-level OCR singleton (was recreated per-call)
_module_ocr_reader = None


def _get_module_ocr_reader():
    """Модульный синглтон EasyOCR (модель ~2GB, нельзя перезагружать на вызов)."""
    global _module_ocr_reader
    if _module_ocr_reader is None and HAS_EASYOCR:
        _module_ocr_reader = easyocr.Reader(["ru", "en"], gpu=False)
    return _module_ocr_reader
def extract_text_ocr(fpath: str) -> str | None:
    if not HAS_EASYOCR or not HAS_PDF2IMAGE:
        return None
    try:
        reader = _get_module_ocr_reader()  # F13 fix: singleton
        images = pdf2image.convert_from_path(fpath, dpi=200)
        text = ""
        for img in images:
            result = reader.readtext(img, paragraph=True, detail=1)
            for block in result:
                text += block[1] + "\n"
        return text
    except Exception:
        return None


def cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    cyr = sum(1 for c in text if "а" <= c <= "я" or "А" <= c <= "Я")
    return cyr / len(text)


# ─── KSA parser ─────────────────────────────────────────────────────────────

def _parse_ksa_line(line: str) -> list[tuple[str, str]]:
    """Parse a line that may contain multiple KSA sections.

    Returns list of (ksa_key, content_after) for each KSA marker found.
    Example: "Знания: foo Навыки: bar" -> [("knowledge", "foo"), ("skills", "bar")]
    """
    results = []
    combined = re.compile(
        r"(Знания|Умения|Навыки|Знать|Уметь|Владеть|Знает|Умеет|Владеет"
        r"|Knowledge|Abilities|Skills|Be able)\s*:\s*",
        re.IGNORECASE
    )
    eng_map_lower = {k.lower(): v for k, v in ENG_KSA_MAP.items()}
    for m in combined.finditer(line):
        key = m.group(1)
        ks_lower = key.lower()
        if ks_lower in ("знания", "знать", "знает"):
            ksa_key = "knowledge"
        elif ks_lower in ("умения", "уметь", "умеет"):
            ksa_key = "abilities"
        elif ks_lower in ("навыки", "владеть", "владеет"):
            ksa_key = "skills"
        else:
            ksa_key = eng_map_lower.get(ks_lower, "skills")
        after = line[m.end():]
        results.append((ksa_key, after))
    return results


def _ksa_markers_in(line: str) -> list[re.Match]:
    """KSA-маркеры в строке: с двоеточием (любое место) или без него (вся строка).

    Двоеточие необязательно: в «РПД_Технологии и фронтиры науки о данных»
    заголовки идут как «Знания» + перевод строки, без двоеточия. Прежний
    жёсткий `\\s*:` не находил ни одного маркера, и вся дисциплина получала
    9 кодов с пустым ЗУН. Голое слово ловим только если оно занимает строку
    целиком - иначе проза «Знания, умения и навыки, формируемые…» стала бы
    маркером.
    """
    found = list(KSA_MARKER_RE.finditer(line))
    if not found:
        bare = KSA_MARKER_BARE_RE.match(line)
        if bare:
            found = [bare]
    return found


def extract_ksa_skills(text: str) -> dict:
    """Parse text for competencies with KSA sections (Russian or English)."""
    lines = text.split("\n")
    current_comp = None
    current_ksa = None
    comp_ksa = {}
    inline_mode = False

    eng_map_lower = {k.lower(): v for k, v in ENG_KSA_MAP.items()}

    def _to_ksa_key(match) -> str:
        ks = match.group(1).lower()
        if ks in ("знания", "знать", "знает"):
            return "knowledge"
        if ks in ("умения", "уметь", "умеет"):
            return "abilities"
        if ks in ("навыки", "владеть", "владеет"):
            return "skills"
        return eng_map_lower.get(ks, "skills")

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # Код компетенции - только ЗАГОЛОВОК строки (после буллета).
        # Внутри фразы KSA ('применять подходы УК-4 к решению задач') это
        # контент: прежний search() переключал парсер на фантомную
        # компетенцию и терял остаток секции (P-ksa, 2026-10-14).
        cm_line = line.lstrip("-•\u2013\u2022 ")
        cm = COMP_PATTERN.match(cm_line)
        ksa_matches = _ksa_markers_in(line)

        # Competency line (possibly with KSA on same line)
        if cm:
            comp_code = normalize_comp_code(cm.group(1))
            if comp_code not in comp_ksa:
                comp_ksa[comp_code] = {
                    "knowledge": [], "abilities": [], "skills": [], "flat": []
                }
            current_comp = comp_code

            # Check if any KSA marker follows the competency on this line
            after_comp = cm_line[cm.end():]
            ksa_on_line = _ksa_markers_in(after_comp)
            if ksa_on_line:
                # KSA on same line as competency
                current_ksa = _to_ksa_key(ksa_on_line[0])
                after = after_comp[ksa_on_line[0].end():].strip()
                if after and len(after) >= 2 and any(c.isalnum() for c in after) and current_comp:
                    comp_ksa[current_comp][current_ksa].append(clean_skill(after))
                    comp_ksa[current_comp]["flat"].append(clean_skill(after))
                inline_mode = False
            else:
                current_ksa = None
                inline_mode = False
            continue

        # KSA header(s) on this line
        if ksa_matches:
            for i, km in enumerate(ksa_matches):
                current_ksa = _to_ksa_key(km)
                # Determine the content: from end of this KSA to the start of next, or end of line
                if i + 1 < len(ksa_matches):
                    content = line[km.end():ksa_matches[i+1].start()].strip()
                else:
                    content = line[km.end():].strip()

                if not content and i == len(ksa_matches) - 1:
                    # Last (or only) KSA with no inline content → inline_mode
                    inline_mode = True
                    break

                if content and len(content) > 3 and current_comp and current_ksa:
                    # Try to split content by bullet markers
                    skill = clean_skill(content)
                    if len(skill) >= 2 and any(c.isalnum() for c in skill):
                        comp_ksa[current_comp][current_ksa].append(skill)
                        comp_ksa[current_comp]["flat"].append(skill)
                        inline_mode = False
                    else:
                        inline_mode = True
                        break

            if len(ksa_matches) == 0:
                inline_mode = False
            continue

        if not current_comp or not current_ksa:
            continue

        # Шапка таблицы компетенций повторяется на каждой странице РПД.
        # В KSA-текст она попадать не должна (раньше оседала как «знание»
        # в нескольких компетенциях сразу).
        if TABLE_HEADER_RE.match(line) and (
            "индикатор" in line.lower() or "результат" in line.lower()
            or line.lower().startswith("компетенци")
        ):
            continue

        flat = comp_ksa[current_comp]["flat"]
        ksa_list = comp_ksa[current_comp][current_ksa]
        is_bullet = bool(BULLET_RE.match(line))

        def _add(text: str, ksa_list=ksa_list, flat=flat) -> None:
            if PROSE_KSA_RE.match(text):
                # «Знания, умения и навыки, формируемые…» - вводная фраза,
                # а не пункт перечня.
                return
            skill = clean_skill(text)
            if len(skill) >= 2 and any(c.isalnum() for c in skill):
                ksa_list.append(skill)
                flat.append(skill)

        if is_bullet:
            inline_mode = False
            _add(line)
        elif inline_mode:
            # Маркер был на своей строке («Знать:» без текста), а текст
            # начинается со строчной буквы. Это НОВЫЙ пункт, а не
            # продолжение: раньше is_continuation() уводил такую строку в
            # ветку «дописать в flat[-1]», flat был пуст, и пункт исчезал
            # молча (59 блоков из 912 в 2025, вплоть до 24/24 в одной РПД).
            _add(line)
        elif flat and _is_wrap_line(line) and not _ends_sentence(flat[-1]):
            # Перенос длинного пункта: строчная буква или аббревиатура
            # («ОС на основе стандартов…», «GPU в своих проектах;»), причём
            # предыдущий пункт не закончен. Без этой ветки терялось до 44%
            # текста KSA. Новая фраза с заглавного слова сюда НЕ попадает -
            # она станет отдельным пунктом.
            flat[-1] = (flat[-1] + " " + line).strip()
            if ksa_list:
                ksa_list[-1] = flat[-1]
        else:
            _add(line)

    return comp_ksa


# ─── Section detection ──────────────────────────────────────────────────────

def find_section_text(text: str, sec_matches: list) -> str | None:
    """Секция с требованиями к результатам - по содержимому, не по длине.

    Раньше бралась самая длинная секция, чей заголовок начинается с «III».
    В «РПД_Методы оптимизации» таких три: оглавление, настоящие требования и
    «III. Практические задания» - побеждали практические задания (8354
    символа, 0 KSA-маркеров), и парсер видел пустую секцию.

    Теперь секция оценивается по числу кодов компетенций и KSA-маркеров:
    оглавление (0 маркеров) и приложения отсеиваются, побеждает настоящая
    таблица. Если ни одна не набрала 3 маркера - фолбэк по ключевым словам.
    """
    best: str | None = None
    best_score = -1
    for i, m in enumerate(sec_matches):
        end = sec_matches[i + 1].start() if i + 1 < len(sec_matches) else len(text)
        cand = text[m.end():end]
        ksas = count_ksa_markers(cand)
        if ksas < 3:
            continue
        comps = len(COMP_PATTERN.findall(cand))
        score = comps * 10 + ksas
        if score > best_score:
            best, best_score = cand, score
    if best is not None:
        return best

    # Fallback: search for keywords
    for pat in ["ТРЕБОВАНИЯ К РЕЗУЛЬТАТАМ", "ПЛАНИРУЕМЫЕ РЕЗУЛЬТАТЫ", "КОМПЕТЕНЦИ",
                "REQUIREMENTS FOR RESULTS", "PLANNED RESULTS", "COMPETENC"]:
        idx = text.upper().find(pat)
        if idx >= 0:
            return text[idx:min(idx + 10000, len(text))]

    return None


# ─── Discipline dedup ───────────────────────────────────────────────────────

def normalize_disc_name(name: str) -> str:
    """Normalize discipline name for dedup matching.

    - NFC: collapses decomposed Cyrillic (U+0438 + combining breve U+0306 → U+0439
      «й»), otherwise decomposed names compare differently from composed ones.
    - lower + strip whitespace so lookups like 'Operating Systems (Операционные системы)'
      are consistent regardless of spacing/case; NFC makes decomposed «й» match «й».
    """
    name = unicodedata.normalize("NFC", (name or "").strip().lower())
    name = re.sub(r"\s+", "", name)
    return name


DISCIPLINE_ALIASES = {
    # English filename → canonical Russian name
    "Operating Systems (Операционные системы)": "Операционные системы",
    "Algorithmization and programming": "Алгоритмизация и программирование",
    "Databases and DBMS (Базы данных и СУБД)": "Базы данных и СУБД",
    "Higher Mathematics (Высшая математика)": "Высшая математика",
    "Special Topics in Mathematics (Специальные разделы математики)": "Специальные разделы математики",
    "Information Technology Security (Безопасность информационных технологий) 09.03.02":
        "Безопасность информационных технологий",
    "Discrete Mathematics (Дискретная математика)": "Дискретная математика",
}

# Pre-compute normalized lookup using the same normalizer as _resolve_canonical
DISC_NORM = {
    normalize_disc_name(k): v
    for k, v in DISCIPLINE_ALIASES.items()
    if v is not None
}


# ─── Main Loader ────────────────────────────────────────────────────────────

class RPDLoader:
    """Load discipline PDFs, extract competencies with KSA taxonomy.

    Features:
    - pypdf extraction with EasyOCR fallback for low-Cyrillic or damaged PDFs
    - Russian + English KSA header detection
    - Multiple section structure fallbacks
    - Duplicate discipline merging (English + Russian PDF variants)
    """

    def __init__(self, pdf_dir: str = "temp/rpd_pdfs"):
        self.pdf_dir = pdf_dir
        self._ocr_reader = None

    def _get_ocr_reader(self):
        if self._ocr_reader is None and HAS_EASYOCR:
            self._ocr_reader = easyocr.Reader(["ru", "en"], gpu=False)
        return self._ocr_reader

    # ── Public API ───────────────────────────────────────────────────────

    def load_all(
        self,
        dir_code: str = "09.03.02",
        direction_name: str | None = None,
        profile: str | None = None,
    ) -> dict:
        """Parse all PDFs in the directory, return structured KRM data.

        Args:
            dir_code: code of the direction (file stem of krm_disciplines_<dir_code>.json).
            direction_name: human-readable direction name (defaults to dir_code).
            profile: educational program profile.

        Returns:
            dict in format {code: {direction_name, profile, disciplines: {name: ...}}}
        """
        files = sorted([
            f for f in os.listdir(self.pdf_dir)
            if f.endswith(".pdf") and self._is_rpd(f)
        ])

        result = {
            dir_code: {
                "direction_name": direction_name or dir_code,
                "profile": profile or "",
                "disciplines": {}
            }
        }

        raw_disciplines = {}

        for fname in files:
            fpath = os.path.join(self.pdf_dir, fname)
            disc_name = self._discipline_name(fname)
            normalize_disc_name(disc_name)

            logger.info("rpd_pdf_loaded", file=fname, discipline=disc_name)

            # Extract text
            text = self.extract_text(fpath)
            if not text:
                continue

            # Parse competencies
            parsed = self.parse_text(text)
            if not parsed:
                continue

            # Dedup: prefer Russian name, merge skills
            canonical = self._resolve_canonical(disc_name)
            if canonical:
                disc_name = canonical

            if disc_name in raw_disciplines:
                # Merge: add missing competencies and skills
                existing = raw_disciplines[disc_name]
                for comp, data in parsed.items():
                    if comp not in existing:
                        existing[comp] = data
                    else:
                        for k in ["knowledge", "abilities", "skills", "flat"]:
                            existing_set = set(existing[comp].get(k, []))
                            for s in data.get(k, []):
                                if s not in existing_set:
                                    existing[comp][k].append(s)
                                    existing_set.add(s)
            else:
                raw_disciplines[disc_name] = parsed

        # Convert to final format
        for disc_name, comp_ksa_raw in raw_disciplines.items():
            # Dedup ДО merge: после слияния каждый child ⊆ parent по построению
            # и dedup удалил бы все суб-компетенции (audit 2026-10-14, P1).
            comp_ksa = dedup_children(comp_ksa_raw)

            # Merge parents from children
            for comp in list(comp_ksa.keys()):
                if "." in comp:
                    parent = comp.rsplit(".", 1)[0]
                    if parent in comp_ksa and parent != comp:
                        for k in ["knowledge", "abilities", "skills", "flat"]:
                            for s in comp_ksa[comp].get(k, []):
                                if s not in comp_ksa[parent].setdefault(k, []):
                                    comp_ksa[parent][k].append(s)

            if not comp_ksa:
                continue

            result[dir_code]["disciplines"][disc_name] = {
                "competencies": sorted(comp_ksa.keys()),
                "skills": {c: v["flat"] for c, v in comp_ksa.items()},
                "ksa": {
                    c: {k: v for k, v in comp_ksa[c].items() if k != "flat"}
                    for c in comp_ksa
                }
            }

        return result

    def load_discipline(self, name: str) -> dict | None:
        """Parse a single discipline by name (avoids full OCR on all PDFs)."""
        fpath = self._find_pdf(name)
        if not fpath:
            return None
        text = self.extract_text(fpath)
        if not text:
            return None
        comp_ksa = self.parse_text(text)
        if not comp_ksa:
            return None

        # Dedup до merge - иначе после слияния все дети удаляются (P1, 2026-10-14).
        comp_ksa = dedup_children(comp_ksa)
        if not comp_ksa:
            return None

        # Merge parents from children
        for comp in list(comp_ksa.keys()):
            if "." in comp:
                parent = comp.rsplit(".", 1)[0]
                if parent in comp_ksa and parent != comp:
                    for k in ["knowledge", "abilities", "skills", "flat"]:
                        for s in comp_ksa[comp].get(k, []):
                            if s not in comp_ksa[parent].setdefault(k, []):
                                comp_ksa[parent][k].append(s)
        if not comp_ksa:
            return None

        return {
            "competencies": sorted(comp_ksa.keys()),
            "skills": {c: v["flat"] for c, v in comp_ksa.items()},
            "ksa": {
                c: {k: v for k, v in comp_ksa[c].items() if k != "flat"}
                for c in comp_ksa
            }
        }

    def extract_text(self, fpath: str, use_ocr: bool = True) -> str | None:
        """Extract text: pypdf first, OCR fallback if low Cyrillic.

        Args:
            fpath: Path to PDF file
            use_ocr: If True, fall back to EasyOCR when pypdf text has <25% Cyrillic
        """
        text = extract_text_pypdf(fpath)
        if not text:
            logger.warning("pypdf_extraction_failed", file=fpath)
            return None

        ratio = cyrillic_ratio(text)
        if ratio >= 0.25:
            return text

        # Low Cyrillic - try OCR only for English-named PDFs with content
        if use_ocr and any(c.isascii() and c.isalpha() for c in text[:200]):
            logger.info("low_cyrillic_ratio_ocr_fallback", file=fpath, cyrillic_ratio=round(ratio, 3))
            ocr_text = extract_text_ocr(fpath)
            if ocr_text:
                return ocr_text

        return text

    def parse_text(self, text: str) -> dict:
        """Parse text into competency/KSA structure."""
        sec_matches = list(SEC_PATTERN.finditer(text))
        section_text = find_section_text(text, sec_matches)

        if section_text:
            comp_ksa = extract_ksa_skills(section_text)
            if comp_ksa:
                return comp_ksa

        # Fallback: scan the entire text
        comp_ksa = extract_ksa_skills(text)
        if comp_ksa:
            return comp_ksa

        return {}

    # ── Private helpers ──────────────────────────────────────────────────

    def _is_rpd(self, fname: str) -> bool:
        skip_keywords = ["Rating", "ECTS", "BPM", "Аннотация", ".sig"]
        return not any(kw in fname for kw in skip_keywords)

    def _discipline_name(self, fname: str) -> str:
        name = fname.replace(".pdf", "").strip()
        for prefix in ["РПД_", "РПП_", "ГИА_"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name.strip()

    def _find_pdf(self, name: str) -> str | None:
        name_lower = name.lower().replace(" ", "")
        candidates = []
        for f in os.listdir(self.pdf_dir):
            if not f.endswith(".pdf"):
                continue
            fn = f.replace(".pdf", "").replace("РПД_", "").strip()
            fn_lower = fn.lower().replace(" ", "")
            if name_lower not in fn_lower and name_lower.replace("(", "").replace(")", "") not in fn_lower:
                continue

            # Scoring: lower = better match
            score = 0
            # Exact match (after removing spaces)
            if fn_lower == name_lower:
                score = 0
            elif fn_lower == name_lower.replace("(", "").replace(")", ""):
                score = 1
            elif name_lower in fn_lower:
                # Substring match
                score = 10
            else:
                # Very loose match
                score = 20

            # Tiebreaker: prefer all-Cyrillic name over mixed
            has_latin = any("a" <= c <= "z" for c in fn)
            has_cyr = any("а" <= c <= "я" for c in fn)
            if has_cyr and not has_latin:
                score -= 5  # Russian-only is best
            elif has_latin and not has_cyr:
                score += 5  # English-only is worst

            # Tiebreaker: shorter filename (more likely to be plain name)
            score += len(fn) / 1000.0

            candidates.append((score, f))

        if not candidates:
            return None
        candidates.sort()
        return os.path.join(self.pdf_dir, candidates[0][1])

    def _resolve_canonical(self, name: str) -> str | None:
        norm = normalize_disc_name(name)
        # Check explicit aliases
        if norm in DISC_NORM:
            return DISC_NORM[norm]

        # Auto-detect: if name contains both Latin and Cyrillic,
        # try to find a Russian-only variant among PDFs
        has_lat = any("a" <= c <= "z" for c in name)
        has_cyr = any("а" <= c <= "я" for c in name)
        if has_lat and has_cyr:
            for f in os.listdir(self.pdf_dir):
                fn = f.replace(".pdf", "").replace("РПД_", "").strip()
                fn_lat = any("a" <= c <= "z" for c in fn)
                fn_cyr = any("а" <= c <= "я" for c in fn)
                if fn_cyr and not fn_lat:
                    # Russian-only filename - check if related
                    name_words = set(re.sub(r"[()]", "", name).lower().split())
                    fn_words = set(re.sub(r"[()]", "", fn).lower().split())
                    common = name_words & fn_words
                    if common:
                        return fn
        return None

    def save(self, data: dict, path: str = "data/reference/krm_disciplines_09.03.02.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def stats(self, data: dict) -> dict:
        sub = next(iter(data.values()), {}).get("disciplines", {}) if isinstance(data, dict) else {}
        discs = sub
        total = len(discs)
        skills = sum(len(sk) for d in discs.values() for sk in d["skills"].values())
        zeros = sum(1 for d in discs.values() if not any(d["skills"].values()))
        return {"disciplines": total, "skills": skills, "zero": zeros}


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load RPD PDFs and extract KRM data")
    parser.add_argument("--pdf-dir", default="C:/Users/aleyn/OneDrive/Рабочий стол/123")
    parser.add_argument("--output", default="data/reference/krm_disciplines_09.03.02.json")
    parser.add_argument("--discipline", help="Load only one discipline")
    args = parser.parse_args()

    loader = RPDLoader(args.pdf_dir)

    if args.discipline:
        disc = loader.load_discipline(args.discipline)
        if disc:
            print(json.dumps(disc, ensure_ascii=False, indent=2))
        else:
            print(f"Discipline '{args.discipline}' not found")
            sys.exit(1)
    else:
        data = loader.load_all()
        loader.save(data, args.output)
        st = loader.stats(data)
        print(f"\n{st['disciplines']} disciplines, {st['skills']} skills, {st['zero']} zero")
