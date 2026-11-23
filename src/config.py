"""
Централизованная конфигурация приложения на основе pydantic-settings.
Все переменные окружения загружаются и валидируются один раз при старте.
Для обратной совместимости все настройки экспортируются как переменные уровня модуля.
"""

import functools
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Дефолтный (dev) DSN. В prod использовать его запрещено (см. _check_prod_environment).
DEV_DEFAULT_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/compare_competencies"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- пути проекта ----------
    BASE_DIR: Path = Path(__file__).parent.parent

    DATA_DIR: Path = Field(default_factory=lambda: Path("data"))
    DATA_RAW_DIR: Path = Field(default_factory=lambda: Path("data/raw"))
    DATA_PROCESSED_DIR: Path = Field(default_factory=lambda: Path("data/processed"))
    DATA_RESULT_DIR: Path = Field(default_factory=lambda: Path("data/result"))
    REPORTS_DIR: Path = Field(default_factory=lambda: Path("data/result/reports"))
    STUDENTS_DIR: Path = Field(default_factory=lambda: Path("data/students"))
    LAST_UPLOADED_DIR: Path = Field(default_factory=lambda: Path("data/last_uploaded"))
    COMPETENCY_MAPPING_FILE: Path = Field(default_factory=lambda: Path("data/processed/competency_mapping.json"))
    COMPETENCY_FREQ_PATH: Path = Field(default_factory=lambda: Path("data/processed/competency_frequency.json"))

    LOG_DIR: Path = Field(default_factory=lambda: Path("logs"))
    LOG_FILE: Path = Field(default_factory=lambda: Path("logs/backend.log"))

    # ---------- справочные данные (reference) ----------
    REFERENCE_DIR: Path = Field(default_factory=lambda: Path("data/reference"))
    IT_SKILLS_PATH: Path = Field(default_factory=lambda: Path("data/reference/it_skills.json"))
    SKILL_TAXONOMY_PATH: Path = Field(default_factory=lambda: Path("data/reference/skill_taxonomy.json"))
    DOMAIN_MAP_PATH: Path = Field(default_factory=lambda: Path("data/reference/domain_map.json"))
    PROFESSION_TAXONOMY_PATH: Path = Field(default_factory=lambda: Path("data/reference/profession_taxonomy.json"))
    KRM_MAPPING_PATH: Path = Field(default_factory=lambda: Path("data/reference/krm_competency_mapping.json"))
    KRM_DISCIPLINES_PATH: Path = Field(default_factory=lambda: Path("data/reference/krm_disciplines_09.03.02.json"))
    TEACHER_RECOMMENDATIONS_PATH: Path = Field(
        default_factory=lambda: Path("data/reference/teacher_recommendations.json")
    )
    HARD_SKILLS_PATH: Path = Field(default_factory=lambda: Path("data/reference/hard_skills.json"))
    TREND_HOT_SKILLS_PATH: Path = Field(default_factory=lambda: Path("data/reference/trend_hot_skills.json"))
    TIMEFRAME_GROUPS_PATH: Path = Field(default_factory=lambda: Path("data/reference/timeframe_groups.json"))
    SKILL_BLACKLIST_PATH: Path = Field(default_factory=lambda: Path("data/reference/skill_blacklist.json"))
    GENERIC_WORDS_PATH: Path = Field(default_factory=lambda: Path("data/reference/generic_words.json"))
    FILLER_WORDS_PATH: Path = Field(default_factory=lambda: Path("data/reference/filler_words.json"))
    STOP_LEMMAS_PATH: Path = Field(default_factory=lambda: Path("data/reference/stop_lemmas.json"))
    # Смешивание рыночных и научных весов по ролям (экспертные приоры, 0..1).
    ROLE_SOURCE_MIX_PATH: Path = Field(default_factory=lambda: Path("data/reference/role_source_mix.json"))

    # ---------- директории кеша ----------
    DATA_CACHE_DIR: Path = Field(default_factory=lambda: Path("data/cache"))
    EMBEDDINGS_CACHE_DIR: Path = Field(default_factory=lambda: Path("data/cache/embeddings"))
    VACANCY_CLUSTERS_CACHE_DIR: Path = Field(default_factory=lambda: Path("data/cache/clusters"))
    STUDENT_EMB_CACHE_DIR: Path = Field(default_factory=lambda: Path("data/cache/students"))
    PARSED_SKILLS_CACHE_PATH: Path = Field(default_factory=lambda: Path("data/cache/parsed_skills.joblib"))

    MODELS_DIR: Path = Field(default_factory=lambda: Path("data/models"))
    HISTORY_DIR: Path = Field(default_factory=lambda: Path("data/history"))
    # Веса навыков из корпуса научных публикаций (артефакт build-article-weights).
    # Нет файла - научная половина смешивания не применяется (поведение не меняется).
    ARTICLE_WEIGHTS_PATH: Path = Field(default_factory=lambda: Path("data/processed/article_weights.json"))

    # ---------- база данных ----------
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/compare_competencies"
    # Окружение: dev/staging/prod. В prod запрещаем дефолтные (dev) значения,
    # иначе сервис молча подключится к localhost или стартует со слабой конфигурацией.
    APP_ENV: str = "dev"
    DATABASE_ECHO: bool = False

    # ---------- n8n ----------
    N8N_API_KEY: SecretStr | None = None
    N8N_WEBHOOK_SECRET: SecretStr | None = None

    # ---------- уведомления ----------
    TELEGRAM_BOT_TOKEN: SecretStr | None = None

    # ---------- безопасность ----------
    SECRET_KEY: SecretStr | None = None
    # Ограничитель частоты. Выключать только для нагрузочных замеров:
    # иначе результат измеряет работу лимитера, а не сервера.
    RATELIMIT_ENABLED: bool = True
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://localhost:8000"
    TOKEN_TTL_DAYS: int = 7
    # Пул соединений asyncpg. Поля обязаны быть объявлены в Settings: раньше
    # src/db.py читал их через getattr(settings, 'DB_POOL_MIN', 2) — при
    # extra='ignore' значения из .env тихо отбрасывались и пул всегда был 2-10.
    DB_POOL_MIN: int = 2
    DB_POOL_MAX: int = 10
    # Таймаут одного SQL-запроса, с (30 по умолчанию). Держим настраиваемым:
    # тяжёлые аналитические запросы иногда легально идут десятки секунд.
    DB_COMMAND_TIMEOUT: float = 30.0
    # Таймаут ожидания соединения из пула, с: запрос лучше упасть быстро (503),
    # чем висеть в очереди под нагрузкой.
    DB_POOL_ACQUIRE_TIMEOUT: float = 10.0
    # Соединения, простаивавшие дольше, с (5 мин), закрываются - защита от
    # полумёртвых коннектов после сетевых сбоев/рестарта сервера БД.
    DB_POOL_MAX_INACTIVE_LIFETIME: float = 300.0

    # ---------- исходящие HTTP-запросы ----------
    # Единый таймаут/ретраи для внешних вызовов (hh, hub, ...). connect отдельно
    # от read: "медленно соединяется" и "медленно отвечает" - разные сбои.
    # Ретраи - на уровне транспорта httpx (для идемпотентных GET).
    HTTP_CONNECT_TIMEOUT_S: float = 5.0
    HTTP_READ_TIMEOUT_S: float = 30.0
    HTTP_RETRIES: int = 2

    @model_validator(mode="after")
    def _check_db_pool(self) -> "Settings":
        """Fail-fast: некорректная конфигурация пула не должна стартовать молча."""
        if self.DB_POOL_MIN < 1:
            raise ValueError("DB_POOL_MIN должен быть >= 1")
        if self.DB_POOL_MAX < self.DB_POOL_MIN:
            raise ValueError("DB_POOL_MAX должен быть >= DB_POOL_MIN")
        for name in ("DB_COMMAND_TIMEOUT", "DB_POOL_ACQUIRE_TIMEOUT", "DB_POOL_MAX_INACTIVE_LIFETIME"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} должен быть > 0")
        return self

    # Токен для скрейпинга /metrics (Prometheus). Если задан, эндпоинт
    # принимает Authorization: Bearer <токен> в обход ролевой проверки:
    # у Prometheus нет пользовательской сессии, и без этого таргет падал 401.
    METRICS_TOKEN: SecretStr | None = None

    # Версия публичного API (заголовок/эндпоинт /api/version). Меняем при
    # несовместимых изменениях контракта; см. docs/API-CHANGELOG.md.
    API_VERSION: str = "1.0"

    # Grafana: адрес и логин дашбордов для панели мониторинга в админке.
    # Пока URL не задан, эндпоинт мониторинга отвечает enabled=false, а не 500.
    GRAFANA_URL: str | None = None
    # Публичный адрес Grafana для ссылок в браузере (может отличаться от
    # внутреннего GRAFANA_URL, по которому API ходит в /api/search, напр.
    # внутренний http://grafana:3000 vs внешний https://host/grafana).
    GRAFANA_PUBLIC_URL: str | None = None
    GRAFANA_USER: str | None = None
    GRAFANA_PASSWORD: SecretStr | None = None

    MAX_REQUEST_SIZE: int = 20 * 1024 * 1024  # 20 MB

    # ---------- hh.ru API ----------
    HH_USER_AGENT: str = "CompetencyAnalyzer (kok.yoko@gmx.com)"
    REQUEST_DELAY: float = 0.1
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0
    PIPELINE_RETRIES: int = 2

    HH_CLIENT_ID: SecretStr | None = None
    HH_CLIENT_SECRET: SecretStr | None = None
    # Готовый OAuth-токен соискателя. Приоритетнее client credentials:
    # HH закрыл /vacancies для анонимных запросов.
    HH_ACCESS_TOKEN: SecretStr | None = None

    # ---------- rabota.ru API (второй агрегатор вакансий) ----------
    # Публичный поиск вакансий: POST /v5/outer_clients/vacancies/search.json
    # с application_id + time + signature (sha256 от канонического JSON + secret).
    # WAF (BI.ZONE) отбивает короткие User-Agent, поэтому UA по умолчанию полный.
    RABOTA_BASE_URL: str = "https://api.rabota.ru/v5"
    RABOTA_APP_ID: int | None = None
    RABOTA_SECRET: SecretStr | None = None
    RABOTA_USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    # Регион Москвы в нумерации rabota.ru (у hh DEFAULT_AREA=1 - тоже Москва,
    # но id разные: у rabota 3=Москва, 4=СПб, 481=Казань, 328=Екатеринбург).
    RABOTA_DEFAULT_REGION_ID: int = 3
    # Публичный поиск отдаёт не больше 100 записей на страницу и упирается
    # в глубину ~4000 (offset). Страховка от бесконечной пагинации.
    RABOTA_MAX_OFFSET: int = 4000
    RABOTA_PER_PAGE: int = 100
    # Сдвиг id вакансий Работы.ру: в таблице vacancies UNIQUE(hh_id), а id
    # у источников независимые. 1e9 выше любого hh id (сейчас ~1.4e8) и
    # ещё помещается в int4 (2.147e9), т.е. колонка Integer не переполнится.
    RABOTA_ID_OFFSET: int = 1_000_000_000
    # Работа.ру не умеет фильтровать выдачу по IT-профессиям (параметр
    # professional_role игнорируется), поэтому режем локально по дереву
    # professional_areas (ветка 01.* = IT/Интернет/Телеком) + ключевым словам.
    RABOTA_IT_ONLY: bool = True
    # Регионы для планового сбора (нумерация rabota.ru, не hh). У Работы.ру
    # нет эндпоинта со справочником регионов, поэтому список фиксированный:
    # Москва, СПб, Казань, Екатеринбург, Н.Новгород, Самара, Ростов, Уфа,
    # Пермь, Красноярск.
    RABOTA_REGION_IDS: list[int] = [
        3, 4, 481, 328, 231, 299, 284, 529, 267, 25,
    ]
    # Запросы для планового сбора с Работы.ру: у них нет фильтра по профессии,
    # поэтому идём по текстовым IT-запросам, а остаток чистит локальный фильтр.
    RABOTA_SEARCH_QUERIES: list[str] = [
        "Python разработчик",
        "Java разработчик",
        "Frontend разработчик",
        "Backend разработчик",
        "DevOps инженер",
        "QA инженер",
        "системный аналитик",
        "бизнес-аналитик",
        "Data Scientist",
        "Data Engineer",
        "системный администратор",
        "1С программист",
        "информационная безопасность",
        "ML инженер",
    ]
    # Страниц на пару (регион, запрос) в фоновом сборе: 1 = 100 вакансий.
    RABOTA_COLLECT_MAX_PAGES: int = 1

    # ---------- агрегатор источников вакансий ----------
    # hh | rabota | both (единый вход через get_vacancy_feed()).
    VACANCY_SOURCE: str = "hh"

    # ---------- параметры поиска ----------
    DEFAULT_AREA: int = 1
    DEFAULT_PERIOD_DAYS: int = 30
    DEFAULT_MAX_PAGES: int = 20
    DEFAULT_PER_PAGE: int = 100

    # ---------- профили дисциплин ----------
    PROFILES_DISCIPLINES: dict[str, list[int]] = {
        "base": [1, 2, 3, 4, 5, 6, 9, 10, 13],
        "dc": [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 18, 20, 22, 24, 25],
        "top_dc": [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 18, 20, 22, 24, 25, 7, 8, 11, 12, 19, 21, 23],
    }

    # ---------- YandexGPT ----------
    YC_API_KEY: SecretStr | None = None
    YC_FOLDER_ID: str | None = None
    YANDEXGPT_MODEL: str = "yandexgpt-lite"

    # ---------- Ollama ----------
    OLLAMA_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "gpt-oss:120b"
    OLLAMA_EMBEDDING_MODEL: str = "qwen2.5:0.5b"
    QWEN_TEMPERATURE: float = 0.7
    QWEN_MAX_TOKENS: int = 2000

    # ---------- LLM chat client (шлюз reformboss, ключ-1) ----------
    # Слоты подключения отдельного чат-клиента к шлюзу reformboss. URL уже
    # оканчивается на /v1 (шлюз OpenAI-совместим), ключ читается из env
    # LLM_CHAT_API_KEY (необязателен). Поведение Ollama (OLLAMA_URL/Model) не трогаем.
    LLM_CHAT_URL: str = "http://127.0.0.1:8797/k1/v1"
    LLM_CHAT_API_KEY: SecretStr | None = None
    LLM_CHAT_MODEL: str = "deepseek-ai/DeepSeek-V4.1-Flash"

    # ---------- LLM chat assessment (opt-in, deterministic fallback) ----------
    # Точка входа сервиса llm_chat: сборка grounded-контекста + два вызова LLM
    # (текст + структурные блоки). Off = детерминированный шаблон из метрик,
    # без вызова модели.
    LLM_ASSESS_ENABLED: bool = False
    # Модель для структурных блоков (второй вызов). Текст использует LLM_CHAT_MODEL.
    LLM_CHAT_BLOCKS_MODEL: str = "deepseek-ai/DeepSeek-V4-Pro-0813"
    # Бюджет контекста в токенах для промптов обоих вызовов.
    LLM_CHAT_CONTEXT_TOKEN_BUDGET: int = 8000

    # ---------- LLM enhancement (opt-in, fallback-first) ----------
    LLM_ENABLED: bool = True
    LLM_ENHANCE_STUDENT: bool = False
    LLM_ENHANCE_TEACHER: bool = False
    LLM_EXTRACT: bool = False
    LLM_TIMEOUT_S: int = 60

    # ---------- LLM semantic judge (opt-in stub, default off) ----------
    # Судья семантического хвоста: проверяет пары (запрос -> совпадение)
    # из косинусной серой зоны. Только слоты подключения - впиши URL/ключ
    # через env (LLM_JUDGE_URL / LLM_JUDGE_API_KEY / LLM_JUDGE_MODEL),
    # затем включи LLM_JUDGE_ENABLED. Без подключения - тихий skip.
    LLM_JUDGE_ENABLED: bool = False
    LLM_JUDGE_URL: str | None = None
    LLM_JUDGE_API_KEY: SecretStr | None = None
    LLM_JUDGE_MODEL: str | None = None
    LLM_JUDGE_TIMEOUT_S: int = 30
    LLM_JUDGE_MAX_PAIRS: int = 200

    # ---------- Hub-gate (default ON с 2026-10-12) ----------
    # Вето семантических матчей серой зоны к хаб-навыкам (prodval: kill if
    # hub>=200 AND morph<0.2 -> P=0.971/R=0.680; ultra hub>=1000&morph==0 ->
    # 26/0). Замер на проде: vetoed=0, хедлайн бит-в-бит. Откат: False.
    # Таблица: data/reference/market_hubs.json (пересборка при смене маркета).
    HUB_GATE_ENABLED: bool = True
    HUB_GATE_HUB_T: int = 200
    HUB_GATE_MORPH_M: float = 0.2
    HUB_GATE_ULTRA_HUB: int = 1000
    HUB_GATE_SIM_LO: float = 0.6
    HUB_GATE_SIM_HI: float = 0.75
    HUB_TABLE_PATH: Path = Path("data/reference/market_hubs.json")

    # ---------- эмбеддинги ----------
    EMBEDDING_MODEL: str = "paraphrase-multilingual-mpnet-base-v2"
    HF_TOKEN: SecretStr | None = None

    # ---------- cosine similarity scale ----------
    # One ladder for every "are these the same thing" decision. Domain-specific
    # cutoffs (academic gap, discipline tiers) stay next to their analyzers but
    # must be read against these values.
    SIM_DEDUP: float = 0.82            # near-duplicate texts (dedup clusters)
    SIM_SKILL_MATCH: float = 0.78      # RPD skill <-> market skill
    SIM_MARKET_MATCH_HIGH: float = 0.75
    SIM_MARKET_MATCH_MEDIUM: float = 0.65
    SIM_ACADEMIC_GAP: float = 0.6   # prodval 2026-10-12 (teacher threshold)
    SIM_ACADEMIC_MARKET: float = 0.45

    # ---------- BM25 ----------
    BM25_MAX_CORPUS_DOCS: int = 200
    BM25_MIN_SCORE: float = 0.005

    # ---------- PCA ----------
    PCA_ENABLED: bool = True
    PCA_TARGET_DIM: int = 256
    PCA_MIN_SAMPLES: int = 100
    PCA_MIN_FEATURES: int = 128

    # ---------- воспроизводимость ----------
    GLOBAL_RANDOM_SEED: int = 42

    # ---------- Параметры рекомендаций и gap-анализа ----------
    # Blend of the profile evaluator (personalisation) and the LTR ranker.
    # Frozen-split eval (scripts/eval_frozen.py, gold.csv, seed 42) gives LTR
    # alone NDCG@10 0.7455 vs frequency 0.5227 and cosine 0.4978, so the ranker
    # carries the majority. The evaluator keeps 0.4 because its per-student
    # signal cannot be scored by a role-level gold set (no student profiles in
    # the DB), and dropping it would remove personalisation entirely.
    BLEND_EVALUATOR_WEIGHT: float = 0.4
    BLEND_LTR_WEIGHT: float = 0.6
    DOMAIN_BONUS: float = 0.1
    DIVERSIFY_MAX_PER_CATEGORY: int = 3
    PRIORITY_HIGH_THRESHOLD: float = 0.4
    PRIORITY_MEDIUM_THRESHOLD: float = 0.2
    TREND_ALWAYS_HOT_BONUS: float = 0.15

    # ---------- смешивание источников весов по ролям (v1) ----------
    # lambda = доля научных (публикации) весов в смеси: 0 - только вакансии,
    # 1 - только публикации. Значения по ролям - экспертные приоры в
    # data/reference/role_source_mix.json (не откалиброваны на трудоустройстве,
    # см. docs/methodology.md). Флаг выключает фичу целиком, не трогая файлы.
    ROLE_SOURCE_MIX_ENABLED: bool = True

    READINESS_MARKET_WEIGHT: float = 0.45
    READINESS_SKILL_WEIGHT: float = 0.30
    READINESS_DOMAIN_WEIGHT: float = 0.0
    READINESS_GAP_PENALTY_WEIGHT: float = -0.25

    SKILL_STRONG_GAP_THRESHOLD: float = 0.2
    SKILL_WEAK_GAP_THRESHOLD: float = 0.6

    DOMINANT_DOMAIN_WEIGHT: float = 0.5

    LEVEL_WEIGHTS_MAP: dict = {
        "student": {"junior": 0.60, "middle": 0.30, "senior": 0.10},
        "junior": {"junior": 0.40, "middle": 0.40, "senior": 0.20},
        "middle": {"junior": 0.20, "middle": 0.50, "senior": 0.30},
    }

    GAP_ANALYZER_FALLBACK_MIN_GAP: float = 0.05
    GAP_ANALYZER_FALLBACK_REDUCTION: float = 0.65

    TQDM_DISABLE: bool = False
    PYDANTIC_VALIDATION_ENABLED: bool = True

    # фоновый сбор вакансий (legacy seed; интервал и вкл/выкл - SCHEDULER_* и админка)
    BACKGROUND_COLLECTOR_ENABLED: bool = False
    # планировщик фоновых задач: seed для data/settings/scheduler.json;
    # рантайм-управление (вкл/выкл, интервал) - через админку, не через env
    SCHEDULER_COLLECT_INTERVAL_HOURS: int = 12
    SCHEDULER_DAILY_GAP_ENABLED: bool = False

    # ---------- academic-api (ЮФУ: компетенции/разрывы по научным темам) ----------
    ACADEMIC_API_BASE: str = "https://academic-api.lib.sfedu.ru"
    # Отдельный хост для Course Trends API (предрассчитанные рекомендации и
    # статьи по программам). Старый API остаётся для /get-competencies и
    # /analyze-gap - переносить их на новый хост нельзя, там таких методов нет.
    ACADEMIC_TRENDS_API_BASE: str = "https://academic-api-dev.lib.sfedu.ru"
    ACADEMIC_EMAIL_DOMAIN: str = "sfedu.ru"
    ACADEMIC_SSO_ENABLED: bool = True

    # ---------- двухфакторная аутентификация (TOTP) ----------
    # Глобальный выключатель: пока False, второй фактор не запрашивается ни при
    # входе, ни при SSO, даже если пользователь его настроил. Так фича не мешает
    # работе, пока её не включат осознанно. Значение по умолчанию - выключено.
    TOTP_2FA_ENABLED: bool = False
    # Ставить сессионную cookie с флагом Secure. Включать на HTTPS-деплое:
    # на http (localhost) браузер такую cookie не сохранит.
    COOKIE_SECURE: bool = False

    # ---------- hub.sfedu.ru: публикации (публичный API хаба ЮФУ) ----------
    HUB_API_BASE: str = "https://hub.sfedu.ru"
    # Токен нужен только для карточки /api/assets/{id}/; список публикаций открыт.
    HUB_API_TOKEN: SecretStr | None = None
    HUB_API_TIMEOUT_S: float = 30.0
    # TTL кэша списка/типов публикаций; 0 — без кэша.
    HUB_CACHE_TTL_S: float = 120.0

    # ---------- RAG (retrieval over РПД / вакансии / публикации) ----------
    # Opt-in: при False retrieval не вызывается, llm_chat работает как раньше.
    RAG_ENABLED: bool = False
    # Профиль retrieval по умолчанию (chat_qa, rpd_gap_explain, market_evidence,
    # curriculum_design, competency_lookup).
    RAG_PROFILE_DEFAULT: str = "chat_qa"
    RAG_DENSE_K: int = 50
    RAG_SPARSE_K: int = 50
    RAG_ENTITY_K: int = 20
    RAG_TOP_N: int = 8
    # Минимальный top-score; ниже — отказ от RAG-контекста (fallback на факты).
    RAG_MIN_SCORE: float = 0.35
    RAG_CONTEXT_BUDGET: int = 8000
    RAG_TIMEOUT_S: float = 1.5
    RAG_INDEX_BATCH: int = 64
    RAG_MMR_LAMBDA: float = 0.7
    RAG_MAX_CHUNK_TOKENS: int = 700
    RAG_MIN_CHUNK_TOKENS: int = 40
    RAG_ACL_DEFAULT: str = "internal"
    # L1-кэш ответов retrieval, секунды; 0 - без кэша.
    RAG_CACHE_TTL_S: float = 60.0

    # валидация путей относительно BASE_DIR
    @field_validator(
        "DATA_DIR",
        "DATA_RAW_DIR",
        "DATA_PROCESSED_DIR",
        "DATA_RESULT_DIR",
        "REPORTS_DIR",
        "STUDENTS_DIR",
        "LAST_UPLOADED_DIR",
        "COMPETENCY_MAPPING_FILE",
        "COMPETENCY_FREQ_PATH",
        "REFERENCE_DIR",
        "IT_SKILLS_PATH",
        "SKILL_TAXONOMY_PATH",
        "DOMAIN_MAP_PATH",
        "PROFESSION_TAXONOMY_PATH",
        "KRM_MAPPING_PATH",
        "KRM_DISCIPLINES_PATH",
        "TEACHER_RECOMMENDATIONS_PATH",
        "HARD_SKILLS_PATH",
        "TREND_HOT_SKILLS_PATH",
        "TIMEFRAME_GROUPS_PATH",
        "SKILL_BLACKLIST_PATH",
        "GENERIC_WORDS_PATH",
        "FILLER_WORDS_PATH",
        "STOP_LEMMAS_PATH",
        "ROLE_SOURCE_MIX_PATH",
        "MODELS_DIR",
        "HISTORY_DIR",
        "ARTICLE_WEIGHTS_PATH",
        "LOG_DIR",
        "LOG_FILE",
        "DATA_CACHE_DIR",
        "EMBEDDINGS_CACHE_DIR",
        "VACANCY_CLUSTERS_CACHE_DIR",
        "STUDENT_EMB_CACHE_DIR",
        "PARSED_SKILLS_CACHE_PATH",
        "HUB_TABLE_PATH",
        mode="after",
    )
    @classmethod
    def make_absolute(cls, v: Path, info) -> Path:
        base = info.data.get("BASE_DIR", Path("."))
        return base / v

    @field_validator("SECRET_KEY", mode="after")
    @classmethod
    def validate_secret_key(cls, v: SecretStr | None) -> SecretStr:
        hint = "SECRET_KEY=$(openssl rand -hex 32)"
        if v is None or not v.get_secret_value():
            raise ValueError(f"SECRET_KEY не задан. Укажите его в .env: {hint}")
        key = v.get_secret_value()
        # Ключом подписываются сессионные токены: слабый/дефолтный ключ = подделка
        # чужих токенов. Ловим и заглушки, и просто короткие значения.
        low = key.lower()
        looks_placeholder = (
            low.startswith(("dev-", "dev_"))
            or any(
                x in low
                for x in ("change", "changeme", "placeholder", "insecure", "default", "example")
            )
        )
        if looks_placeholder or len(key) < 32:
            raise ValueError(
                "SECRET_KEY слабый или значение-заглушка (нужно ≥32 случайных символов, "
                f"без 'dev'/'change'/'insecure' и т.п.). Замените: {hint}"
            )
        return v

    @model_validator(mode="after")
    def _check_prod_environment(self) -> "Settings":
        """Fail-fast в prod: дефолтные (dev) значения недопустимы."""
        env = (self.APP_ENV or "dev").strip().lower()
        if env in {"prod", "production"} and self.DATABASE_URL.strip() == DEV_DEFAULT_DATABASE_URL:
            raise ValueError(
                "APP_ENV=prod, но DATABASE_URL оставлен дефолтным (localhost). "
                "Задайте реальный DATABASE_URL в .env"
            )
        return self

    def ensure_dirs(self):
        dirs = [
            self.DATA_RAW_DIR,
            self.DATA_PROCESSED_DIR,
            self.STUDENTS_DIR,
            self.LAST_UPLOADED_DIR,
            self.LOG_DIR,
            self.MODELS_DIR,
            self.HISTORY_DIR,
            self.DATA_RESULT_DIR,
            self.REPORTS_DIR,
            self.REFERENCE_DIR,
            self.DATA_CACHE_DIR,
            self.EMBEDDINGS_CACHE_DIR,
            self.VACANCY_CLUSTERS_CACHE_DIR,
            self.STUDENT_EMB_CACHE_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Синглтон настроек
# ---------------------------------------------------------------------------
settings = Settings()

# ---------------------------------------------------------------------------
# Экспорт всех параметров для обратной совместимости
# ---------------------------------------------------------------------------
BASE_DIR = settings.BASE_DIR
DATA_DIR = settings.DATA_DIR
DATA_RAW_DIR = settings.DATA_RAW_DIR
DATA_PROCESSED_DIR = settings.DATA_PROCESSED_DIR
DATA_RESULT_DIR = settings.DATA_RESULT_DIR
REPORTS_DIR = settings.REPORTS_DIR
STUDENTS_DIR = settings.STUDENTS_DIR
LAST_UPLOADED_DIR = settings.LAST_UPLOADED_DIR
COMPETENCY_MAPPING_FILE = settings.COMPETENCY_MAPPING_FILE
COMPETENCY_FREQ_PATH = settings.COMPETENCY_FREQ_PATH
IT_SKILLS_PATH = settings.IT_SKILLS_PATH
MODELS_DIR = settings.MODELS_DIR
HISTORY_DIR = settings.HISTORY_DIR
LOG_DIR = settings.LOG_DIR
LOG_FILE = settings.LOG_FILE

DATA_CACHE_DIR = settings.DATA_CACHE_DIR
EMBEDDINGS_CACHE_DIR = settings.EMBEDDINGS_CACHE_DIR
VACANCY_CLUSTERS_CACHE_DIR = settings.VACANCY_CLUSTERS_CACHE_DIR
STUDENT_EMB_CACHE_DIR = settings.STUDENT_EMB_CACHE_DIR
PARSED_SKILLS_CACHE_PATH = settings.PARSED_SKILLS_CACHE_PATH

ALLOWED_ORIGINS = settings.ALLOWED_ORIGINS
MAX_REQUEST_SIZE = settings.MAX_REQUEST_SIZE
TOKEN_TTL_DAYS = settings.TOKEN_TTL_DAYS
METRICS_TOKEN = settings.METRICS_TOKEN
GRAFANA_URL = settings.GRAFANA_URL
GRAFANA_PUBLIC_URL = settings.GRAFANA_PUBLIC_URL
API_VERSION = settings.API_VERSION
GRAFANA_USER = settings.GRAFANA_USER
GRAFANA_PASSWORD = settings.GRAFANA_PASSWORD

KRM_DISCIPLINES_PATH = settings.KRM_DISCIPLINES_PATH
TEACHER_RECOMMENDATIONS_PATH = settings.TEACHER_RECOMMENDATIONS_PATH
@functools.lru_cache
def get_secret_key() -> str:
    """Lazy-load SECRET_KEY - не падает при импорте если .env отсутствует."""
    return str(settings.SECRET_KEY.get_secret_value()) if settings.SECRET_KEY else ""
HH_USER_AGENT = settings.HH_USER_AGENT
REQUEST_DELAY = settings.REQUEST_DELAY
MAX_RETRIES = settings.MAX_RETRIES
RETRY_DELAY = settings.RETRY_DELAY
PIPELINE_RETRIES = settings.PIPELINE_RETRIES
HH_CLIENT_ID = settings.HH_CLIENT_ID
HH_CLIENT_SECRET = settings.HH_CLIENT_SECRET
HH_ACCESS_TOKEN = settings.HH_ACCESS_TOKEN

RABOTA_BASE_URL = settings.RABOTA_BASE_URL
RABOTA_APP_ID = settings.RABOTA_APP_ID
RABOTA_SECRET = settings.RABOTA_SECRET
RABOTA_USER_AGENT = settings.RABOTA_USER_AGENT
RABOTA_DEFAULT_REGION_ID = settings.RABOTA_DEFAULT_REGION_ID
RABOTA_MAX_OFFSET = settings.RABOTA_MAX_OFFSET
RABOTA_PER_PAGE = settings.RABOTA_PER_PAGE
RABOTA_ID_OFFSET = settings.RABOTA_ID_OFFSET
RABOTA_IT_ONLY = settings.RABOTA_IT_ONLY
RABOTA_REGION_IDS = settings.RABOTA_REGION_IDS
RABOTA_SEARCH_QUERIES = settings.RABOTA_SEARCH_QUERIES
RABOTA_COLLECT_MAX_PAGES = settings.RABOTA_COLLECT_MAX_PAGES

VACANCY_SOURCE = settings.VACANCY_SOURCE

DEFAULT_AREA = settings.DEFAULT_AREA
DEFAULT_PERIOD_DAYS = settings.DEFAULT_PERIOD_DAYS
DEFAULT_MAX_PAGES = settings.DEFAULT_MAX_PAGES
DEFAULT_PER_PAGE = settings.DEFAULT_PER_PAGE

PROFILES_DISCIPLINES = settings.PROFILES_DISCIPLINES

YC_API_KEY = settings.YC_API_KEY
YC_FOLDER_ID = settings.YC_FOLDER_ID
YANDEXGPT_MODEL = settings.YANDEXGPT_MODEL

OLLAMA_URL = settings.OLLAMA_URL
OLLAMA_MODEL = settings.OLLAMA_MODEL
OLLAMA_EMBEDDING_MODEL = settings.OLLAMA_EMBEDDING_MODEL
QWEN_TEMPERATURE = settings.QWEN_TEMPERATURE
QWEN_MAX_TOKENS = settings.QWEN_MAX_TOKENS

LLM_CHAT_URL = settings.LLM_CHAT_URL
LLM_CHAT_API_KEY = settings.LLM_CHAT_API_KEY
LLM_CHAT_MODEL = settings.LLM_CHAT_MODEL
LLM_ASSESS_ENABLED = settings.LLM_ASSESS_ENABLED
LLM_CHAT_BLOCKS_MODEL = settings.LLM_CHAT_BLOCKS_MODEL
LLM_CHAT_CONTEXT_TOKEN_BUDGET = settings.LLM_CHAT_CONTEXT_TOKEN_BUDGET

LLM_ENABLED = settings.LLM_ENABLED
LLM_ENHANCE_STUDENT = settings.LLM_ENHANCE_STUDENT
LLM_ENHANCE_TEACHER = settings.LLM_ENHANCE_TEACHER
LLM_EXTRACT = settings.LLM_EXTRACT
LLM_TIMEOUT_S = settings.LLM_TIMEOUT_S

LLM_JUDGE_ENABLED = settings.LLM_JUDGE_ENABLED
LLM_JUDGE_URL = settings.LLM_JUDGE_URL
LLM_JUDGE_API_KEY = settings.LLM_JUDGE_API_KEY
LLM_JUDGE_MODEL = settings.LLM_JUDGE_MODEL
LLM_JUDGE_TIMEOUT_S = settings.LLM_JUDGE_TIMEOUT_S
LLM_JUDGE_MAX_PAIRS = settings.LLM_JUDGE_MAX_PAIRS

HUB_GATE_ENABLED = settings.HUB_GATE_ENABLED
HUB_GATE_HUB_T = settings.HUB_GATE_HUB_T
HUB_GATE_MORPH_M = settings.HUB_GATE_MORPH_M
HUB_GATE_ULTRA_HUB = settings.HUB_GATE_ULTRA_HUB
HUB_GATE_SIM_LO = settings.HUB_GATE_SIM_LO
HUB_GATE_SIM_HI = settings.HUB_GATE_SIM_HI
HUB_TABLE_PATH = settings.HUB_TABLE_PATH

EMBEDDING_MODEL = settings.EMBEDDING_MODEL
HF_TOKEN = settings.HF_TOKEN

SIM_DEDUP = settings.SIM_DEDUP
SIM_SKILL_MATCH = settings.SIM_SKILL_MATCH
SIM_MARKET_MATCH_HIGH = settings.SIM_MARKET_MATCH_HIGH
SIM_MARKET_MATCH_MEDIUM = settings.SIM_MARKET_MATCH_MEDIUM
SIM_ACADEMIC_GAP = settings.SIM_ACADEMIC_GAP
SIM_ACADEMIC_MARKET = settings.SIM_ACADEMIC_MARKET

BM25_MAX_CORPUS_DOCS = settings.BM25_MAX_CORPUS_DOCS
BM25_MIN_SCORE = settings.BM25_MIN_SCORE

PCA_ENABLED = settings.PCA_ENABLED
PCA_TARGET_DIM = settings.PCA_TARGET_DIM
PCA_MIN_SAMPLES = settings.PCA_MIN_SAMPLES
PCA_MIN_FEATURES = settings.PCA_MIN_FEATURES

GLOBAL_RANDOM_SEED = settings.GLOBAL_RANDOM_SEED

REFERENCE_DIR = settings.REFERENCE_DIR
SKILL_TAXONOMY_PATH = settings.SKILL_TAXONOMY_PATH
DOMAIN_MAP_PATH = settings.DOMAIN_MAP_PATH
PROFESSION_TAXONOMY_PATH = settings.PROFESSION_TAXONOMY_PATH
KRM_MAPPING_PATH = settings.KRM_MAPPING_PATH
HARD_SKILLS_PATH = settings.HARD_SKILLS_PATH
TREND_HOT_SKILLS_PATH = settings.TREND_HOT_SKILLS_PATH
TIMEFRAME_GROUPS_PATH = settings.TIMEFRAME_GROUPS_PATH
SKILL_BLACKLIST_PATH = settings.SKILL_BLACKLIST_PATH
GENERIC_WORDS_PATH = settings.GENERIC_WORDS_PATH
FILLER_WORDS_PATH = settings.FILLER_WORDS_PATH
STOP_LEMMAS_PATH = settings.STOP_LEMMAS_PATH
ROLE_SOURCE_MIX_ENABLED = settings.ROLE_SOURCE_MIX_ENABLED
ROLE_SOURCE_MIX_PATH = settings.ROLE_SOURCE_MIX_PATH
ARTICLE_WEIGHTS_PATH = settings.ARTICLE_WEIGHTS_PATH

BLEND_EVALUATOR_WEIGHT = settings.BLEND_EVALUATOR_WEIGHT
BLEND_LTR_WEIGHT = settings.BLEND_LTR_WEIGHT
DOMAIN_BONUS = settings.DOMAIN_BONUS
DIVERSIFY_MAX_PER_CATEGORY = settings.DIVERSIFY_MAX_PER_CATEGORY
PRIORITY_HIGH_THRESHOLD = settings.PRIORITY_HIGH_THRESHOLD
PRIORITY_MEDIUM_THRESHOLD = settings.PRIORITY_MEDIUM_THRESHOLD
TREND_ALWAYS_HOT_BONUS = settings.TREND_ALWAYS_HOT_BONUS
READINESS_MARKET_WEIGHT = settings.READINESS_MARKET_WEIGHT
READINESS_SKILL_WEIGHT = settings.READINESS_SKILL_WEIGHT
READINESS_DOMAIN_WEIGHT = settings.READINESS_DOMAIN_WEIGHT
READINESS_GAP_PENALTY_WEIGHT = settings.READINESS_GAP_PENALTY_WEIGHT
SKILL_STRONG_GAP_THRESHOLD = settings.SKILL_STRONG_GAP_THRESHOLD
SKILL_WEAK_GAP_THRESHOLD = settings.SKILL_WEAK_GAP_THRESHOLD
DOMINANT_DOMAIN_WEIGHT = settings.DOMINANT_DOMAIN_WEIGHT
LEVEL_WEIGHTS_MAP = settings.LEVEL_WEIGHTS_MAP
GAP_ANALYZER_FALLBACK_MIN_GAP = settings.GAP_ANALYZER_FALLBACK_MIN_GAP
GAP_ANALYZER_FALLBACK_REDUCTION = settings.GAP_ANALYZER_FALLBACK_REDUCTION
N8N_API_KEY = settings.N8N_API_KEY
N8N_WEBHOOK_SECRET = settings.N8N_WEBHOOK_SECRET
TELEGRAM_BOT_TOKEN = settings.TELEGRAM_BOT_TOKEN
TQDM_DISABLE = settings.TQDM_DISABLE
PYDANTIC_VALIDATION_ENABLED = settings.PYDANTIC_VALIDATION_ENABLED

ACADEMIC_API_BASE = settings.ACADEMIC_API_BASE
ACADEMIC_TRENDS_API_BASE = settings.ACADEMIC_TRENDS_API_BASE
ACADEMIC_EMAIL_DOMAIN = settings.ACADEMIC_EMAIL_DOMAIN
ACADEMIC_SSO_ENABLED = settings.ACADEMIC_SSO_ENABLED
TOTP_2FA_ENABLED = settings.TOTP_2FA_ENABLED
COOKIE_SECURE = settings.COOKIE_SECURE

HUB_API_BASE = settings.HUB_API_BASE
HUB_API_TOKEN = settings.HUB_API_TOKEN
HUB_API_TIMEOUT_S = settings.HUB_API_TIMEOUT_S
HUB_CACHE_TTL_S = settings.HUB_CACHE_TTL_S

RAG_ENABLED = settings.RAG_ENABLED
RAG_PROFILE_DEFAULT = settings.RAG_PROFILE_DEFAULT
RAG_DENSE_K = settings.RAG_DENSE_K
RAG_SPARSE_K = settings.RAG_SPARSE_K
RAG_ENTITY_K = settings.RAG_ENTITY_K
RAG_TOP_N = settings.RAG_TOP_N
RAG_MIN_SCORE = settings.RAG_MIN_SCORE
RAG_CONTEXT_BUDGET = settings.RAG_CONTEXT_BUDGET
RAG_TIMEOUT_S = settings.RAG_TIMEOUT_S
RAG_INDEX_BATCH = settings.RAG_INDEX_BATCH
RAG_MMR_LAMBDA = settings.RAG_MMR_LAMBDA
RAG_MAX_CHUNK_TOKENS = settings.RAG_MAX_CHUNK_TOKENS
RAG_MIN_CHUNK_TOKENS = settings.RAG_MIN_CHUNK_TOKENS
RAG_ACL_DEFAULT = settings.RAG_ACL_DEFAULT
RAG_CACHE_TTL_S = settings.RAG_CACHE_TTL_S
