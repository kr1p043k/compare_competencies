"""
Централизованная конфигурация приложения на основе pydantic-settings.
Все переменные окружения загружаются и валидируются один раз при старте.
Для обратной совместимости все настройки экспортируются как переменные уровня модуля.
"""

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Модель настроек
# ---------------------------------------------------------------------------
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
    STUDENTS_DIR: Path = Field(default_factory=lambda: Path("data/students"))
    LAST_UPLOADED_DIR: Path = Field(default_factory=lambda: Path("data/last_uploaded"))
    COMPETENCY_MAPPING_FILE: Path = Field(default_factory=lambda: Path("data/processed/competency_mapping.json"))
    COMPETENCY_FREQ_PATH: Path = Field(default_factory=lambda: Path("data/processed/competency_frequency.json"))
    IT_SKILLS_PATH: Path = Field(default_factory=lambda: Path("data/it_skills.json"))
    MODELS_DIR: Path = Field(default_factory=lambda: Path("data/models"))
    HISTORY_DIR: Path = Field(default_factory=lambda: Path("data/history"))

    LOG_DIR: Path = Field(default_factory=lambda: Path("logs"))
    LOG_FILE: Path = Field(default_factory=lambda: Path("logs/app.log"))

    DATA_EMBEDDINGS_DIR: Path = Field(default_factory=lambda: Path("data/embeddings"))
    EMBEDDINGS_CACHE_DIR: Path = Field(default_factory=lambda: Path("data/embeddings/cache"))

    # ---------- hh.ru API ----------
    HH_USER_AGENT: str = "CompetencyAnalyzer (kok.yoko@gmx.com)"
    REQUEST_DELAY: float = 0.5
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0

    HH_CLIENT_ID: SecretStr | None = None
    HH_CLIENT_SECRET: SecretStr | None = None

    # ---------- параметры поиска по умолчанию ----------
    DEFAULT_AREA: int = 1
    DEFAULT_PERIOD_DAYS: int = 30
    DEFAULT_MAX_PAGES: int = 20
    DEFAULT_PER_PAGE: int = 100

    # ---------- профили дисциплин ----------
    PROFILES_DISCIPLINES: dict = {
        "base": [1, 2, 3, 4, 5, 6, 9, 10, 13],
        "dc": [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 18, 20, 22, 24, 25],
        "top_dc": [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 18, 20, 22, 24, 25, 7, 8, 11, 12, 19, 21, 23],
    }

    # ---------- YandexGPT ----------
    YC_API_KEY: SecretStr | None = None
    YC_FOLDER_ID: str | None = None
    YANDEXGPT_MODEL: str = "yandexgpt-lite"

    # ---------- эмбеддинги ----------
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_DIM: int = 384
    HF_TOKEN: SecretStr | None = None
    SIMILARITY_THRESHOLD: float = 0.80

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

    # валидация путей относительно BASE_DIR
    @field_validator(
        "DATA_DIR",
        "DATA_RAW_DIR",
        "DATA_PROCESSED_DIR",
        "DATA_RESULT_DIR",
        "STUDENTS_DIR",
        "LAST_UPLOADED_DIR",
        "COMPETENCY_MAPPING_FILE",
        "COMPETENCY_FREQ_PATH",
        "IT_SKILLS_PATH",
        "MODELS_DIR",
        "HISTORY_DIR",
        "LOG_DIR",
        "LOG_FILE",
        "DATA_EMBEDDINGS_DIR",
        "EMBEDDINGS_CACHE_DIR",
        mode="after",
    )
    @classmethod
    def make_absolute(cls, v: Path, info) -> Path:
        base = info.data.get("BASE_DIR", Path("."))
        return base / v

    # создание всех директорий при инициализации
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dirs = [
            self.DATA_RAW_DIR,
            self.DATA_PROCESSED_DIR,
            self.STUDENTS_DIR,
            self.LAST_UPLOADED_DIR,
            self.LOG_DIR,
            self.MODELS_DIR,
            self.HISTORY_DIR,
            self.DATA_EMBEDDINGS_DIR,
            self.EMBEDDINGS_CACHE_DIR,
            self.DATA_RESULT_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        # ---------- Параметры рекомендаций и gap-анализа ----------

    BLEND_EVALUATOR_WEIGHT: float = 0.6
    BLEND_LTR_WEIGHT: float = 0.4
    DOMAIN_BONUS: float = 0.1
    DIVERSIFY_MAX_PER_CATEGORY: int = 3
    PRIORITY_HIGH_THRESHOLD: float = 0.7
    PRIORITY_MEDIUM_THRESHOLD: float = 0.4
    TREND_ALWAYS_HOT_BONUS: float = 0.15

    # readiness
    READINESS_MARKET_WEIGHT: float = 0.50
    READINESS_SKILL_WEIGHT: float = 0.20
    READINESS_DOMAIN_WEIGHT: float = 0.15
    READINESS_GAP_PENALTY_WEIGHT: float = 0.10

    # категории навыков
    SKILL_STRONG_GAP_THRESHOLD: float = 0.2
    SKILL_WEAK_GAP_THRESHOLD: float = 0.6

    # доменный анализ
    DOMINANT_DOMAIN_WEIGHT: float = 0.5

    # уровень анализа
    LEVEL_WEIGHTS_MAP: dict = {
        "student": {"junior": 0.60, "middle": 0.30, "senior": 0.10},
        "junior": {"junior": 0.40, "middle": 0.40, "senior": 0.20},
        "middle": {"junior": 0.20, "middle": 0.50, "senior": 0.30},
    }

    # gap-analyzer
    GAP_ANALYZER_FALLBACK_MIN_GAP: float = 0.05
    GAP_ANALYZER_FALLBACK_REDUCTION: float = 0.65

    # tqdm
    TQDM_DISABLE: bool = False


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
STUDENTS_DIR = settings.STUDENTS_DIR
LAST_UPLOADED_DIR = settings.LAST_UPLOADED_DIR
COMPETENCY_MAPPING_FILE = settings.COMPETENCY_MAPPING_FILE
COMPETENCY_FREQ_PATH = settings.COMPETENCY_FREQ_PATH
IT_SKILLS_PATH = settings.IT_SKILLS_PATH
MODELS_DIR = settings.MODELS_DIR
HISTORY_DIR = settings.HISTORY_DIR
LOG_DIR = settings.LOG_DIR
LOG_FILE = settings.LOG_FILE
DATA_EMBEDDINGS_DIR = settings.DATA_EMBEDDINGS_DIR
EMBEDDINGS_CACHE_DIR = settings.EMBEDDINGS_CACHE_DIR

HH_USER_AGENT = settings.HH_USER_AGENT
REQUEST_DELAY = settings.REQUEST_DELAY
MAX_RETRIES = settings.MAX_RETRIES
RETRY_DELAY = settings.RETRY_DELAY
HH_CLIENT_ID = settings.HH_CLIENT_ID  # Теперь SecretStr
HH_CLIENT_SECRET = settings.HH_CLIENT_SECRET  # SecretStr

DEFAULT_AREA = settings.DEFAULT_AREA
DEFAULT_PERIOD_DAYS = settings.DEFAULT_PERIOD_DAYS
DEFAULT_MAX_PAGES = settings.DEFAULT_MAX_PAGES
DEFAULT_PER_PAGE = settings.DEFAULT_PER_PAGE

PROFILES_DISCIPLINES = settings.PROFILES_DISCIPLINES

YC_API_KEY = settings.YC_API_KEY  # SecretStr
YC_FOLDER_ID = settings.YC_FOLDER_ID
YANDEXGPT_MODEL = settings.YANDEXGPT_MODEL

EMBEDDING_MODEL = settings.EMBEDDING_MODEL
EMBEDDING_DIM = settings.EMBEDDING_DIM
HF_TOKEN = settings.HF_TOKEN  # SecretStr
SIMILARITY_THRESHOLD = settings.SIMILARITY_THRESHOLD

BM25_MAX_CORPUS_DOCS = settings.BM25_MAX_CORPUS_DOCS
BM25_MIN_SCORE = settings.BM25_MIN_SCORE

PCA_ENABLED = settings.PCA_ENABLED
PCA_TARGET_DIM = settings.PCA_TARGET_DIM
PCA_MIN_SAMPLES = settings.PCA_MIN_SAMPLES
PCA_MIN_FEATURES = settings.PCA_MIN_FEATURES

GLOBAL_RANDOM_SEED = settings.GLOBAL_RANDOM_SEED

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
TQDM_DISABLE = settings.TQDM_DISABLE
