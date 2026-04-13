from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ====================== YandexGPT API ======================
YC_API_KEY = os.getenv("YC_API_KEY")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")
YANDEXGPT_MODEL = os.getenv("YANDEXGPT_MODEL", "yandexgpt-lite")

# ====================== Пути проекта ======================
BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
STUDENTS_DIR = DATA_DIR / "students"
LAST_UPLOADED_DIR = DATA_DIR / "last_uploaded"
COMPETENCY_MAPPING_FILE = DATA_PROCESSED_DIR / "competency_mapping.json"
COMPETENCY_FREQ_PATH = DATA_PROCESSED_DIR / "competency_frequency.json"
IT_SKILLS_PATH = DATA_DIR / "it_skills.json"
MODELS_DIR = DATA_DIR / "models"
HISTORY_DIR = DATA_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

IT_SKILLS_FILE = DATA_DIR / "it_skills.json"

for dir_path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, STUDENTS_DIR, LAST_UPLOADED_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ====================== hh.ru API ======================
HH_USER_AGENT = os.getenv("HH_USER_AGENT", "CompetencyAnalyzer/1.0 (opik@sfedu.ru)")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "2"))

# ====================== Параметры поиска по умолчанию ======================
DEFAULT_AREA = int(os.getenv("DEFAULT_AREA", "1"))
DEFAULT_PERIOD_DAYS = int(os.getenv("DEFAULT_PERIOD_DAYS", "30"))
DEFAULT_MAX_PAGES = int(os.getenv("DEFAULT_MAX_PAGES", "20"))
DEFAULT_PER_PAGE = int(os.getenv("DEFAULT_PER_PAGE", "100"))

# ====================== Профили дисциплин ======================
PROFILES_DISCIPLINES = {
    "base": [1, 2, 3, 4, 5, 6, 9, 10, 13],
    "dc": [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 18, 20, 22, 24, 25],
    "top_dc": [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 18, 20, 22, 24, 25, 7, 8, 11, 12, 19, 21, 23],
}

# ====================== Эмбеддинги ======================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DIM = 384
DATA_EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DATA_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_CACHE_DIR = DATA_EMBEDDINGS_DIR / "cache"
EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))