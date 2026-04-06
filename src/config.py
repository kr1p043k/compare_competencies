from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Основные директории данных
DATA_DIR = BASE_DIR / "data"                     
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
STUDENTS_DIR = DATA_DIR / "students"              
LAST_UPLOADED_DIR = DATA_DIR / "last_uploaded"
COMPETENCY_MAPPING_FILE = DATA_PROCESSED_DIR / "competency_mapping.json"    

# Логи
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

# Файл с IT-навыками
IT_SKILLS_FILE = DATA_DIR / "it_skills.json"

for dir_path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, STUDENTS_DIR, LAST_UPLOADED_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Настройки API hh.ru
HH_USER_AGENT = "CompetencyAnalyzer/1.0 (opik@sfedu.ru)"
REQUEST_DELAY = 0.5
MAX_RETRIES = 3
RETRY_DELAY = 2

# Параметры поиска
DEFAULT_AREA = 76
DEFAULT_PERIOD_DAYS = 30
DEFAULT_MAX_PAGES = 50
DEFAULT_PER_PAGE = 100

# Номера дисциплин для каждого профиля (по данным из матрицы компетенций)
PROFILES_DISCIPLINES = {
    "base": [1, 2, 3, 4, 5, 6, 9, 10, 13],
    "dc": [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 18, 20, 22, 24, 25],
    "top_dc": [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 18, 20, 22, 24, 25, 7, 8, 11, 12, 19, 21, 23],
}

# ====================== ЭМБЕДДИНГИ ======================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
EMBEDDINGS_CACHE_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Порог схожести для embedding_comparator (будет использоваться позже)
SIMILARITY_THRESHOLD = 0.75
