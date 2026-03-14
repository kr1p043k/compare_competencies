from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_AREAS_DIR = BASE_DIR / "data" / "areas"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

for dir_path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_AREAS_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Настройки API hh.ru
HH_USER_AGENT = "CompetencyAnalyzer/1.0 (opik@sfedu.ru)" 
REQUEST_DELAY = 0.2

# Параметры поиска
DEFAULT_AREA = 76            # 76 = Ростовская область (исправлено!)
DEFAULT_PERIOD_DAYS = 30
DEFAULT_MAX_PAGES = 20