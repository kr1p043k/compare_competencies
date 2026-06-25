"""DataSource — загружает вакансии из кэша или через API hh.ru."""

from datetime import datetime
from typing import Protocol

import structlog

from src import DataSourceError, Err, Ok, Result, config
from src.parsing.api.hh_api import HeadHunterAPI
from src.parsing.skills.vacancy_parser import VacancyParser
from src.parsing.utils import (
    IT_PROFESSIONAL_ROLES,
    collect_vacancies_multiple,
    interactive_config,
    load_queries_from_file,
)
from src.utils import safe_read_json, validate_safe_path

from .helpers import console_header, console_info, get_load_mode, load_vacancies_details, save_detailed_vacancies

logger = structlog.get_logger("data_source")


class DataSourceProtocol(Protocol):
    def get_vacancies(self, queries: list[str], max_pages: int) -> Result[tuple[list[dict], VacancyParser], DataSourceError]: ...


class HhDataSource(DataSourceProtocol):
    """Загружает вакансии, возвращает (список, парсер)."""

    def __init__(self, args):
        self.args = args

    def get_vacancies(self) -> Result[tuple[list, VacancyParser], DataSourceError]:
        try:
            if self.args.skip_collection:
                return self._load_from_cache()
            return self._collect_from_hh()
        except DataSourceError:
            raise
        except Exception as e:
            return Err(DataSourceError(message=f"Ошибка сбора вакансий: {e}"))

    def _load_from_cache(self) -> Result[tuple[list, VacancyParser], DataSourceError]:
        raw_file = self._find_file()
        if raw_file is None:
            return Err(DataSourceError(message="❌ Файлы вакансий не найдены."))
        data = safe_read_json(raw_file)
        if not data:
            return Err(DataSourceError(message="❌ Не удалось прочитать файл вакансий."))
        parser = VacancyParser()
        from src.models.vacancy import Vacancy
        vacancies = [Vacancy.from_api(v) if isinstance(v, dict) else v for v in data]
        return Ok((vacancies, parser))

    def _collect_from_hh(self) -> Result[tuple[list, VacancyParser], DataSourceError]:
        hh_api = HeadHunterAPI()
        parser = VacancyParser()

        use_multiple = (
            self.args.interactive
            or self.args.queries_file is not None
            or self.args.regions is not None
            or self.args.industry is not None
            or self.args.it_sector
        )

        console_header("СБОР ВАКАНСИЙ С HH.RU")

        # Инкрементальный сбор: определить период с даты последнего запуска
        date_from = getattr(self.args, '_date_from', None)
        if date_from:
            delta = (datetime.now() - datetime.strptime(date_from, "%Y-%m-%d")).days
            if 1 <= delta <= 30:
                self.args.period = delta
                console_info(f"Инкрементальный сбор: {delta} дней (с {date_from})")
            else:
                date_from = None
                self.args.period = self.args.period

        if use_multiple:
            if self.args.interactive:
                params = interactive_config()
                self.args.query = params.get("query", self.args.query)
                self.args.queries = params.get("queries", [self.args.query])
                self.args.area_ids = params.get("area_ids", [self.args.area_id])
                self.args.industry = params.get("industry")
                self.args.period = params.get("period", self.args.period)
                self.args.max_pages = params.get("max_pages", self.args.max_pages)
                self.args.skip_details = params.get("skip_details", self.args.skip_details)
                self.args.show_vacancies = params.get("show_vacancies", self.args.show_vacancies)
                self.args.excel = params.get("excel", self.args.excel)
                self.args.no_filter = params.get("no_filter", self.args.no_filter)
                self.args.max_vacancies_per_query = params.get(
                    "max_vacancies_per_query", self.args.max_vacancies_per_query
                )
            else:
                if self.args.it_sector:
                    self.args.queries = [
                        "Data Scientist",
                        "Data Analyst",
                        "Machine Learning Engineer",
                        "Computer Vision Engineer",
                        "NLP Engineer",
                        "Data Architect",
                        "ETL Developer",
                        "Python Developer",
                        "Java Developer",
                        "Frontend Developer",
                        "Backend Developer",
                        "Fullstack Developer",
                        "DevOps Engineer",
                        "Embedded Developer",
                        "Blockchain Developer",
                        "iOS Developer",
                        "Android Developer",
                        "React Native Developer",
                        "Flutter Developer",
                        "QA Engineer",
                        "Automation QA Engineer",
                        "Performance QA Engineer",
                        "Специалист по кибербезопасности",
                        "Security Engineer",
                        "DevSecOps Engineer",
                        "SRE инженер",
                        "Системный администратор",
                        "Облачный инженер",
                        "Сетевой инженер",
                        "Администратор баз данных",
                        "Системный аналитик",
                        "Бизнес-аналитик",
                        "Архитектор программного обеспечения",
                        "Solution Architect",
                        "Team Lead",
                        "Tech Lead",
                        "Project Manager IT",
                        "Scrum Master",
                        "UX/UI дизайнер",
                        "Product Designer",
                        "Unity Developer",
                        "Unreal Engine Developer",
                        "Technical Writer",
                    ]
                    self.args.industry = 7
                    self.args.max_vacancies_per_query = 100000
                    console_info("Режим: поиск по всему IT-сектору (40+ профессий)")
                elif self.args.queries_file:
                    safe_path = validate_safe_path(self.args.queries_file)
                    self.args.queries = load_queries_from_file(safe_path)
                else:
                    self.args.queries = [self.args.query]

                if self.args.regions:
                    self.args.area_ids = [int(x.strip()) for x in self.args.regions.split(",")]
                else:
                    self.args.area_ids = [self.args.area_id]

            basic_vacancies = collect_vacancies_multiple(
                hh_api=hh_api,
                queries=self.args.queries,
                area_ids=self.args.area_ids,
                period_days=self.args.period,
                max_pages=self.args.max_pages,
                industry=self.args.industry,
                professional_role=IT_PROFESSIONAL_ROLES,
                max_vacancies_per_query=self.args.max_vacancies_per_query,
            )

            if not basic_vacancies:
                return Err(DataSourceError(message="❌ Не найдено вакансий."))

            console_info(f"Найдено {len(basic_vacancies)} базовых вакансий")
            match parser.save_raw_vacancies(basic_vacancies, filename="hh_vacancies_basic.json"):
                case Ok(_): pass
                case Err(e):
                    return Err(DataSourceError(message=f"❌ Ошибка сохранения вакансий: {e}"))

            if self.args.skip_details:
                from src.models.vacancy import Vacancy
                vacancies_to_process = [Vacancy.from_api(v) for v in basic_vacancies]
                console_info("Пропуск загрузки деталей (--skip-details)")
            else:
                total_vacs = len(basic_vacancies)
                use_async, async_workers, reason = get_load_mode(total_vacs, self.args, logger)
                match load_vacancies_details(
                    basic_vacancies=basic_vacancies,
                    hh_api=hh_api,
                    use_async=use_async,
                    async_workers=async_workers,
                    parser=parser,
                    log=logger,
                ):
                    case Ok(vacancies_to_process):
                        pass
                    case Err(e):
                        return Err(DataSourceError(message=str(e)))
                save_detailed_vacancies(vacancies_to_process, logger)

            if self.args.show_vacancies:
                parser.print_vacancies_list(vacancies_to_process)

        else:
            console_info(f"Поиск: '{self.args.query}', регион {self.args.area_id}")
            match hh_api.search_vacancies(
                text=self.args.query,
                area=self.args.area_id,
                period_days=self.args.period,
                max_pages=self.args.max_pages,
                professional_role=IT_PROFESSIONAL_ROLES,
            ):
                case Ok(basic_vacancies):
                    pass
                case Err(e):
                    return Err(DataSourceError(message=f"❌ Ошибка поиска вакансий: {e}"))

            if not basic_vacancies:
                return Err(DataSourceError(message="❌ Не найдено вакансий."))

            console_info(f"Найдено {len(basic_vacancies)} вакансий")
            match parser.save_raw_vacancies(basic_vacancies, filename="hh_vacancies_basic.json"):
                case Ok(_): pass
                case Err(e):
                    return Err(DataSourceError(message=f"❌ Ошибка сохранения вакансий: {e}"))

            if self.args.skip_details:
                from src.models.vacancy import Vacancy
                vacancies_to_process = [Vacancy.from_api(v) for v in basic_vacancies]
            else:
                total_vacs = len(basic_vacancies)
                use_async, async_workers, reason = get_load_mode(total_vacs, self.args, logger)
                match load_vacancies_details(
                    basic_vacancies=basic_vacancies,
                    hh_api=hh_api,
                    use_async=use_async,
                    async_workers=async_workers,
                    parser=parser,
                    log=logger,
                ):
                    case Ok(vacancies_to_process):
                        pass
                    case Err(e):
                        return Err(DataSourceError(message=str(e)))
                save_detailed_vacancies(vacancies_to_process, logger)

        return Ok((vacancies_to_process, parser))

    def _find_file(self):
        detailed = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
        basic = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
        if detailed.exists():
            return detailed
        elif basic.exists():
            return basic
        return None

    @staticmethod
    def _console_info(msg):
        print(f"  {msg}")



