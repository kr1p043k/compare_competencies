"""адаптеры для существующих классов с метриками конверсии."""
import json
import time
from pathlib import Path
from typing import Any

import structlog

from src import Err, Ok, config
from src.models.vacancy import Vacancy
from src.pipeline.data_source import HhDataSource
from src.pipeline.gap_runner import GapRunner
from src.pipeline.level_builder import LevelBuilder
from src.pipeline.skill_extractor import SkillExtractor
from src.pipeline.stage import PipelineStage
from src.pipeline.weight_cleaner import WeightCleaner
from src.pipeline.progress import write as write_progress
from src.result import Result
from src.scoring.vacancy_quality_scorer import VacancyQualityScorer
from src.predictors import create_ranking_predictor
from src.utils import safe_read_json
from src.monitoring.pipeline_metrics import pipeline_metrics
from src.monitoring.gap_metrics import gap_metrics

logger = structlog.get_logger(__name__)


class VacancyFetchStep(PipelineStage):
    """Шаг 1: Загрузка вакансий с hh.ru."""
    
    name = "vacancy_fetch"
    pct_range = (0, 18)

    def __init__(self, args):
        self.args = args

    def run(self, **kwargs) -> Result[tuple, Any]:
        pipeline_metrics.record_step_start(self.name)
        
        self._progress(0, "Инициализация сбора вакансий с hh.ru...")
        source = HhDataSource(self.args)
        
        match source.get_vacancies():
            case Ok((vacancies, parser)):
                self._progress(80, f"Загружено {len(vacancies)} вакансий с hh.ru")
                raw_file = None
                if self.args.skip_collection:
                    raw_file = source._find_file()
                self._progress(100, f"Сбор завершён: {len(vacancies)} вакансий")
                
                pipeline_metrics.record_step_end(self.name, "success", len(vacancies))
                pipeline_metrics.record_conversion("start", self.name)
                
                return Ok({"vacancies": vacancies, "parser": parser, "raw_file": raw_file})
            case Err(e):
                pipeline_metrics.record_step_end(self.name, "failed")
                return Err(e.message)


class SpamFilterStep(PipelineStage):
    """Шаг 2: Фильтрация спама и нерелевантных вакансий."""
    
    name = "spam_filter"
    pct_range = (18, 26)

    def __init__(self, args):
        self.args = args

    def run(self, vacancies, parser, **kwargs) -> Result[dict, Any]:
        pipeline_metrics.record_step_start(self.name)
        pipeline_metrics.record_conversion("vacancy_fetch", self.name)
        
        self._progress(0, "Оценка качества вакансий...")
        scorer = VacancyQualityScorer()
        scores = []
        spam_count = 0
        total = len(vacancies)

        for idx, v in enumerate(vacancies):
            if idx % 50 == 0:
                pct = int(idx / total * 80) if total else 0
                self._progress(pct, f"Оценка качества: {idx}/{total} вакансий")
            match scorer.score(v):
                case Ok(s):
                    scores.append(s)
                    if s.is_spam:
                        spam_count += 1
                        reason = "; ".join(f.reason for f in s.flags)
                        if hasattr(v, "raw_data") and isinstance(v.raw_data, dict):
                            v.raw_data["is_spam"] = True
                            v.raw_data["spam_reason"] = reason
                case Err(e):
                    pipeline_metrics.record_step_end(self.name, "failed")
                    return Err(e.message)

        self._progress(80, f"Спам-фильтр: отсеяно {spam_count} из {total}")
        quality_report = scorer._build_report(scores, len(vacancies))
        scorer.print_report(quality_report)

        from src.pipeline.helpers import save_detailed_vacancies
        save_detailed_vacancies(vacancies, logger)

        spam_path = config.REPORTS_DIR / "spam_vacancies_report.json"
        with open(spam_path, "w", encoding="utf-8") as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)

        if self.args.excel and vacancies:
            self._progress(90, "Генерация Excel-отчёта...")
            df = parser.aggregate_to_dataframe(vacancies, quality_report)
            excel_name = f"vacancies_{self.args.query.replace(' ', '_')}.xlsx"
            match parser.save_to_excel(df, excel_name):
                case Ok(_): pass
                case Err(e):
                    logger.warning("excel_save_failed", error=str(e))

        clean_pct = (total - spam_count) / total * 100 if total else 0
        self._progress(100, f"Оценка качества завершена: {total - spam_count} качественных вакансий")

        if clean_pct < 30:
            logger.warning(
                "hh_possible_similar_queries",
                clean_pct=round(clean_pct, 1),
                spam_count=spam_count,
                total=total,
            )
            print(f"\n  Обнаружено {spam_count}/{total} нерелевантных вакансий ({clean_pct:.0f}% качественных).")
            print(f"     Возможно, HH.ru вернул «похожие запросы» вместо точных результатов.")
            print(f"     Попробуйте другой регион или уточните запрос.\n")

        clean_count = total - spam_count
        pipeline_metrics.record_step_end(self.name, "success", clean_count)
        pipeline_metrics.record_conversion(self.name, "skill_parse")
        
        return Ok({"quality_report": quality_report})


class SkillParseStep(PipelineStage):
    """Шаг 3: Извлечение навыков из текста вакансий."""
    
    name = "skill_parse"
    pct_range = (26, 40)

    def __init__(self, args):
        self.args = args

    def run(self, vacancies, parser, raw_file, **kwargs) -> Result[dict, Any]:
        pipeline_metrics.record_step_start(self.name)
        pipeline_metrics.record_conversion("spam_filter", self.name)
        
        total = len(vacancies)
        self._progress(0, f"Извлечение навыков из {total} вакансий...")
        extractor = SkillExtractor(self.args)
        
        match extractor.extract(vacancies, parser, raw_file):
            case Ok((skill_freq, hybrid_weights_raw, trend_analyzer)):
                for i, v in enumerate(vacancies):
                    if isinstance(v, Vacancy):
                        match parser.skill_parser.parse_vacancy(v):
                            case Ok(extracted):
                                texts = list(dict.fromkeys(s.text for s in extracted if s.text))
                            case Err(_):
                                texts = []
                        v.raw_data["extracted_skills"] = texts
                    elif isinstance(v, dict):
                        vac_obj = Vacancy.from_api(v)
                        match parser.skill_parser.parse_vacancy(vac_obj):
                            case Ok(extracted):
                                texts = list(dict.fromkeys(s.text for s in extracted if s.text))
                            case Err(_):
                                texts = []
                        v["extracted_skills"] = texts
                    if (i + 1) % 100 == 0 or i == total - 1:
                        pct = int((i + 1) / total * 95)
                        self._progress(pct, f"Сохранены навыки: {i + 1}/{total}")
                        
                from src.pipeline.helpers import save_detailed_vacancies
                save_detailed_vacancies(vacancies, logger)
                self._progress(100, f"Извлечено {len(skill_freq)} уникальных навыков из {total} вакансий")
                
                pipeline_metrics.record_step_end(self.name, "success", len(skill_freq))
                pipeline_metrics.record_conversion(self.name, "weight_normalize")
                
                return Ok({
                    "skill_freq": skill_freq,
                    "hybrid_weights_raw": hybrid_weights_raw,
                    "trend_analyzer": trend_analyzer,
                })
            case Err(err):
                pipeline_metrics.record_step_end(self.name, "failed")
                return Err(err)


class WeightNormalizeStep(PipelineStage):
    """Шаг 4: Нормализация и очистка весов навыков."""
    
    name = "weight_normalize"
    pct_range = (40, 50)

    def run(self, hybrid_weights_raw, **kwargs) -> Result[dict, Any]:
        pipeline_metrics.record_step_start(self.name)
        pipeline_metrics.record_conversion("skill_parse", self.name)
        
        self._progress(0, f"Очистка {len(hybrid_weights_raw)} сырых весов навыков...")
        cleaner = WeightCleaner()
        
        match cleaner.clean(hybrid_weights_raw):
            case Ok(hybrid_weights):
                skill_weights_path = config.DATA_PROCESSED_DIR / "skill_weights.json"
                with open(skill_weights_path, "w", encoding="utf-8") as f:
                    json.dump(hybrid_weights, f, ensure_ascii=False, indent=2)
                self._progress(100, f"Очищено и сохранено {len(hybrid_weights)} весов навыков")
                
                pipeline_metrics.record_step_end(self.name, "success", len(hybrid_weights))
                pipeline_metrics.record_conversion(self.name, "level_assign")
                
                return Ok({"hybrid_weights": hybrid_weights})
            case Err(err):
                pipeline_metrics.record_step_end(self.name, "failed")
                return Err(err)


class LevelAssignStep(PipelineStage):
    """Шаг 5: Присвоение уровней (junior/middle/senior)."""
    
    name = "level_assign"
    pct_range = (50, 60)

    def run(self, vacancies, parser, **kwargs) -> Result[dict, Any]:
        pipeline_metrics.record_step_start(self.name)
        pipeline_metrics.record_conversion("weight_normalize", self.name)
        
        self._progress(0, f"Построение уровней компетенций для {len(vacancies)} вакансий...")
        builder = LevelBuilder()
        
        match builder.build(vacancies, parser):
            case Ok((level_data, vacancies_skills)):
                self._progress(100, f"Построено {len(level_data)} уровней компетенций")
                
                pipeline_metrics.record_step_end(self.name, "success", len(level_data))
                pipeline_metrics.record_conversion(self.name, "cluster_train")
                
                return Ok({"level_data": level_data, "vacancies_skills": vacancies_skills})
            case Err(err):
                pipeline_metrics.record_step_end(self.name, "failed")
                return Err(err)


class ClusterTrainStep(PipelineStage):
    """Шаг 6: Кластеризация вакансий."""
    
    name = "cluster_train"
    pct_range = (60, 65)

    def run(self, **kwargs) -> Result[dict, Any]:
        pipeline_metrics.record_step_start(self.name)
        pipeline_metrics.record_conversion("level_assign", self.name)
        
        self._progress(0, "Обучение кластеров вакансий...")
        from src.ml.clusters import train_clusters
        
        self._progress(10, "Кластеризация: загрузка данных...")
        ok = train_clusters(level="all", save_report=True, interpret=True)
        
        if ok:
            self._progress(100, "Кластеры вакансий успешно обучены")
            pipeline_metrics.record_step_end(self.name, "success")
            pipeline_metrics.record_conversion(self.name, "ltr_train")
            return Ok({"clusters_trained": True})
        
        self._progress(0, "Ошибка обучения кластеров")
        pipeline_metrics.record_step_end(self.name, "failed")
        return Err("Обучение кластеров не выполнено")


class LTRTrainStep(PipelineStage):
    """Шаг 7: Обучение LTR-модели ранжирования."""
    
    name = "ltr_train"
    pct_range = (65, 70)

    def run(self, **kwargs) -> Result[dict, Any]:
        pipeline_metrics.record_step_start(self.name)
        pipeline_metrics.record_conversion("cluster_train", self.name)
        
        self._progress(0, "Обучение LTR-модели ранжирования...")
        detailed_file = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
        basic_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
        raw_file = detailed_file if detailed_file.exists() else basic_file
        
        if not raw_file.exists():
            self._progress(100, "Нет данных для обучения модели")
            pipeline_metrics.record_step_end(self.name, "success")
            return Ok({"model_trained": False, "reason": "no_vacancy_file"})

        model_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
        if model_path.exists():
            self._progress(5, "Проверка актуальности модели...")
            match create_ranking_predictor(model_path=model_path):
                case Ok(ltr_engine) if ltr_engine.is_fitted:
                    pass
                case _:
                    ltr_engine = None
            if ltr_engine:
                model_mtime = model_path.stat().st_mtime
                data_mtime = raw_file.stat().st_mtime
                if model_mtime > data_mtime:
                    self._progress(100, "Модель уже актуальна (пропускаем обучение)")
                    pipeline_metrics.record_step_end(self.name, "success")
                    pipeline_metrics.record_conversion(self.name, "gap_compute")
                    return Ok({"model_trained": False, "reason": "already_up_to_date"})

        training_vacancies = safe_read_json(raw_file)
        if not training_vacancies:
            pipeline_metrics.record_step_end(self.name, "failed")
            return Err("Не удалось прочитать файл вакансий")

        self._progress(20, f"Загружено {len(training_vacancies)} вакансий для обучения LTR")
        from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine
        
        ltr_engine = LTRRecommendationEngine()
        self._progress(30, "Запуск обучения LTR-модели (может занять время)...")
        
        match ltr_engine.fit(training_vacancies):
            case Ok(_):
                if hasattr(ltr_engine, "last_metrics"):
                    m = ltr_engine.last_metrics
                    self._progress(90, f"LTR: R²={m['r2']:.4f} MAE={m['mae']:.4f}")
                if ltr_engine.is_fitted:
                    self._progress(100, "LTR-модель ранжирования успешно обучена")
                    pipeline_metrics.record_step_end(self.name, "success")
                    pipeline_metrics.record_conversion(self.name, "gap_compute")
                    return Ok({"model_trained": True})
                pipeline_metrics.record_step_end(self.name, "failed")
                return Err("Обучение модели не удалось")
            case Err(err):
                pipeline_metrics.record_step_end(self.name, "failed")
                return Err(str(err))


class GapComputeStep(PipelineStage):
    """Шаг 8: Вычисление разрывов компетенций."""
    
    name = "gap_compute"
    pct_range = (70, 92)

    def __init__(self, profiles, ctx, args):
        self.profiles = profiles
        self.ctx = ctx
        self.args = args

    def run(self, **kwargs) -> Result[dict, Any]:
        pipeline_metrics.record_step_start(self.name)
        pipeline_metrics.record_conversion("ltr_train", self.name)
        
        num_profiles = len(self.profiles)
        self._progress(0, f"GAP-анализ для {num_profiles} профилей...")
        
        # Засекаем время gap-анализа
        gap_start = time.time()
        
        runner = GapRunner(self.profiles, self.ctx, self.args)
        match runner.run():
            case Ok((evaluations, recs)):
                gap_duration_val = time.time() - gap_start
                
                # Записываем метрики gap-анализа
                total_recommendations = 0
                for pname, rec_data in recs.items():
                    if isinstance(rec_data, dict):
                        rec_list = rec_data.get("recommendations", [])
                        count = len(rec_list)
                        total_recommendations += count
                        
                        # Распределяем по приоритетам (пример)
                        priorities = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                        for rec in rec_list:
                            if isinstance(rec, dict):
                                priority = rec.get("priority", "medium")
                                if priority in priorities:
                                    priorities[priority] += 1
                        
                        gap_metrics.start(pname, self.args.area_id or 1)
                        gap_metrics.add_recommendations(priorities)
                        gap_metrics.end(success=True)
                
                gap_metrics.record_analysis(
                    profile_type="all",
                    duration=gap_duration_val,
                    success=True,
                    recommendations_count_val=total_recommendations
                )
                
                self._progress(100, f"GAP-анализ завершён за {gap_duration_val:.2f} сек для {num_profiles} профилей")
                
                pipeline_metrics.record_step_end(self.name, "success", total_recommendations)
                pipeline_metrics.end_pipeline("main")
                
                return Ok({"evaluations": evaluations, "recommendations": recs})
            case Err(err):
                gap_duration_val = time.time() - gap_start
                gap_metrics.record_analysis(
                    profile_type="all",
                    duration=gap_duration_val,
                    success=False
                )
                pipeline_metrics.record_step_end(self.name, "failed")
                return Err(err)