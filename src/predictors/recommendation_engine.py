"""
Движок рекомендаций: анализирует gap между студентом и рынком,
даёт человекочитаемые советы по развитию навыков.
"""

import logging
from typing import List, Dict, Optional, Tuple
from src.analyzers.comparator import CompetencyComparator
from src.analyzers.gap_analyzer import GapAnalyzer
from src.analyzers.skill_filter import SkillFilter
from src.models.student import StudentProfile

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Движок рекомендаций с TF-IDF анализом и генерацией естественного языка.
    
    Улучшения:
    - Интегрирована фильтрация навыков (убираются bigrams и мусор)
    - Детальные шаблоны для всех tech-стеков
    - Умные рекомендации по приоритизации
    - Различие между hard и soft skills
    """
    
    # === HARD SKILLS (технические) ===
    HARD_SKILL_TEMPLATES = {
        # Backend
        "python": "Python — основной язык для бэкенда и data science. Нужны глубокие знания.",
        "java": "Java — стандарт для enterprise-приложений. Актуально для больших проектов.",
        "javascript": "JavaScript — базовый язык веб-разработки. Обязателен для фронтенда и Node.js.",
        "typescript": "TypeScript добавляет типизацию и предотвращает ошибки. Стандарт в современной разработке.",
        "go": "Go — язык для системного программирования и микросервисов. Очень актуален.",
        "rust": "Rust гарантирует безопасность памяти. Тренд в системном программировании.",
        
        # Фреймворки
        "django": "Django — мощный фреймворк для быстрой разработки на Python. Подходит для MVP и production.",
        "fastapi": "FastAPI отлично для REST API и микросервисов. Современный, быстрый.",
        "react": "React — лидер на фронтенде. Нужно уметь писать компоненты и управлять состоянием.",
        "node": "Node.js позволяет писать бэкенд на JavaScript. Популярен в стартапах.",
        "express": "Express — минималистичный фреймворк для Node.js. Простой в освоении.",
        
        # БД и хранилища
        "postgresql": "PostgreSQL — мощная СУБД. Стандарт для production систем.",
        "mongodb": "MongoDB подходит для документо-ориентированных данных. NoSQL альтернатива.",
        "redis": "Redis нужен для кэширования и real-time приложений. Критичен для performance.",
        "sql": "SQL — язык работы с базами данных. Базовый навык любого разработчика.",
        "elasticsearch": "Elasticsearch нужен для полнотекстового поиска и аналитики.",
        
        # DevOps и infrastructure
        "docker": "Docker — стандарт для контейнеризации. Нужен для deployment и collaboration.",
        "kubernetes": "Kubernetes необходим для масштабирования контейнеров. Стандарт в больших системах.",
        "git": "Git — система контроля версий. Обязателен для любого проекта.",
        "linux": "Linux — ОС для серверов. Нужно уметь работать в консоли.",
        "aws": "AWS — облачный провайдер №1. Знание дает огромный плюс к зарплате.",
        "docker git": "Docker + Git вместе образуют стандартный CI/CD пайплайн.",
        
        # Frontend
        "html": "HTML — основа веб-страниц. Базовый навык веб-разработчика.",
        "css": "CSS нужен для красивого оформления. Современный CSS очень мощный.",
        "sass": "SASS упрощает написание CSS. Стандарт в больших проектах.",
        "webpack": "Webpack — инструмент сборки. Нужен для оптимизации production-сборки.",
        "redux": "Redux управляет состоянием в больших приложениях. Альтернатива MobX.",
        "graphql": "GraphQL — альтернатива REST для более эффективного API.",
        
        # Data Science / ML
        "machine learning": "Machine Learning — быстро растущая область. Высокие зарплаты.",
        "numpy": "NumPy — основа для численных вычислений в Python.",
        "pandas": "Pandas незаменим для работы с данными (DataFrames).",
        "scikit-learn": "scikit-learn — стандартная библиотека для ML алгоритмов.",
        "tensorflow": "TensorFlow — лидер для deep learning и neural networks.",
        "pytorch": "PyTorch альтернатива TensorFlow, популярен в research.",
    }
    
    # === SOFT SKILLS ===
    SOFT_SKILL_TEMPLATES = {
        "английский язык": "Английский язык B2+ открывает доступ к документации и конференциям.",
        "аналитическое мышление": "Аналитическое мышление критично для решения сложных задач.",
        "системный анализ": "Системный анализ помогает понимать требования и архитектуру.",
        "коммуникация": "Навык коммуникации важен для teamwork и presentation.",
        "лидерство": "Лидерство нужно для роста в старший разработчик или техлид.",
    }
    
    # === ПУТИ ОБУЧЕНИЯ (hard skills) ===
    HARD_LEARNING_PATHS = {
        "python": "1. Основы: переменные, функции, ООП. 2. Практика: 10+ мини-проектов. 3. Углубление: декораторы, генераторы, async/await.",
        "java": "1. Синтаксис и ООП. 2. Collections и Streams. 3. Spring Framework для production кода.",
        "javascript": "1. ES6+ синтаксис. 2. DOM и события. 3. Async/await и Promises.",
        "typescript": "1. Типы и interfaces. 2. Generics. 3. Advanced types и utility types.",
        "django": "1. Туториал на официальном сайте. 2. Создайте 2-3 собственных проекта. 3. Изучите DRF для API.",
        "fastapi": "1. Официальная документация. 2. Создайте REST API. 3. Добавьте аутентификацию и валидацию.",
        "react": "1. Компоненты и hooks. 2. State management (Redux/Zustand). 3. Performance optimization.",
        "postgresql": "1. SQL основы. 2. Индексы и оптимизация. 3. Транзакции и ACID.",
        "docker": "1. Dockerfile и образы. 2. docker-compose для development. 3. Registry и deployment.",
        "kubernetes": "1. Pods, Services, Deployments. 2. ConfigMaps и Secrets. 3. Ingress и networking.",
        "aws": "1. EC2, S3, RDS. 2. VPC и IAM. 3. Lambda и serverless.",
    }
    
    # === ПУТИ ОБУЧЕНИЯ (soft skills) ===
    SOFT_LEARNING_PATHS = {
        "английский язык": "Занимайтесь ежедневно 30 минут: читайте техническую документацию, смотрите конференции на YouTube.",
        "аналитическое мышление": "Решайте задачи на LeetCode, участвуйте в code review, анализируйте чужой код.",
        "системный анализ": "Участвуйте в design review, пишите ТЗ, думайте о масштабируемости.",
        "коммуникация": "Пишите документацию, проводите презентации, получайте обратную связь.",
        "лидерство": "Менторьте junior разработчиков, возглавляйте small features, растите ответственность.",
    }
    
    def __init__(self):
        """Инициализирует движок с компаратором и фильтром."""
        self.comparator = CompetencyComparator()
        self.gap_analyzer = None
        self.skill_filter = SkillFilter()
        self.is_fitted = False
        logger.info("✓ RecommendationEngine инициализирован")
    
    def fit(self, vacancies_skills: List[List[str]]):
        """Обучает на рынке вакансий (TF-IDF)."""
        if not vacancies_skills:
            logger.warning("❌ Нет данных вакансий для обучения")
            return
        
        self.comparator.fit_market(vacancies_skills)
        
        # Инициализируем gap_analyzer с чистыми весами
        skill_weights = self.comparator.get_skill_weights()
        self.gap_analyzer = GapAnalyzer(skill_weights)
        
        self.is_fitted = True
        logger.info(f"✓ RecommendationEngine обучен на {len(vacancies_skills)} вакансиях")
    
    def analyze(self, student_skills: List[str]) -> Dict:
        """
        Анализирует студента: match_score, coverage, missing_skills.
        
        Returns:
            Словарь с результатами анализа
        """
        if not self.is_fitted:
            logger.warning("❌ RecommendationEngine не обучен")
            return {}
        
        # Получаем веса
        skill_weights = self.comparator.get_skill_weights()
        
        # Сравнение (вернёт кортеж!)
        score, confidence = self.comparator.compare(student_skills)
        
        # Gap анализ
        gaps = self.gap_analyzer.analyze_gap(student_skills, top_n=30)
        coverage, coverage_details = self.gap_analyzer.coverage(student_skills)
        top_market = self.gap_analyzer.top_market_skills(20)
        
        return {
            "match_score": round(score, 4),
            "confidence": round(confidence, 4),
            "coverage": round(coverage, 2),
            "coverage_details": coverage_details,
            "gaps": gaps,
            "top_market_skills": top_market
        }
    
    def generate_recommendations(
        self,
        student_skills: List[str],
        student_profile: Optional[StudentProfile] = None
    ) -> Dict:
        """
        Генерирует детальные, человекочитаемые рекомендации.
        
        Args:
            student_skills: Навыки студента
            student_profile: Профиль студента (опционально)
        
        Returns:
            Словарь с детальными рекомендациями
        """
        analysis = self.analyze(student_skills)
        if not analysis:
            return {}
        
        gaps = analysis.get("gaps", {})
        high_priority_gaps = gaps.get("high_priority", [])[:10]
        medium_priority_gaps = gaps.get("medium_priority", [])[:8]
        
        # Генерируем детальные рекомендации
        detailed_recs = []
        for i, gap in enumerate(high_priority_gaps, 1):
            skill = gap.get("skill", "")
            importance = gap.get("importance", 0)
            
            rec = self._generate_skill_recommendation(
                skill=skill,
                importance=importance,
                priority="HIGH",
                rank=i,
                student_profile=student_profile
            )
            if rec:
                detailed_recs.append(rec)
        
        # Добавляем medium priority
        for i, gap in enumerate(medium_priority_gaps, 1):
            skill = gap.get("skill", "")
            importance = gap.get("importance", 0)
            
            rec = self._generate_skill_recommendation(
                skill=skill,
                importance=importance,
                priority="MEDIUM",
                rank=len(high_priority_gaps) + i,
                student_profile=student_profile
            )
            if rec:
                detailed_recs.append(rec)
        
        return {
            "summary": {
                "match_score": analysis["match_score"],
                "confidence": analysis["confidence"],
                "coverage": analysis["coverage"],
                "covered_skills": analysis["coverage_details"]["covered_skills_count"],
                "total_market_skills": analysis["coverage_details"]["total_market_skills"],
            },
            "recommendations": detailed_recs,
            "top_market_skills": analysis["top_market_skills"]
        }
    
    # ========== Вспомогательные методы ==========
    
    def _generate_skill_recommendation(
        self,
        skill: str,
        importance: float,
        priority: str,
        rank: int,
        student_profile: Optional[StudentProfile] = None
    ) -> Dict:
        """Генерирует детальную рекомендацию по одному навыку."""
        
        skill_lower = skill.lower()
        is_soft = self._is_soft_skill(skill_lower)
        
        return {
            "rank": rank,
            "skill": skill,
            "importance_score": round(importance, 4),
            "priority": priority,
            "is_soft_skill": is_soft,
            "suggestion": self._get_suggestion(skill, is_soft),
            "why_important": self._why_important(skill, importance),
            "how_to_learn": self._get_learning_path(skill, is_soft),
            "expected_timeframe": self._get_timeframe(skill),
            "expected_outcome": self._get_expected_outcome(skill, student_profile),
            "market_frequency_percent": round(importance * 100, 1),
        }
    
    def _is_soft_skill(self, skill_lower: str) -> bool:
        """Определяет, soft skill это или hard skill."""
        soft_keywords = [
            "английский", "язык", "коммуникация", "лидерство",
            "мышление", "анализ", "аналитическое", "системный"
        ]
        return any(kw in skill_lower for kw in soft_keywords)
    
    def _get_suggestion(self, skill: str, is_soft: bool) -> str:
        """Получает описание навыка из шаблонов или генерирует новое."""
        skill_lower = skill.lower()
        
        if is_soft:
            return self.SOFT_SKILL_TEMPLATES.get(
                skill_lower,
                f"Развитие soft skill '{skill}' улучшит вашу эффективность в команде."
            )
        else:
            template = self.HARD_SKILL_TEMPLATES.get(skill_lower)
            if template:
                return template
            
            # Для неизвестных навыков генерируем сообщение
            return f"Навык '{skill}' высоко востребован на рынке и повысит вашу конкурентоспособность."
    
    def _why_important(self, skill: str, importance: float) -> str:
        """Объясняет, почему навык важен."""
        freq_percent = round(importance * 100, 1)
        
        if freq_percent >= 50:
            return f"🔴 КРИТИЧНЫЙ: '{skill}' требуется в {freq_percent}% вакансий. Это основной навык вашей роли."
        elif freq_percent >= 20:
            return f"🟠 ВЫСОКИЙ: '{skill}' встречается в {freq_percent}% вакансий. Нужно срочно освоить."
        elif freq_percent >= 10:
            return f"🟡 СРЕДНИЙ: '{skill}' требуется в {freq_percent}% вакансий. Даст конкурентное преимущество."
        else:
            return f"🟢 НИЗКИЙ: '{skill}' встречается в {freq_percent}% вакансий. Полезно для специализированных ролей."
    
    def _get_learning_path(self, skill: str, is_soft: bool) -> str:
        """Возвращает путь обучения для навыка."""
        skill_lower = skill.lower()
        
        if is_soft:
            return self.SOFT_LEARNING_PATHS.get(
                skill_lower,
                "Практикуйте навык постоянно в реальных проектах, просите обратную связь."
            )
        else:
            return self.HARD_LEARNING_PATHS.get(
                skill_lower,
                f"1. Изучите официальную документацию '{skill}'. 2. Выполните 3+ практических проекта. 3. Примите участие в open source."
            )
    
    def _get_timeframe(self, skill: str) -> str:
        """Рекомендуемое время на обучение."""
        skill_lower = skill.lower()
        
        # Легко учить (1-2 недели)
        easy = {"git", "html", "css", "sass", "english", "английский язык"}
        # Средне (1-2 месяца)
        medium = {"javascript", "python", "sql", "redis", "docker", "react"}
        # Сложно (2-6 месяцев)
        hard = {"java", "kubernetes", "aws", "machine learning", "tensorflow"}
        
        if any(k in skill_lower for k in easy):
            return "1-2 недели"
        elif any(k in skill_lower for k in medium):
            return "1-2 месяца"
        elif any(k in skill_lower for k in hard):
            return "2-6 месяцев"
        else:
            return "1-3 месяца"
    
    def _get_expected_outcome(self, skill: str, student_profile: Optional[StudentProfile]) -> str:
        """Описывает, что получит студент после обучения."""
        role = student_profile.target_role if student_profile else "вашей целевой роли"
        
        return (
            f"Освоение '{skill}' позволит вам уверенно работать в роли '{role}', "
            f"понимать код коллег и писать production-quality решения."
        )