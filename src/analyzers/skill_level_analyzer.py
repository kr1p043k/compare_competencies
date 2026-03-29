"""
Анализатор навыков по уровню опыта
Разделяет навыки на: junior, middle, senior
"""

import logging
from typing import Dict, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class SkillLevelAnalyzer:
    """
    Определяет, какие навыки относятся к какому уровню
    на основе анализа вакансий
    """
    
    def __init__(self):
        self.skill_by_level = defaultdict(lambda: defaultdict(int))
        self.level_thresholds = {
            'junior': 0.3,    # навык в 30%+ junior вакансиях
            'middle': 0.5,    # навык в 50%+ middle вакансиях
            'senior': 0.7     # навык в 70%+ senior вакансиях
        }
    
    def analyze_vacancies(self, vacancies: List[Dict]):
        """
        Анализирует вакансии и определяет распределение навыков по уровням
        
        Args:
            vacancies: List[Dict] с полями:
                - 'skills': List[str]
                - 'experience': str (junior/middle/senior/не указано)
        """
        logger.info("Анализируем распределение навыков по уровням...")
        
        total_by_level = defaultdict(int)
        
        # Подсчитываем навыки по уровням
        for vac in vacancies:
            skills = vac.get('skills', [])
            experience = vac.get('experience', 'middle').lower()
            
            # Нормализуем experience
            if experience not in ['junior', 'middle', 'senior']:
                experience = 'middle'
            
            total_by_level[experience] += 1
            
            for skill in skills:
                skill_lower = skill.lower().strip()
                self.skill_by_level[skill_lower][experience] += 1
        
        logger.info(f"Распределение вакансий: {dict(total_by_level)}")
        logger.info(f"Анализировано {len(self.skill_by_level)} уникальных навыков")
    
    def get_skill_level(self, skill: str) -> str:
        """
        Определяет основной уровень для навыка
        (на основе где он встречается чаще всего)
        
        Returns:
            'junior', 'middle', 'senior' или 'all_levels'
        """
        skill_lower = skill.lower().strip()
        levels_count = self.skill_by_level.get(skill_lower, {})
        
        if not levels_count:
            return 'middle'  # default
        
        # Найти уровень с максимальным количеством упоминаний
        max_level = max(levels_count.items(), key=lambda x: x[1])[0]
        total = sum(levels_count.values())
        
        # Если навык встречается в 50%+ вакансиях разных уровней
        if len([l for l in levels_count if levels_count[l] / total > 0.3]) >= 2:
            return 'all_levels'
        
        return max_level
    
    def get_weights_for_level(
        self, 
        skill_weights: Dict[str, float],
        target_level: str
    ) -> Dict[str, float]:
        """
        Фильтрует и переустанавливает веса для конкретного уровня
        
        Args:
            skill_weights: исходные веса навыков (с рынка)
            target_level: 'junior', 'middle', 'senior'
        
        Returns:
            новый словарь весов, где навыки переупорядочены по релевантности для уровня
        """
        level_weights = {}
        
        for skill, original_weight in skill_weights.items():
            skill_lower = skill.lower().strip()
            levels_count = self.skill_by_level.get(skill_lower, {})
            
            if not levels_count:
                level_weights[skill] = original_weight
                continue
            
            total = sum(levels_count.values())
            skill_level = self.get_skill_level(skill)
            
            # Получаем процент для целевого уровня
            level_percentage = levels_count.get(target_level, 0) / total if total > 0 else 0
            
            # Умножаем исходный вес на релевантность для этого уровня
            adjusted_weight = original_weight * (0.3 + level_percentage)
            
            # Логика приоритета:
            if skill_level == target_level:
                # Навык специфичен для этого уровня - повышаем вес
                adjusted_weight *= 1.5
            elif skill_level == 'all_levels':
                # Навык для всех уровней - слегка повышаем
                adjusted_weight *= 1.1
            elif self._is_lower_level(skill_level, target_level):
                # Навык более junior - может быть базой
                adjusted_weight *= 0.8
            else:
                # Навык более senior - сложнее, но может быть полезен
                adjusted_weight *= 0.6
            
            level_weights[skill] = adjusted_weight
        
        return level_weights
    
    def _is_lower_level(self, current: str, target: str) -> bool:
        """Проверяет, ниже ли текущий уровень целевого"""
        levels = ['junior', 'middle', 'senior']
        try:
            return levels.index(current) < levels.index(target)
        except ValueError:
            return False
    
    def get_skill_roadmap(self, skill: str) -> Dict[str, bool]:
        """
        Возвращает дорожку развития навыка по уровням
        
        Returns:
            {'junior': True/False, 'middle': True/False, 'senior': True/False}
        """
        skill_lower = skill.lower().strip()
        levels_count = self.skill_by_level.get(skill_lower, {})
        
        return {
            'junior': levels_count.get('junior', 0) > 0,
            'middle': levels_count.get('middle', 0) > 0,
            'senior': levels_count.get('senior', 0) > 0
        }