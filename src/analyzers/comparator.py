from typing import Set, Tuple, Dict

class CompetencyComparator:
    """Класс для сравнения компетенций"""
    
    @staticmethod
    def compare_skills(program_skills: list, market_skills: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
        """Сравнивает навыки программы с рыночными"""
        program_set = set(program_skills)
        common = program_set & market_skills
        only_program = program_set - market_skills
        only_market = market_skills - program_set
        return common, only_program, only_market
    
    @staticmethod
    def calculate_metrics(program_skills: list, market_skills: Dict[str, int]) -> Dict[str, float]:
        """Рассчитывает метрики покрытия"""
        program_set = set(program_skills)
        market_set = set(market_skills.keys())
        
        if not market_set:
            return {
                'coverage': 0,
                'weighted_coverage': 0,
                'program_size': len(program_set),
                'market_size': 0,
                'common_count': 0
            }
        
        common = program_set & market_set
        coverage = len(common) / len(market_set)
        
        total_weight = sum(market_skills.values())
        weighted_coverage = sum(market_skills.get(s, 0) for s in program_set) / total_weight if total_weight > 0 else 0
        
        return {
            'coverage': coverage,
            'weighted_coverage': weighted_coverage,
            'program_size': len(program_set),
            'market_size': len(market_set),
            'common_count': len(common)
        }