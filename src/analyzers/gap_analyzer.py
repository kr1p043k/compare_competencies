from typing import Dict, List, Tuple

class GapAnalyzer:
    """Класс для анализа дефицита компетенций"""
    
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
    
    def analyze_gap(self, program_skills: list, market_skills: Dict[str, int]) -> Dict[str, List[Tuple[str, int]]]:
        """Анализирует дефицит компетенций"""
        program_set = set(program_skills)
        market_set = set(market_skills.keys())
        
        missing = market_set - program_set
        
        if not market_skills:
            return {
                'high_demand': [],
                'medium_demand': [],
                'low_demand': [],
                'total_missing': len(missing)
            }
        
        max_freq = max(market_skills.values())
        high_demand, medium_demand, low_demand = [], [], []
        
        for skill in missing:
            freq = market_skills.get(skill, 0)
            normalized = freq / max_freq
            
            if normalized > self.threshold:
                high_demand.append((skill, freq))
            elif normalized > self.threshold * 0.3:
                medium_demand.append((skill, freq))
            else:
                low_demand.append((skill, freq))
        
        return {
            'high_demand': sorted(high_demand, key=lambda x: x[1], reverse=True),
            'medium_demand': sorted(medium_demand, key=lambda x: x[1], reverse=True),
            'low_demand': sorted(low_demand, key=lambda x: x[1], reverse=True),
            'total_missing': len(missing)
        }