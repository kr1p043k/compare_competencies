from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Student:
    id: str
    name: str
    skills: List[str]
    target_profession: Optional[str] = None