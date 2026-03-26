from pydantic import BaseModel, Field
from typing import List, Dict

class StudentProfile(BaseModel):
    student_id: str
    name: str = "Unnamed Student"
    competencies: List[str] = Field(default_factory=list)   # коды компетенций (SS1.1, DL-1.3 и т.д.)
    target_role: str = "Data Scientist"                     # для будущей приоритизации