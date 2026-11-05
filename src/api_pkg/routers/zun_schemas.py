"""Pydantic request models for the ZUN (знать/уметь/навыки) router.

Extracted from ``zun.py`` to shrink the router module. The models are imported
back into ``zun.py`` verbatim, so any external importer of ``ZUNIn`` & co. keeps
working unchanged.
"""

from pydantic import BaseModel


class ZUNIn(BaseModel):
    ksa_type: str
    text: str


class ZUNPatch(BaseModel):
    text: str


class ScopeChange(BaseModel):
    discipline_name: str
    included: bool


class ScopePut(BaseModel):
    dir_code: str = "09.03.02"
    changes: list[ScopeChange] = []


class CompetencyIn(BaseModel):
    code: str
    name: str = ""
    description: str = ""


class SkillLinkIn(BaseModel):
    skill_name: str
    ksa_type: str = "skills"