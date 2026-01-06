from pydantic import BaseModel
from typing import List, Dict, Any

class AnalysisInput(BaseModel):
    results: List[Dict[str, Any]]
    exams: List[Dict[str, Any]]
    subjects: Dict[str, str]