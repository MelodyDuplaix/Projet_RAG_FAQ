from pydantic import BaseModel, Field
from typing import List, Optional

class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(..., example="Comment obtenir un acte de naissance ?")

class AnswerResponse(BaseModel):
    """Response model for an answer."""
    answer: str
    confidence: float
    sources: List[str] = []
    latency_ms: float

class FAQ(BaseModel):
    """Model for a single FAQ item."""
    id: str
    question: str
    answer: str
    category: Optional[str] = None
    theme: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)
