from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional
from .enums import TargetType, SessionStatus


class SearchRequest(BaseModel):
    name: str
    target_type: TargetType = TargetType.PERSON
    context: Optional[str] = None  # any extra info the user provides upfront


class SearchSession(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    target_name: str
    target_type: TargetType
    context: Optional[str] = None
    status: SessionStatus = SessionStatus.SEARCHING
    narrowing_round: int = 0
    candidates_count: int = 0
    known_facts: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AnswerRequest(BaseModel):
    answer: str
