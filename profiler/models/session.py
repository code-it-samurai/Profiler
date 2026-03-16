from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional
from .enums import TargetType, SessionStatus


class SearchRequest(BaseModel):
    name: str
    target_type: TargetType = TargetType.PERSON
    # Optional structured fields — used to improve search + pre-populate known_facts
    email: Optional[str] = None
    location: Optional[str] = None
    school: Optional[str] = None
    employer: Optional[str] = None
    twitter_handle: Optional[str] = None  # e.g., "@pratham" or "pratham"
    facebook_url: Optional[str] = None  # direct profile link
    linkedin_url: Optional[str] = None  # direct profile link
    instagram_handle: Optional[str] = None
    website: Optional[str] = None  # personal site or blog
    context: Optional[str] = None  # free-text catch-all for anything else


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
