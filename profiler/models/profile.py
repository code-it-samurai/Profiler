from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from .enums import TargetType


class SocialProfile(BaseModel):
    platform: str
    url: str
    username: Optional[str] = None
    bio: Optional[str] = None
    followers: Optional[int] = None


class Source(BaseModel):
    url: str
    title: Optional[str] = None
    accessed_at: datetime = Field(default_factory=datetime.utcnow)


class NewsMention(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    published_date: Optional[str] = None


class Profile(BaseModel):
    target_name: str
    target_type: TargetType
    summary: str
    social_profiles: list[SocialProfile] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    employment: list[str] = Field(default_factory=list)
    associated_entities: list[str] = Field(default_factory=list)
    news_mentions: list[NewsMention] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    confidence_score: float = 0.0
    candidates_found: int = 0
    candidates_remaining: int = 0
    narrowing_summary: str = ""
    candidate_profiles: list[dict] = Field(default_factory=list)
    compiled_at: datetime = Field(default_factory=datetime.utcnow)
