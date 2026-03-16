from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional
from .enums import Platform


class CandidateProfile(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    platform: Platform = Platform.GENERIC
    profile_url: Optional[str] = None
    location: Optional[str] = None
    school: Optional[str] = None
    employer: Optional[str] = None
    bio: Optional[str] = None
    profile_photo_url: Optional[str] = None
    raw_data: dict = Field(default_factory=dict)
    source_urls: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
