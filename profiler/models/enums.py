from enum import Enum


class TargetType(str, Enum):
    PERSON = "person"
    COMPANY = "company"


class Platform(str, Enum):
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    INSTAGRAM = "instagram"
    GENERIC = "generic"


class SessionStatus(str, Enum):
    SEARCHING = "searching"
    NARROWING = "narrowing"
    ASKING_USER = "asking_user"
    COMPILING = "compiling"
    DONE = "done"
    FAILED = "failed"
