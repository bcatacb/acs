from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import Optional, List
from datetime import datetime, timezone
import uuid

# Auth Models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    username: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    username: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User

# Project Models
class ProjectCreate(BaseModel):
    name: str
    genre: str = "trap"
    instrument_pack: str = "auto"

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    genre: Optional[str] = None
    instrument_pack: Optional[str] = None

class AudioAnalysis(BaseModel):
    bpm: int
    flow_style: str
    cadence: str
    mood: str
    lyric_density: str
    suggested_genres: List[str]
    transcript: Optional[str] = None

class BeatGeneration(BaseModel):
    task_id: str
    status: str
    audio_url: Optional[str] = None
    image_url: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None

class Project(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str
    genre: str
    instrument_pack: str = "auto"
    acapella_url: Optional[str] = None
    analysis: Optional[AudioAnalysis] = None
    beat: Optional[BeatGeneration] = None
    status: str = "draft"  # draft, analyzing, generating, complete, error
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Genre Options
GENRES = [
    {"id": "trap", "name": "Trap", "description": "Heavy 808s, hi-hats, dark synths"},
    {"id": "boom_bap", "name": "Boom Bap", "description": "Classic hip-hop drums, samples"},
    {"id": "drill", "name": "Drill", "description": "Dark, aggressive, sliding 808s"},
    {"id": "lo_fi", "name": "Lo-Fi", "description": "Chill, jazzy, vinyl crackle"},
    {"id": "west_coast", "name": "West Coast", "description": "G-funk synths, bouncy bass"},
    {"id": "east_coast", "name": "East Coast", "description": "Hard drums, piano samples"},
    {"id": "southern", "name": "Southern", "description": "Crunk, bounce, heavy bass"},
    {"id": "melodic", "name": "Melodic Rap", "description": "Emotional melodies, ambient pads"}
]

class GenreResponse(BaseModel):
    genres: List[dict]
