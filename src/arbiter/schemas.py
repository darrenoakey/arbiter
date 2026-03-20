"""Request and response schemas for all Arbiter job types."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Job type enum ---

class JobType(str, Enum):
    IMAGE_GENERATE = "image-generate"
    IMAGE_EDIT = "image-edit"
    BACKGROUND_REMOVE = "background-remove"
    CAPTION = "caption"
    QUERY = "query"
    DETECT = "detect"
    TRANSCRIBE = "transcribe"
    TTS_CUSTOM = "tts-custom"
    TTS_CLONE = "tts-clone"
    TTS_DESIGN = "tts-design"
    TALKING_HEAD = "talking-head"
    VIDEO_GENERATE = "video-generate"


# Maps job type to model_id
JOB_TYPE_TO_MODEL: dict[str, str] = {
    "image-generate": "flux-schnell",
    "image-edit": "flux-schnell",
    "background-remove": "birefnet",
    "caption": "moondream",
    "query": "moondream",
    "detect": "moondream",
    "transcribe": "whisper-large",
    "tts-custom": "tts-custom",
    "tts-clone": "tts-clone",
    "tts-design": "tts-design",
    "talking-head": "sonic",
    "video-generate": "ltx2",
}


# --- Job submission ---

class JobSubmitRequest(BaseModel):
    type: JobType
    params: dict = Field(default_factory=dict)


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str = "queued"
    model: str
    estimated_seconds: Optional[float] = None


# --- Job status ---

class JobState(str, Enum):
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobState
    model: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[dict] = None


# --- System status ---

class ModelStatus(BaseModel):
    id: str
    state: str
    memory_gb: float
    active_jobs: int = 0
    queued_jobs: int = 0
    idle_seconds: Optional[float] = None


class SystemStatus(BaseModel):
    vram_budget_gb: float
    vram_used_gb: float
    models: list[ModelStatus]
    queue: dict[str, int]  # state -> count


class HealthResponse(BaseModel):
    status: str = "ok"
    uptime_seconds: float = 0


# --- Per-job-type parameter schemas (for validation) ---

class ImageGenerateParams(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    aspect_ratio: Optional[str] = None
    steps: int = 4
    seed: int = 42
    transparent: bool = False


class ImageEditParams(BaseModel):
    prompt: str
    image: str  # base64
    width: int = 1024
    height: int = 1024
    steps: int = 4
    seed: int = 42
    strength: float = 0.75
    transparent: bool = False


class BackgroundRemoveParams(BaseModel):
    image: str  # base64


class CaptionParams(BaseModel):
    image: str  # base64
    length: str = "normal"


class QueryParams(BaseModel):
    image: str  # base64
    question: str


class DetectParams(BaseModel):
    image: str  # base64
    object: str


class TranscribeParams(BaseModel):
    audio: str  # base64
    language: Optional[str] = "en"


class TTSCustomParams(BaseModel):
    text: str
    speaker: str = "Aiden"
    language: str = "English"
    temperature: float = 0.9


class TTSCloneParams(BaseModel):
    text: str
    ref_audio: str  # base64
    ref_text: Optional[str] = None
    language: str = "English"
    temperature: float = 0.9


class TTSDesignParams(BaseModel):
    text: str
    voice_description: str = "A clear neutral voice."
    language: str = "English"
    temperature: float = 0.9


class TalkingHeadParams(BaseModel):
    image: str  # base64
    audio: str  # base64
    dynamic_scale: float = 1.0
    seed: Optional[int] = None


class VideoGenerateParams(BaseModel):
    images: list[str]  # list of base64
    audio: str  # base64
    transcript: Optional[str] = None
    resolution: str = "large"
    fps: int = 24
    seed: int = 42


# Maps job type to its parameter validation schema
JOB_TYPE_PARAMS: dict[str, type[BaseModel]] = {
    "image-generate": ImageGenerateParams,
    "image-edit": ImageEditParams,
    "background-remove": BackgroundRemoveParams,
    "caption": CaptionParams,
    "query": QueryParams,
    "detect": DetectParams,
    "transcribe": TranscribeParams,
    "tts-custom": TTSCustomParams,
    "tts-clone": TTSCloneParams,
    "tts-design": TTSDesignParams,
    "talking-head": TalkingHeadParams,
    "video-generate": VideoGenerateParams,
}
