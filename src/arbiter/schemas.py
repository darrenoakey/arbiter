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
    POINT = "point"
    TRANSCRIBE = "transcribe"
    TTS_CUSTOM = "tts-custom"
    TTS_CLONE = "tts-clone"
    TTS_DESIGN = "tts-design"
    TALKING_HEAD = "talking-head"
    TALKING_HEAD_SADTALKER = "talking-head-sadtalker"
    LIPSYNC = "lipsync"
    VIDEO_GENERATE = "video-generate"
    AESTHETIC_SCORE = "aesthetic-score"
    TTS_VOXTRAL = "tts-voxtral"
    LORA_TRAIN = "lora-train"


# Maps job type to model_id
JOB_TYPE_TO_MODEL: dict[str, str] = {
    "image-generate": "flux-schnell",
    "image-edit": "flux-schnell",
    "background-remove": "birefnet",
    "caption": "moondream",
    "query": "moondream",
    "detect": "moondream",
    "point": "moondream",
    "transcribe": "whisper-large",
    "tts-custom": "tts-custom",
    "tts-clone": "tts-clone",
    "tts-design": "tts-design",
    "talking-head": "sonic",
    "talking-head-sadtalker": "sadtalker",
    "lipsync": "latentsync",
    "video-generate": "ltx2",
    "aesthetic-score": "aesthetic-scorer",
    "tts-voxtral": "tts-voxtral",
    "lora-train": "lora-train",
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
    image: Optional[str] = None  # base64
    image_file: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 4
    seed: int = 42
    strength: float = 0.75
    transparent: bool = False


class BackgroundRemoveParams(BaseModel):
    image: Optional[str] = None
    image_file: Optional[str] = None


class CaptionParams(BaseModel):
    image: Optional[str] = None  # base64
    image_file: Optional[str] = None  # local path on spark
    length: str = "normal"


class QueryParams(BaseModel):
    image: Optional[str] = None
    image_file: Optional[str] = None
    question: str


class DetectParams(BaseModel):
    image: Optional[str] = None
    image_file: Optional[str] = None
    object: str


class PointParams(BaseModel):
    image: Optional[str] = None
    image_file: Optional[str] = None
    object: str


class TranscribeParams(BaseModel):
    audio: Optional[str] = None  # base64
    audio_file: Optional[str] = None
    language: Optional[str] = "en"


class TTSCustomParams(BaseModel):
    text: str
    speaker: str = "Aiden"
    language: str = "English"
    temperature: float = 0.9


class TTSCloneParams(BaseModel):
    text: str
    ref_audio: Optional[str] = None  # base64
    ref_audio_file: Optional[str] = None
    ref_text: Optional[str] = None
    language: str = "English"
    temperature: float = 0.9


class TTSDesignParams(BaseModel):
    text: str
    voice_description: str = "A clear neutral voice."
    language: str = "English"
    temperature: float = 0.9


class TalkingHeadParams(BaseModel):
    image: Optional[str] = None
    image_file: Optional[str] = None
    audio: Optional[str] = None
    audio_file: Optional[str] = None
    dynamic_scale: float = 1.0
    seed: Optional[int] = None


class TalkingHeadSadTalkerParams(BaseModel):
    image: Optional[str] = None
    image_file: Optional[str] = None
    audio: Optional[str] = None
    audio_file: Optional[str] = None
    size: int = 256
    facerender: str = "pirender"
    expression_scale: float = 1.0
    preprocess: str = "crop"
    enhancer: str = ""
    still: bool = False


class LipsyncParams(BaseModel):
    video: Optional[str] = None
    video_file: Optional[str] = None
    audio: Optional[str] = None
    audio_file: Optional[str] = None
    inference_steps: int = 20
    guidance_scale: float = 1.5


class VideoSegmentParams(BaseModel):
    description: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    start_image_b64: str = ""
    end_image_b64: str = ""


class VideoGenerateParams(BaseModel):
    segments: list[VideoSegmentParams]
    audio_b64: str  # base64-encoded audio file
    resolution: str = "large"
    fps: int = 24
    seed: int = 42
    chunk_frames: int = 121


class AestheticScoreParams(BaseModel):
    image: Optional[str] = None
    image_file: Optional[str] = None



class TTSVoxtralParams(BaseModel):
    text: str
    voice: str = "alloy"
    language: str = "English"
    temperature: float = 0.7
    speed: float = 1.0



class LoraTrainParams(BaseModel):
    data_dir: str
    model_name: str
    run_name: Optional[str] = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 4
    grad_accum_steps: int = 4
    num_epochs: int = 1
    max_iters: int = 0
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    save_steps: int = 500
    eval_steps: int = 500
    load_in_4bit: bool = True
    full_finetune: bool = False
    chat_template: Optional[str] = None


# Maps job type to its parameter validation schema
JOB_TYPE_PARAMS: dict[str, type[BaseModel]] = {
    "image-generate": ImageGenerateParams,
    "image-edit": ImageEditParams,
    "background-remove": BackgroundRemoveParams,
    "caption": CaptionParams,
    "query": QueryParams,
    "detect": DetectParams,
    "point": PointParams,
    "transcribe": TranscribeParams,
    "tts-custom": TTSCustomParams,
    "tts-clone": TTSCloneParams,
    "tts-design": TTSDesignParams,
    "talking-head": TalkingHeadParams,
    "talking-head-sadtalker": TalkingHeadSadTalkerParams,
    "lipsync": LipsyncParams,
    "video-generate": VideoGenerateParams,
    "aesthetic-score": AestheticScoreParams,
    "tts-voxtral": TTSVoxtralParams,
    "lora-train": LoraTrainParams,
}
