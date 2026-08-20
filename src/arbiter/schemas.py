"""Request and response schemas for all Arbiter job types."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Job type enum ---


class JobType(str, Enum):
    BACKGROUND_REMOVE = "background-remove"
    CAPTION = "caption"
    QUERY = "query"
    DETECT = "detect"
    POINT = "point"
    TRANSCRIBE = "transcribe"
    TTS_CUSTOM = "tts-custom"
    TTS_CLONE = "tts-clone"
    TTS_DESIGN = "tts-design"
    TTS_KOKORO = "tts-kokoro"
    TALKING_HEAD = "talking-head"
    TALKING_HEAD_SADTALKER = "talking-head-sadtalker"
    LIPSYNC = "lipsync"
    VIDEO_GENERATE = "video-generate"
    VIDEO_GENERATE_H3 = "video-generate-h3"
    LTX25_ENCODE = "ltx25-encode"
    LTX25_DENOISE1 = "ltx25-denoise1"
    AESTHETIC_SCORE = "aesthetic-score"
    TTS_VOXTRAL = "tts-voxtral"
    LORA_TRAIN = "lora-train"
    EMBED_TEXT = "embed-text"
    DEMUCS = "demucs"
    RVC_TRAIN = "rvc-train"
    RVC_CONVERT = "rvc-convert"
    VOICE_FIT = "voice-fit"


# Maps job type to model_id
JOB_TYPE_TO_MODEL: dict[str, str] = {
    "background-remove": "birefnet",
    "caption": "moondream",
    "query": "moondream",
    "detect": "moondream",
    "point": "moondream",
    "transcribe": "whisper-large",
    "tts-custom": "tts-custom",
    "tts-clone": "tts-clone",
    "tts-design": "tts-design",
    "tts-kokoro": "tts-kokoro",
    "talking-head": "sonic",
    "talking-head-sadtalker": "sadtalker",
    "lipsync": "latentsync",
    "video-generate": "ltx2",
    "video-generate-h3": "minimax-h3-local",
    "ltx25-encode": "ltx25-encode",
    "ltx25-denoise1": "ltx25-denoise1",
    "aesthetic-score": "aesthetic-scorer",
    "tts-voxtral": "tts-voxtral",
    "lora-train": "lora-train",
    "embed-text": "embed-text",
    "demucs": "demucs",
    "rvc-train": "rvc-train",
    "rvc-convert": "rvc-convert",
    "voice-fit": "voice-fit",
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


class TTSKokoroParams(BaseModel):
    # Single-line mode: text (+ voice/speed). Batch mode: items=[{text,voice,speed}]
    # synthesized in one job and returned as one concatenated wav + item_samples.
    text: str = ""
    voice: str = "af_heart"  # name or weighted blend "af_heart*0.6+am_michael*0.4"
    speed: float = 1.0
    lang_code: str = ""  # "" → derive from voice prefix (a/b/...)
    items: Optional[list[dict]] = None
    gap_seconds: float = 0.0


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
    fps: int = 25
    seed: int = 42
    chunk_frames: int = 121


class VideoGenerateH3Params(BaseModel):
    prompt: str = ""
    first_image_b64: str = ""
    last_image_b64: str = ""
    first_image_file: Optional[str] = None
    last_image_file: Optional[str] = None
    duration: int = 6
    width: int = 960
    height: int = 544
    seed: int = 42
    num_inference_steps: int = 8


class LTX25EncodeParams(BaseModel):
    prompt: str = ""
    description: Optional[str] = None
    negative_prompt: Optional[str] = None
    audio_file: str
    audio_start_time: float = 0.0
    audio_duration: float
    image_file: Optional[str] = None
    num_frames: int
    height: int = 1088
    width: int = 1920
    fps: float = 25.0
    seed: int = 42
    chunk_index: int = 0


class LTX25Denoise1Params(BaseModel):
    encoded_file: str
    audio_file: str
    start_time: float = 0.0
    fps: float = 25.0
    num_inference_steps: int = 30


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


class EmbedTextParams(BaseModel):
    texts: Optional[list[str]] = None
    text: Optional[str] = None
    task: str = "search_document"
    batch_size: int = 16


class DemucsParams(BaseModel):
    audio: Optional[str] = None  # base64
    audio_file: Optional[str] = None
    return_b64: bool = False  # inline vocals/accompaniment base64 in the result
    duration: Optional[float] = None  # optional hint for time estimation


class RvcTrainParams(BaseModel):
    name: str  # voice/model id (sanitized -> stable model dir)
    dataset_b64: Optional[str] = None  # zip of wavs, or a single wav
    dataset_file: Optional[str] = None  # path to a dir of wavs, a zip, or a wav
    epochs: int = 300
    sample_rate: int = 40000
    batch_size: int = 4
    f0_method: str = "rmvpe"


class VoiceFitParams(BaseModel):
    name: str  # voice id -> /home/darren/voice-models/<name>.pt
    targets_dir: Optional[str] = None  # spark-local dir of *.wav + sibling *.txt
    targets_file: Optional[str] = None  # staged zip (or dir) of the same layout
    seed_voice: str = "auto"  # stock voice name, or "auto" embedding search
    exclude: Optional[str] = None  # comma-separated voices excluded from auto seed
    init_pack: Optional[str] = None  # prior fit name under voice-models/, or abs .pt path
    steps: int = 300
    lr: float = 0.05
    w_self: float = 0.5
    w_reg: float = 1.0
    eval_every: int = 10


class RvcConvertParams(BaseModel):
    model: str  # trained model id or absolute .pth path
    audio: Optional[str] = None  # base64
    audio_file: Optional[str] = None
    transpose: int = 0  # semitone pitch shift
    index_rate: float = 0.5
    f0_method: str = "rmvpe"
    protect: float = 0.33
    index_path: Optional[str] = None
    return_b64: bool = False


# Maps job type to its parameter validation schema
JOB_TYPE_PARAMS: dict[str, type[BaseModel]] = {
    "background-remove": BackgroundRemoveParams,
    "caption": CaptionParams,
    "query": QueryParams,
    "detect": DetectParams,
    "point": PointParams,
    "transcribe": TranscribeParams,
    "tts-custom": TTSCustomParams,
    "tts-clone": TTSCloneParams,
    "tts-design": TTSDesignParams,
    "tts-kokoro": TTSKokoroParams,
    "talking-head": TalkingHeadParams,
    "talking-head-sadtalker": TalkingHeadSadTalkerParams,
    "lipsync": LipsyncParams,
    "video-generate": VideoGenerateParams,
    "ltx25-encode": LTX25EncodeParams,
    "ltx25-denoise1": LTX25Denoise1Params,
    "aesthetic-score": AestheticScoreParams,
    "tts-voxtral": TTSVoxtralParams,
    "video-generate-h3": VideoGenerateH3Params,
    "lora-train": LoraTrainParams,
    "embed-text": EmbedTextParams,
    "demucs": DemucsParams,
    "rvc-train": RvcTrainParams,
    "voice-fit": VoiceFitParams,
    "rvc-convert": RvcConvertParams,
}
