"""Request and response schemas for all Arbiter job types."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


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
    VIDEO_GENERATE_FAST_H3 = "video-generate-fast-h3"
    LTX25_ENCODE = "ltx25-encode"
    LTX25_DENOISE1 = "ltx25-denoise1"
    AESTHETIC_SCORE = "aesthetic-score"
    REFERENCE_IMAGE_EDIT = "reference-image-edit"
    QWEN_IMAGE = "qwen-image"
    QWEN_IMAGE_HERETIC = "qwen-image-heretic"
    TTS_VOXTRAL = "tts-voxtral"
    LORA_TRAIN = "lora-train"
    FINE_TUNE = "fine-tune"
    EMBED_TEXT = "embed-text"
    DEMUCS = "demucs"
    VOCAL_STEM = "vocal-stem"
    RVC_TRAIN = "rvc-train"
    RVC_CONVERT = "rvc-convert"
    VOICE_FIT = "voice-fit"
    MUSIC_GENERATE = "music-generate"
    MUSIC_GENERATE_YUE2 = "music-generate-yue2"
    PHOTO_ENHANCE = "photo-enhance"
    IMAGE_TO_3D = "image-to-3d"


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
    "video-generate-fast-h3": "minimax-fast-h3",
    "ltx25-encode": "ltx25-encode",
    "ltx25-denoise1": "ltx25-denoise1",
    "aesthetic-score": "aesthetic-scorer",
    "reference-image-edit": "reference-image-edit",
    "qwen-image": "qwen-image-2.1",
    "qwen-image-heretic": "qwen-image-2.1-heretic",
    "tts-voxtral": "tts-voxtral",
    "lora-train": "lora-train",
    "fine-tune": "fine-tune",
    "embed-text": "embed-text",
    "demucs": "demucs",
    "vocal-stem": "vocal-stem",
    "rvc-train": "rvc-train",
    "rvc-convert": "rvc-convert",
    "voice-fit": "voice-fit",
    "music-generate": "music-generate",
    "music-generate-yue2": "yue2",
    "photo-enhance": "photo-enhance",
    "image-to-3d": "trellis2",
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
    a2v_guidance_scale: float = 3.0  # stage-1 audio-conditioning guidance; >= 1.0
    # Opt-in. False preserves combined_image_conditionings frame-0 replacement.
    stage1_guiding_keyframes: bool = False

    @field_validator("stage1_guiding_keyframes", mode="before")
    @classmethod
    def _strict_stage1_guiding_keyframes(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(
                "stage1_guiding_keyframes must be a JSON boolean, "
                f"got {type(value).__name__}"
            )
        return value


class AestheticScoreParams(BaseModel):
    image: Optional[str] = None
    image_file: Optional[str] = None


class ReferenceImageEditParams(BaseModel):
    """Reference render of an existing image; an input image is mandatory."""

    prompt: str
    image: Optional[str] = None  # base64
    image_file: Optional[str] = None
    steps: int = 4
    guidance_scale: float = 1.0
    seed: int = 42


class QwenImageParams(BaseModel):
    """Qwen-Image-2.1 unified generation: prompt alone is text-to-image,
    prompt + image is editing/enhancement/multi-reference composition."""

    prompt: str
    image: Optional[str] = None  # base64 condition image (edit mode)
    image_file: Optional[str] = None  # staged path on spark
    negative_prompt: Optional[str] = None  # only used with true_cfg_scale > 1
    steps: int = 40
    true_cfg_scale: float = 1.0  # model is meant to be sampled without guidance
    seed: int = 42
    width: Optional[int] = None  # both or neither; snapped to /16, cap 2752
    height: Optional[int] = None
    output_resolution: int = 1024  # target side for T2I default and edit mode


class ImageTo3DParams(BaseModel):
    image: Optional[str] = None  # base64; RGBA with real alpha skips BiRefNet
    image_file: Optional[str] = None  # staged path on spark
    resolution: Literal[512, 1024, 1536] = 1024
    steps: int = Field(default=12, ge=1, le=50)
    seed: int = Field(default=42, ge=0, le=4294967295)
    texture_size: Literal[1024, 2048, 4096] = 2048
    decimation: int = Field(default=500000, ge=100000, le=1000000)
    include_stl: bool = False  # untextured geometry-only STL beside result.glb


class PhotoEnhanceParams(BaseModel):
    """Full photo-enhance pipeline: detail recovery, reference render, guided
    photometric climb. Everything runs on spark; the caller only submits."""

    image: Optional[str] = None  # base64
    image_file: Optional[str] = None  # staged path on spark
    turns: int = 45
    candidates: int = 6
    detail: bool = True
    seed: int = 42
    stage: str = "full"  # "full" | "probe" (SeedVR2 detail pass only, for bring-up)
    why: Optional[str] = None


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
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    batch_size: int = 4
    grad_accum_steps: int = 4
    num_epochs: int = 1
    max_iters: int = 0
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    save_steps: int = 500
    eval_steps: int = 500
    load_in_4bit: bool = True
    full_finetune: bool = False
    mask_prompt: bool = True
    export_merged: bool = True
    export_format: str = "merged_16bit"
    export_dir: Optional[str] = None
    chat_template: Optional[str] = None


class FineTuneParams(LoraTrainParams):
    pass


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


class VocalStemParams(BaseModel):
    """vocal-stem: Demucs htdemucs separation + loudness normalization.

    audio_file is REQUIRED — an absolute path on spark's local disk (the
    renderer stages its master there). Unlike `demucs`, no base64 input:
    this job exists to normalize the on-disk master, so it operates on the
    file directly and lands outputs in the job's output_dir.
    """

    audio_file: str  # absolute path on spark; must exist
    model: str = "htdemucs"  # the ONLY supported separator; anything else is rejected
    target_lufs: float = -14.0  # integrated-loudness target for vocals_normalized.wav
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

class MusicGenerateParams(BaseModel):
    prompt: str = ""
    lyrics: Optional[str] = None
    audio_duration: float = 30.0
    num_inference_steps: int = 50
    guidance_scale: float = 7.0
    shift: float = 3.0
    vocal_language: str = "en"
    bpm: Optional[int] = None
    keyscale: Optional[str] = None
    timesignature: Optional[str] = None
    seed: Optional[int] = None
    format: str = "mp3"  # mp3 (320kbps, default) | wav | flac | ogg
    model: Optional[str] = None

class Yue2MusicParams(BaseModel):
    style: str = ""  # style/genre prompt, e.g. "funk, upbeat, horns, ..."
    lyrics: Optional[str] = None  # section-tagged lyrics ([Verse], [Chorus], ...)
    cot: str = "full"  # full (melody+chords) | melody | off
    seed: Optional[int] = None
    cfg_scale: Optional[float] = None  # semantic CFG, default 1.0 (1.01 for cot=off)
    abc: Optional[str] = None  # external ABC score (requires cot=melody/full)
    format: str = "mp3"  # mp3 (320kbps, default) | wav | flac | ogg
    model: Optional[str] = None


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
    "reference-image-edit": ReferenceImageEditParams,
    "qwen-image": QwenImageParams,
    "qwen-image-heretic": QwenImageParams,
    "tts-voxtral": TTSVoxtralParams,
    "video-generate-h3": VideoGenerateH3Params,
    "video-generate-fast-h3": VideoGenerateH3Params,
    "lora-train": LoraTrainParams,
    "fine-tune": FineTuneParams,
    "embed-text": EmbedTextParams,
    "demucs": DemucsParams,
    "vocal-stem": VocalStemParams,
    "rvc-train": RvcTrainParams,
    "voice-fit": VoiceFitParams,
    "rvc-convert": RvcConvertParams,
    "music-generate": MusicGenerateParams,
    "music-generate-yue2": Yue2MusicParams,
    "photo-enhance": PhotoEnhanceParams,
    "image-to-3d": ImageTo3DParams,
}
