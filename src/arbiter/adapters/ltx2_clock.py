NATIVE_VIDEO_FPS = 25


# ##################################################################
# native frame rate
# reject any request that would introduce a 24/25 conversion.
def native_frame_rate(params: dict) -> int:
    frame_rate = int(params.get("fps", NATIVE_VIDEO_FPS))
    if frame_rate != NATIVE_VIDEO_FPS:
        raise ValueError(f"LTX-2 requires {NATIVE_VIDEO_FPS} fps, got {frame_rate}")
    return frame_rate


# ##################################################################
# lattice frame count
# retain an explicit upstream frame count and enforce LTX's 8n+1 shape.
def lattice_frame_count(params: dict, frame_rate: int, start_time: float, end_time: float) -> int:
    frame_count = int(params.get("num_frames") or round((end_time - start_time) * frame_rate))
    if frame_count < 1 or (frame_count - 1) % 8 != 0:
        raise ValueError(f"LTX-2 frame count must be 8n+1, got {frame_count}")
    return frame_count
