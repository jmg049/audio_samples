import os
from pathlib import Path
from typing import Optional

import audio_samples

import numpy as np
import soundfile as sf

NUMPY_TYPE_TO_SF_COMPATIBLE_TYPE = {
    np.int16: "int16",
    np.int32: "int32",
    np.float32: "float32",
    np.float64: "float64",
}


def _determine_dtype(fp: Path | str) -> np.dtype:
    """Determine the data type of the audio file.
    Args:
        fp: Path to the audio file.
    Returns:
        np.dtype: Data type of the audio samples in the file.
    """
    with sf.SoundFile(fp) as f:
        subtype = f.subtype
    subtype_to_dtype = {
        "PCM_16": np.int16,
        "PCM_24": np.int32,
        "PCM_32": np.int32,
        "FLOAT": np.float32,
        "DOUBLE": np.float64,
        "ULAW": np.uint8,
        "ALAW": np.uint8,
    }
    if subtype not in subtype_to_dtype:
        raise ValueError(f"Unsupported audio subtype: {subtype}")
    return subtype_to_dtype[subtype]


def load(
    fp: Path | str, 
    *, 
    dtype: Optional[np.dtype] = None,
    start_time: Optional[float] = None,
    stop_time: Optional[float] = None,
    duration: Optional[float] = None,
    start_frame: Optional[int] = None,
    stop_frame: Optional[int] = None,
    frames: Optional[int] = None
) -> audio_samples.AudioSamples:
    """Load audio samples from a file with optional time/frame range specification.
    
    Args:
        fp: Path to the audio file.
        dtype: Desired data type of the returned samples. If None, the original data type is preserved.
        start_time: Start time in seconds. Cannot be used with start_frame.
        stop_time: Stop time in seconds. Cannot be used with stop_frame or duration.
        duration: Duration in seconds from start_time. Cannot be used with stop_time or frames.
        start_frame: Start frame index. Cannot be used with start_time.
        stop_frame: Stop frame index. Cannot be used with stop_time or frames.
        frames: Number of frames to read. Cannot be used with stop_time, stop_frame, or duration.
        
    Returns:
        AudioSamples: Loaded audio samples.
        
    Raises:
        ValueError: If conflicting time/frame parameters are specified.
    """
    global NUMPY_TYPE_TO_SF_COMPATIBLE_TYPE

    fp = Path(fp)
    
    # Validate parameter combinations
    time_params = [start_time, stop_time, duration]
    frame_params = [start_frame, stop_frame, frames]
    
    if any(t is not None for t in time_params) and any(f is not None for f in frame_params):
        raise ValueError("Cannot mix time-based and frame-based parameters")
    
    if start_time is not None and start_frame is not None:
        raise ValueError("Cannot specify both start_time and start_frame")
        
    if sum(x is not None for x in [stop_time, duration, frames]) > 1:
        raise ValueError("Cannot specify more than one of: stop_time, duration, frames")
        
    if sum(x is not None for x in [stop_frame, frames]) > 1:
        raise ValueError("Cannot specify both stop_frame and frames")

    if dtype is None:
        dtype = _determine_dtype(fp)

    # Get sample rate for time-based calculations
    with sf.SoundFile(fp) as f:
        file_sample_rate = f.samplerate
        total_frames = f.frames

    # Convert time-based parameters to frame-based
    sf_start = None
    sf_stop = None  
    sf_frames = None
    
    if start_time is not None:
        sf_start = int(start_time * file_sample_rate)
    elif start_frame is not None:
        sf_start = start_frame
        
    if stop_time is not None:
        sf_stop = int(stop_time * file_sample_rate)
    elif stop_frame is not None:
        sf_stop = stop_frame
    elif duration is not None:
        if sf_start is None:
            sf_start = 0
        sf_frames = int(duration * file_sample_rate)
    elif frames is not None:
        sf_frames = frames

    # Validate frame ranges
    if sf_start is not None and sf_start < 0:
        sf_start = max(0, total_frames + sf_start)
    if sf_stop is not None and sf_stop < 0:
        sf_stop = max(0, total_frames + sf_stop)
        
    if sf_start is not None and sf_start >= total_frames:
        raise ValueError(f"start frame {sf_start} exceeds file length {total_frames}")
    if sf_stop is not None and sf_stop > total_frames:
        raise ValueError(f"stop frame {sf_stop} exceeds file length {total_frames}")

    # Read with appropriate parameters
    read_kwargs = {"dtype": NUMPY_TYPE_TO_SF_COMPATIBLE_TYPE[dtype]}
    if sf_start is not None:
        read_kwargs["start"] = sf_start
    if sf_stop is not None:
        read_kwargs["stop"] = sf_stop
    elif sf_frames is not None:
        read_kwargs["frames"] = sf_frames

    data, sample_rate = sf.read(fp, **read_kwargs)
    data = audio_samples.from_numpy(data, sample_rate=sample_rate)

    return data


def save(
    fp: Path | str, audio: audio_samples.AudioSamples, dtype: Optional[np.dtype] = None
) -> None:
    """Save audio samples to a file.
    Args:
        fp: Path to the output audio file.
        samples: AudioSamples object containing the samples to save.
    """

    fp = Path(fp)

    if not os.path.exists(fp.parent):
        os.makedirs(fp.parent, exist_ok=True)

    if dtype is not None:
        audio.astype(dtype)

    dtype = audio.dtype

    sf.write(fp, audio.numpy(dtype=dtype, copy=True), samplerate=audio.sample_rate)
