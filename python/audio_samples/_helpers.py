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
    fp: Path | str, *, dtype: Optional[np.dtype] = None
) -> audio_samples.AudioSamples:
    """Load audio samples from a file.
    Args:
        fp: Path to the audio file.
        dtype: Desired data type of the returned samples. If None, the original data type is preserved.
    Returns:

        AudioSamples: Loaded audio samples.
    """
    global NUMPY_TYPE_TO_SF_COMPATIBLE_TYPE

    fp = Path(fp)
    if dtype is None:
        dtype = _determine_dtype(fp)

    data, sample_rate = sf.read(
        fp, dtype=NUMPY_TYPE_TO_SF_COMPATIBLE_TYPE[dtype]
    )  # Double check the output of str(dtype), soundfile expects 'int16', 'float32', etc
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

    sf.write(fp, audio.to_numpy(dtype=dtype, copy=True), samplerate=audio.sample_rate)
