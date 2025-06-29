from printing import pprint
from audio_samples import AudioSamples
import numpy as np

if __name__ == "__main__":
    pprint("> audio_samples examples!")

    # Create an instance of AudioSamples from a list or ndarray
    audio_samples = AudioSamples([0.1, 0.2, 0.3, 0.4, 0.5], sr=44100, channels=1)
    pprint(f"AudioSamples instance created: {audio_samples}")
    np_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    audio_samples_from_np = AudioSamples(np_array, sr=44100) # channels by default is inferred from shape
    pprint(f"AudioSamples from numpy array: {audio_samples_from_np}")

    # Accessing properties
    pprint(f"Sample rate: {audio_samples.sr}")
    pprint(f"Number of channels: {audio_samples.channels}")
    pprint(f"Duration: {audio_samples.duration} seconds")
    pprint(f"Number of samples: {audio_samples.num_samples}")

    # Operations 
    audio_samples_scaled = audio_samples * 2.0
    audio_samples_normalized = audio_samples.normalize()
    audio_samples_clipped = audio_samples.clip(-0.5, 0.5)
    audio_samples_peak = audio_samples.peak()
    audio_samples_zero_crossings = audio_samples.zero_crossings()
    audio_samples_resampled = audio_samples.resample(16000)
    
