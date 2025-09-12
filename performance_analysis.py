#!/usr/bin/env python3

import numpy as np
import librosa
import audio_samples as aus
import time

# Test file path
DEMO_WAV = "test_resources/synthetic_test.wav"

print("Performance Analysis: audio_samples vs librosa")
print("=" * 50)

# Load data once for both tests
print("Loading audio data...")
y, sr = librosa.load(DEMO_WAV, sr=None, mono=True)
audio = aus.load(DEMO_WAV, dtype=np.float32)

print(f"Audio length: {len(y)} samples, Sample rate: {sr} Hz")
print(f"Duration: {len(y) / sr:.2f} seconds")
print()

# Test parameters
n_fft = 2048
hop_length = 512
window = "hann"
n_trials = 5


def librosa_spec(y, sr):
    """Librosa implementation"""
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    S_db = librosa.power_to_db(np.abs(D) ** 2, ref=np.max(np.abs(D) ** 2))
    return S_db


def audio_samples_spec(audio):
    """Audio samples implementation"""
    return audio.spectrogram_db(
        n_fft=n_fft, hop_length=hop_length, window=window, top_db=80.0
    )


# Benchmark librosa
print("Benchmarking librosa...")
librosa_times = []
for i in range(n_trials):
    start = time.time()
    librosa_result = librosa_spec(y, sr)
    end = time.time()
    librosa_times.append(end - start)
    print(f"  Trial {i + 1}: {(end - start) * 1000:.1f} ms")

# Benchmark audio_samples
print("\nBenchmarking audio_samples...")
audio_samples_times = []
for i in range(n_trials):
    start = time.time()
    audio_samples_result = audio_samples_spec(audio)
    end = time.time()
    audio_samples_times.append(end - start)
    print(f"  Trial {i + 1}: {(end - start) * 1000:.1f} ms")

# Results
print("\n" + "=" * 50)
print("RESULTS:")
librosa_avg = np.mean(librosa_times) * 1000
audio_samples_avg = np.mean(audio_samples_times) * 1000
librosa_std = np.std(librosa_times) * 1000
audio_samples_std = np.std(audio_samples_times) * 1000

print(f"Librosa:      {librosa_avg:.1f} ± {librosa_std:.1f} ms")
print(f"Audio samples: {audio_samples_avg:.1f} ± {audio_samples_std:.1f} ms")
print(f"Ratio:        {audio_samples_avg / librosa_avg:.2f}x slower")

# Check output shapes and ranges
print(f"\nOutput verification:")
print(f"Librosa shape:      {librosa_result.shape}")
print(f"Audio samples shape: {audio_samples_result.shape}")
print(
    f"Librosa range:      [{librosa_result.min():.1f}, {librosa_result.max():.1f}] dB"
)
print(
    f"Audio samples range: [{audio_samples_result.min():.1f}, {audio_samples_result.max():.1f}] dB"
)

# Now let's break down the audio_samples implementation step by step
print("\n" + "=" * 50)
print("DETAILED BREAKDOWN (audio_samples only):")


def benchmark_stft_only(audio):
    """Just STFT computation"""
    return audio.stft(n_fft=n_fft, hop_length=hop_length, window=window)


def benchmark_stft_to_power(audio):
    """STFT + power computation"""
    stft = audio.stft(n_fft=n_fft, hop_length=hop_length, window=window)
    return np.abs(stft) ** 2


print("\nSTFT only:")
stft_times = []
for i in range(n_trials):
    start = time.time()
    stft_result = benchmark_stft_only(audio)
    end = time.time()
    stft_times.append(end - start)
    print(f"  Trial {i + 1}: {(end - start) * 1000:.1f} ms")

print(
    f"STFT average: {np.mean(stft_times) * 1000:.1f} ± {np.std(stft_times) * 1000:.1f} ms"
)

print("\nSTFT + Power computation:")
power_times = []
for i in range(n_trials):
    start = time.time()
    power_result = benchmark_stft_to_power(audio)
    end = time.time()
    power_times.append(end - start)
    print(f"  Trial {i + 1}: {(end - start) * 1000:.1f} ms")

print(
    f"STFT+Power average: {np.mean(power_times) * 1000:.1f} ± {np.std(power_times) * 1000:.1f} ms"
)

# Component analysis
stft_portion = np.mean(stft_times) / np.mean(audio_samples_times) * 100
power_portion = (
    (np.mean(power_times) - np.mean(stft_times)) / np.mean(audio_samples_times) * 100
)
db_portion = 100 - stft_portion - power_portion

print(f"Component breakdown:")
print(f"STFT computation:    {stft_portion:.1f}% of total time")
print(f"Power computation:   {power_portion:.1f}% of total time")
print(f"dB conversion:       {db_portion:.1f}% of total time")
