from dataclasses import dataclass
import math
from typing import Any, Literal
import matplotlib.pyplot as plt
import seaborn as sns

from audio_samples import AudioSamples

from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


def _estimate_target_ticks(
    fig_width_in: float, dpi: int = 100, min_spacing_px: int = 110
) -> int:
    total_px = max(1.0, fig_width_in) * max(1, dpi)
    return max(3, int(round(total_px / min_spacing_px)))  # 3..10 is typical


def _nice_step_seconds(rough: float) -> float:
    # 1–2–5 heuristic
    if rough <= 0 or not np.isfinite(rough):
        return 1.0
    exp = np.floor(np.log10(rough))
    base = 10.0**exp
    mant = rough / base
    if mant <= 1.0:
        m = 1.0
    elif mant <= 2.0:
        m = 2.0
    elif mant <= 5.0:
        m = 5.0
    else:
        m = 10.0
    return m * base


def _choose_time_format(duration_s: float) -> tuple[str, callable]:
    """Return (xlabel_unit, formatter(seconds)->str)."""
    if duration_s >= 3600:  # hours

        def fmt(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = t % 60
            return f"{h:d}:{m:02d}:{s:02.0f}"

        return "Time (h:mm:ss)", fmt
    elif duration_s >= 60:  # minutes

        def fmt(t):
            m = int(t // 60)
            s = t % 60
            return f"{m:d}:{s:02.0f}"

        return "Time (m:ss)", fmt
    else:  # seconds

        def fmt(t):
            return f"{t:.2f}"

        return "Time (s)", fmt


# --- choose a time-friendly step (seconds) ---
_CLOCK_STEPS = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    15,
    20,
    30,
    60,
    120,
    180,
    300,
    600,
    900,
    1200,
    1800,
    3600,
    7200,
    10800,
]


def _pick_time_step(duration_s: float, target_ticks: int) -> float:
    """
    Pick a 'clock-friendly' step so the number of intervals is close to target_ticks-1.
    Prefers smaller steps on ties (more ticks rather than fewer).
    """
    desired_intervals = max(1, target_ticks - 1)
    best = _CLOCK_STEPS[0]
    best_cost = float("inf")
    for step in _CLOCK_STEPS:
        intervals = max(1, math.ceil(duration_s / step))
        cost = abs(intervals - desired_intervals)
        if cost < best_cost or (cost == best_cost and step < best):
            best_cost, best = cost, step
    return best


def _seconds_ticks(duration_s: float, target_ticks: int) -> tuple[list[float], float]:
    """
    Generate ticks from 0 with the chosen step, ensuring the LAST tick >= duration.
    """
    step = _pick_time_step(duration_s, target_ticks)
    n_intervals = max(1, math.ceil(duration_s / step))
    return [k * step for k in range(n_intervals + 1)], step


# ---------- Y‑axis helpers ----------

_DTYPE_MAX = {
    np.dtype("int16"): 32768.0,
    np.dtype("int32"): 2147483648.0,
    np.dtype("uint8"): 128.0,  # typical PCM offset; adjust if your loader recentres
}


def _to_float_minus1_1(arr: np.ndarray) -> np.ndarray:
    """Return a float copy in [-1, 1] if integer‑typed; else a float view/copy as‑is."""
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        scale = _DTYPE_MAX.get(arr.dtype, float(np.iinfo(arr.dtype).max))
        return (arr.astype(np.float64, copy=False) / scale).clip(-1.0, 1.0)
    # already float: ensure float64 for plotting stability
    return arr.astype(np.float64, copy=False)


def _format_seconds_label(t: float) -> str:
    # mm:ss(.ms) formatting with sensible precision
    if t >= 3600:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:d}:{m:02d}:{s:06.3f}"
    elif t >= 60:
        m = int(t // 60)
        s = t % 60
        return f"{m:d}:{s:06.3f}" if t < 600 else f"{m:d}:{s:05.2f}"
    else:
        return f"{t:.3f}" if t < 10 else f"{t:.2f}"


@dataclass
class WaveformPlotOptions:
    figsize: tuple[int, int] = (10, 6)
    use_seaborn_style: bool = True
    seaborn_style: (
        dict[str, Any] | Literal["white", "dark", "whitegrid", "darkgrid", "ticks"]
    ) = "darkgrid"
    seaborn_context: dict[str, Any] | Literal["paper", "notebook", "talk", "poster"] = (
        "notebook"
    )
    use_tex: bool = False
    font_family: str = "serif"  # explicit control
    serif_fonts: tuple[str, ...] = ("Times New Roman", "DejaVu Serif", "Georgia")
    sans_fonts: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans")
    wave_color: str = "#6ff025"
    title: str = "Waveform"
    xlabel: str = "Time (s)"
    ylabel: str = "Amplitude"
    grid: bool = True
    grid_linestyle: str = "--"
    grid_linewidth: float = 0.75
    grid_alpha: float = 0.8
    ylim: tuple[float, float] | None = None
    save_path: str | None = None
    title_fontsize: int = 16
    label_fontsize: int = 14
    tick_fontsize: int = 12
    wave_linewidth: float = 1.0
    wave_alpha: float = 0.8


@dataclass
class SpectrogramPlotOptions:
    figsize: tuple[int, int] = (12, 8)
    use_seaborn_style: bool = True
    seaborn_style: (
        dict[str, Any] | Literal["white", "dark", "whitegrid", "darkgrid", "ticks"]
    ) = "darkgrid"
    seaborn_context: dict[str, Any] | Literal["paper", "notebook", "talk", "poster"] = (
        "notebook"
    )
    use_tex: bool = False
    font_family: str = "serif"
    serif_fonts: tuple[str, ...] = ("Times New Roman", "DejaVu Serif", "Georgia")
    sans_fonts: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans")
    colormap: str = "viridis"
    title: str = "Spectrogram"
    xlabel: str = "Time (s)"
    ylabel: str = "Frequency (Hz)"
    grid: bool = False
    save_path: str | None = None
    title_fontsize: int = 16
    label_fontsize: int = 14
    tick_fontsize: int = 12
    colorbar: bool = True
    colorbar_label: str = "Power (dB)"
    # STFT parameters (librosa-style)
    n_fft: int = 2048
    hop_length: int | None = None  # Default to n_fft // 4
    window: str = "hann"
    # Legacy parameters for backward compatibility
    window_size: int | None = None  # Will be mapped to n_fft if provided
    hop_size: int | None = None  # Will be mapped to hop_length if provided
    # Display parameters
    log_scale: bool = True
    db_range: tuple[float, float] = (-80, 0)
    freq_scale: Literal["linear", "log"] = "linear"


@dataclass
class ComparisonPlotOptions:
    figsize: tuple[int, int] = (14, 10)
    use_seaborn_style: bool = True
    seaborn_style: (
        dict[str, Any] | Literal["white", "dark", "whitegrid", "darkgrid", "ticks"]
    ) = "darkgrid"
    seaborn_context: dict[str, Any] | Literal["paper", "notebook", "talk", "poster"] = (
        "notebook"
    )
    use_tex: bool = False
    font_family: str = "serif"
    serif_fonts: tuple[str, ...] = ("Times New Roman", "DejaVu Serif", "Georgia")
    sans_fonts: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans")
    colors: tuple[str, ...] = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")
    title: str = "Audio Comparison"
    xlabel: str = "Time (s)"
    ylabel: str = "Amplitude"
    grid: bool = True
    grid_linestyle: str = "--"
    grid_linewidth: float = 0.75
    grid_alpha: float = 0.8
    ylim: tuple[float, float] | None = None
    save_path: str | None = None
    title_fontsize: int = 16
    label_fontsize: int = 14
    tick_fontsize: int = 12
    wave_linewidth: float = 1.0
    wave_alpha: float = 0.7
    legend: bool = True
    legend_loc: str = "upper right"


def _font_rc(
    options: WaveformPlotOptions | SpectrogramPlotOptions | ComparisonPlotOptions,
) -> dict[str, Any]:
    if options.use_tex:
        return {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],  # handled by LaTeX
        }
    if options.font_family == "serif":
        return {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],  # installed system fonts
        }
    return {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
    }


def waveform(
    audio: AudioSamples, options: WaveformPlotOptions = WaveformPlotOptions()
) -> None:
    """Display the waveform of the audio samples."""
    rc = {
        "figure.figsize": options.figsize,
        "axes.titlesize": options.title_fontsize,
        "axes.labelsize": options.label_fontsize,
        "xtick.labelsize": options.tick_fontsize,
        "ytick.labelsize": options.tick_fontsize,
        **_font_rc(options),
    }

    if options.use_seaborn_style:
        sns.set_theme(
            style=options.seaborn_style, context=options.seaborn_context, rc=rc
        )
    else:
        plt.rcParams.update(rc)

    num_channels = audio.num_channels
    fig, axes = plt.subplots(
        num_channels,
        1,
        figsize=options.figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    # duration and sample rate
    sr = float(audio.sample_rate)
    duration_s = audio.duration

    # Y scaling & ticks: normalise integers and add enough y‑ticks
    # We'll compute limits from the first channel and reuse (sharey=True).
    first = _to_float_minus1_1(audio.extract_channel(0).numpy())
    # symmetric limits around 0 with a small headroom
    ymax = float(np.max(np.abs(first)))
    ylim = options.ylim or (-1.05 * ymax, 1.05 * ymax)

    for ch in range(num_channels):
        ax = axes[ch, 0]
        raw = audio.extract_channel(ch).numpy()
        y = _to_float_minus1_1(raw)

        ax.plot(
            y,
            color=options.wave_color,
            linewidth=options.wave_linewidth,
            alpha=options.wave_alpha,
        )

        # figure-based tick target (as before)
        target = _estimate_target_ticks(options.figsize[0], dpi=100, min_spacing_px=120)

        # ticks + step
        (ticks_s, step_s) = _seconds_ticks(duration_s, target)
        tick_pos = [int(round(t * sr)) for t in ticks_s]
        xlabel, tick_fmt_fn = _choose_time_format(duration_s)
        tick_labels = [tick_fmt_fn(t) for t in ticks_s]  # your formatter from earlier

        # x-limits: pad a bit before 0, and extend to the LAST tick (not just last sample)
        left_pad_s = min(0.5, 0.5 * step_s)  # up to 0.5 s, scaled by step
        x_start = -int(round(left_pad_s * sr))
        x_end = int(round(ticks_s[-1] * sr))  # ensures 9:00 appears for an 8:52 file

        ax.set_xlim(x_start, x_end)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        # y ticks & grid
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=7, prune=None, steps=[1, 2, 5, 10])
        )
        if options.grid:
            ax.grid(
                True,
                linestyle=options.grid_linestyle,
                linewidth=options.grid_linewidth,
                alpha=options.grid_alpha,
            )

        ax.set_ylabel(
            f"Channel {ch + 1}" if num_channels > 1 else options.ylabel,
            fontsize=options.label_fontsize,
        )

    # label with unit **once** on the bottom axis
    axes[-1, 0].set_xlabel(xlabel, fontsize=options.label_fontsize)

    # Optional: nicer formatting of y tick labels (fixed decimals)
    axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

    if num_channels > 1:
        fig.suptitle(options.title, fontsize=options.title_fontsize)
    else:
        axes[0, 0].set_title(options.title, fontsize=options.title_fontsize)

    plt.tight_layout()
    if options.save_path:
        plt.savefig(options.save_path, dpi=300)
    plt.show()
    plt.close()


def spectrogram(
    audio: AudioSamples, options: SpectrogramPlotOptions = SpectrogramPlotOptions()
) -> None:
    """Display the spectrogram of the audio samples."""
    rc = {
        "figure.figsize": options.figsize,
        "axes.titlesize": options.title_fontsize,
        "axes.labelsize": options.label_fontsize,
        "xtick.labelsize": options.tick_fontsize,
        "ytick.labelsize": options.tick_fontsize,
        **_font_rc(options),
    }

    if options.use_seaborn_style:
        sns.set_theme(
            style=options.seaborn_style, context=options.seaborn_context, rc=rc
        )
    else:
        plt.rcParams.update(rc)

    num_channels = audio.num_channels
    fig, axes = plt.subplots(
        num_channels,
        1,
        figsize=options.figsize,
        squeeze=False,
        sharex=True,
    )

    # Resolve STFT parameters with librosa priority
    n_fft = (
        options.window_size or options.n_fft
    )  # Legacy window_size takes precedence if specified
    hop_length = (
        options.hop_size or options.hop_length or n_fft // 4
    )  # Legacy hop_size takes precedence
    sample_rate = float(audio.sample_rate)

    for ch in range(num_channels):
        ax = axes[ch, 0]

        # Get channel data as float
        channel_data = _to_float_minus1_1(audio.extract_channel(ch).numpy())

        # Compute STFT using matplotlib's built-in specgram
        _, _, _, im = ax.specgram(
            channel_data,
            NFFT=n_fft,
            Fs=sample_rate,
            noverlap=n_fft - hop_length,
            window=options.window,
            cmap=options.colormap,
            scale="dB" if options.log_scale else "linear",
        )

        # Apply dB range if using log scale
        if options.log_scale:
            im.set_clim(options.db_range)

        # Set frequency scale
        if options.freq_scale == "log":
            ax.set_yscale("log")
            ax.set_ylabel(
                f"Channel {ch + 1} - {options.ylabel} (log)"
                if num_channels > 1
                else f"{options.ylabel} (log)"
            )
        else:
            ax.set_ylabel(
                f"Channel {ch + 1} - {options.ylabel}"
                if num_channels > 1
                else options.ylabel
            )

        # Grid
        if options.grid:
            ax.grid(True, alpha=0.3)

        # Colorbar for the last subplot or single channel
        if options.colorbar and ch == num_channels - 1:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(options.colorbar_label, fontsize=options.label_fontsize)

    # Set x-axis label only on bottom subplot
    axes[-1, 0].set_xlabel(options.xlabel, fontsize=options.label_fontsize)

    # Title
    if num_channels > 1:
        fig.suptitle(options.title, fontsize=options.title_fontsize)
    else:
        axes[0, 0].set_title(options.title, fontsize=options.title_fontsize)

    plt.tight_layout()
    if options.save_path:
        plt.savefig(options.save_path, dpi=300)
    plt.show()
    plt.close()


def specshow(
    data: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    x_axis: str = "time",
    y_axis: str = "hz",
    cmap: str = "viridis",
    title: str = "Spectrogram",
    figsize: tuple[int, int] = (12, 8),
    colorbar: bool = True,
    colorbar_label: str = "Magnitude (dB)",
    db_range: tuple[float, float] | None = None,
    save_path: str | None = None,
) -> None:
    """
    Display a spectrogram or other 2D data with appropriate axis labeling.

    Similar to librosa.display.specshow() for displaying spectrograms with
    proper time and frequency axis labels.

    # Arguments
    * `data` - 2D NumPy array to display (frequency bins × time frames)
    * `sr` - Sample rate in Hz (for axis labeling)
    * `hop_length` - Hop length used in STFT (for time axis)
    * `x_axis` - X-axis type ("time", "frames", "s", "ms")
    * `y_axis` - Y-axis type ("hz", "log", "mel", "linear")
    * `cmap` - Colormap name
    * `title` - Plot title
    * `figsize` - Figure size tuple
    * `colorbar` - Whether to show colorbar
    * `colorbar_label` - Colorbar label
    * `db_range` - dB range for color scaling (optional)
    * `save_path` - Path to save figure (optional)

    # Examples
    ```python
    import numpy as np
    import audio_samples as aus
    import audio_samples.display as aus_display

    # Display STFT result
    D = audio.stft(n_fft=2048, hop_length=512)
    S_db = aus.power_to_db(np.abs(D)**2, ref=np.max(np.abs(D)**2))
    aus_display.specshow(S_db, sr=audio.sample_rate, hop_length=512,
                        y_axis="log", x_axis="time")

    # Display mel spectrogram
    mel_spec = audio.mel_spectrogram(n_mels=128, hop_length=512)
    mel_spec_db = aus.power_to_db(mel_spec, ref=np.max(mel_spec))
    aus_display.specshow(mel_spec_db, sr=audio.sample_rate, hop_length=512,
                        y_axis="mel", x_axis="time", title="Mel Spectrogram")
    ```
    """
    plt.figure(figsize=figsize)

    # Generate axis coordinates
    n_frames = data.shape[1] if data.ndim == 2 else len(data)
    n_freqs = data.shape[0] if data.ndim == 2 else 1

    # Time axis
    if x_axis in ["time", "s"]:
        time_coords = np.arange(n_frames) * hop_length / sr
        xlabel = "Time (s)"
    elif x_axis == "ms":
        time_coords = np.arange(n_frames) * hop_length / sr * 1000
        xlabel = "Time (ms)"
    else:  # frames
        time_coords = np.arange(n_frames)
        xlabel = "Frames"

    # Frequency axis
    if y_axis in ["hz", "linear"]:
        freq_coords = np.linspace(0, sr / 2, n_freqs)
        ylabel = "Frequency (Hz)"
        yscale = "linear"
    elif y_axis == "log":
        freq_coords = np.linspace(0, sr / 2, n_freqs)
        ylabel = "Frequency (Hz)"
        yscale = "log"
    elif y_axis == "mel":
        # Mel scale approximation
        mel_coords = np.linspace(0, 2595 * np.log10(1 + sr / 2 / 700), n_freqs)
        freq_coords = 700 * (10 ** (mel_coords / 2595) - 1)
        ylabel = "Mel Frequency"
        yscale = "linear"
    else:  # bins
        freq_coords = np.arange(n_freqs)
        ylabel = "Frequency Bin"
        yscale = "linear"

    # Create the plot - only 2D spectrograms supported
    if data.ndim == 2:
        extent = (time_coords[0], time_coords[-1], freq_coords[0], freq_coords[-1])
        im = plt.imshow(data, aspect="auto", origin="lower", cmap=cmap, extent=extent)

        # Apply dB range if specified
        if db_range is not None:
            im.set_clim(db_range)

        # Add colorbar
        if colorbar:
            cbar = plt.colorbar(im)
            cbar.set_label(colorbar_label)

    else:
        raise ValueError("specshow only supports 2D arrays (spectrograms)")

    # Set axis properties
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if yscale == "log":
        plt.yscale("log")

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()
    plt.close()


def compare_waveforms(
    audio_samples: list[AudioSamples],
    labels: list[str] | None = None,
    options: ComparisonPlotOptions = ComparisonPlotOptions(),
) -> None:
    """Compare multiple audio waveforms in the same plot."""
    if not audio_samples:
        raise ValueError("At least one audio sample is required")

    if labels is None:
        labels = [f"Audio {i + 1}" for i in range(len(audio_samples))]

    if len(labels) != len(audio_samples):
        raise ValueError("Number of labels must match number of audio samples")

    rc = {
        "figure.figsize": options.figsize,
        "axes.titlesize": options.title_fontsize,
        "axes.labelsize": options.label_fontsize,
        "xtick.labelsize": options.tick_fontsize,
        "ytick.labelsize": options.tick_fontsize,
        **_font_rc(options),
    }

    if options.use_seaborn_style:
        sns.set_theme(
            style=options.seaborn_style, context=options.seaborn_context, rc=rc
        )
    else:
        plt.rcParams.update(rc)

    # Use the first audio sample to determine channels
    num_channels = audio_samples[0].num_channels

    fig, axes = plt.subplots(
        num_channels,
        1,
        figsize=options.figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    # Find global ylim based on all samples
    if options.ylim is None:
        all_max = 0.0
        for audio in audio_samples:
            for ch in range(min(num_channels, audio.num_channels)):
                channel_data = _to_float_minus1_1(audio.extract_channel(ch).numpy())
                all_max = max(all_max, float(np.max(np.abs(channel_data))))
        ylim = (-1.05 * all_max, 1.05 * all_max)
    else:
        ylim = options.ylim

    for ch in range(num_channels):
        ax = axes[ch, 0]

        for i, audio in enumerate(audio_samples):
            if ch >= audio.num_channels:
                continue  # Skip if this audio doesn't have this channel

            channel_data = _to_float_minus1_1(audio.extract_channel(ch).numpy())
            color = options.colors[i % len(options.colors)]

            # Create time axis based on this audio's sample rate
            duration_samples = len(channel_data)
            sample_rate = float(audio.sample_rate)
            time_axis = np.arange(duration_samples) / sample_rate

            ax.plot(
                time_axis,
                channel_data,
                color=color,
                linewidth=options.wave_linewidth,
                alpha=options.wave_alpha,
                label=labels[i] if ch == 0 else None,  # Only label on first channel
            )

        ax.set_ylim(*ylim)

        if options.grid:
            ax.grid(
                True,
                linestyle=options.grid_linestyle,
                linewidth=options.grid_linewidth,
                alpha=options.grid_alpha,
            )

        ax.set_ylabel(
            f"Channel {ch + 1}" if num_channels > 1 else options.ylabel,
            fontsize=options.label_fontsize,
        )

    # X-axis label and legend on bottom subplot
    axes[-1, 0].set_xlabel(options.xlabel, fontsize=options.label_fontsize)

    if options.legend:
        axes[0, 0].legend(loc=options.legend_loc, fontsize=options.tick_fontsize)

    # Title
    if num_channels > 1:
        fig.suptitle(options.title, fontsize=options.title_fontsize)
    else:
        axes[0, 0].set_title(options.title, fontsize=options.title_fontsize)

    plt.tight_layout()
    if options.save_path:
        plt.savefig(options.save_path, dpi=300)
    plt.show()
    plt.close()


def plot_difference(
    audio1: AudioSamples,
    audio2: AudioSamples,
    labels: tuple[str, str] = ("Original", "Modified"),
    options: WaveformPlotOptions = WaveformPlotOptions(),
) -> None:
    """Plot the difference between two audio samples."""
    if audio1.num_channels != audio2.num_channels:
        raise ValueError("Both audio samples must have the same number of channels")

    if audio1.sample_rate != audio2.sample_rate:
        raise ValueError("Both audio samples must have the same sample rate")

    # Ensure both have the same length by truncating to shorter
    min_length = min(
        len(audio1.extract_channel(0).numpy()), len(audio2.extract_channel(0).numpy())
    )

    rc = {
        "figure.figsize": options.figsize,
        "axes.titlesize": options.title_fontsize,
        "axes.labelsize": options.label_fontsize,
        "xtick.labelsize": options.tick_fontsize,
        "ytick.labelsize": options.tick_fontsize,
        **_font_rc(options),
    }

    if options.use_seaborn_style:
        sns.set_theme(
            style=options.seaborn_style, context=options.seaborn_context, rc=rc
        )
    else:
        plt.rcParams.update(rc)

    num_channels = audio1.num_channels
    fig, axes = plt.subplots(
        num_channels,
        1,
        figsize=options.figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    sample_rate = float(audio1.sample_rate)
    time_axis = np.arange(min_length) / sample_rate

    for ch in range(num_channels):
        ax = axes[ch, 0]

        # Get channel data and compute difference
        data1 = _to_float_minus1_1(audio1.extract_channel(ch).numpy()[:min_length])
        data2 = _to_float_minus1_1(audio2.extract_channel(ch).numpy()[:min_length])
        diff = data1 - data2

        ax.plot(
            time_axis,
            diff,
            color=options.wave_color,
            linewidth=options.wave_linewidth,
            alpha=options.wave_alpha,
        )

        if options.grid:
            ax.grid(
                True,
                linestyle=options.grid_linestyle,
                linewidth=options.grid_linewidth,
                alpha=options.grid_alpha,
            )

        ax.set_ylabel(
            f"Channel {ch + 1} - Difference" if num_channels > 1 else "Difference",
            fontsize=options.label_fontsize,
        )

    # X-axis label
    axes[-1, 0].set_xlabel(options.xlabel, fontsize=options.label_fontsize)

    # Title
    title = f"Difference: {labels[0]} - {labels[1]}"
    if num_channels > 1:
        fig.suptitle(title, fontsize=options.title_fontsize)
    else:
        axes[0, 0].set_title(title, fontsize=options.title_fontsize)

    plt.tight_layout()
    if options.save_path:
        plt.savefig(options.save_path, dpi=300)
    plt.show()
    plt.close()
