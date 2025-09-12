"""Jupyter/IPython integration for audio_samples.

Provides convenient audio playback and visualization in Jupyter notebooks.
"""

from typing import Optional
import tempfile
import os

from audio_samples import AudioSamples

try:
    from IPython.display import Audio, display
    from IPython import get_ipython

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


def play_audio(
    audio: AudioSamples,
    autoplay: bool = False,
    normalize: bool = True,
) -> None:
    """Play audio in Jupyter notebook using IPython.display.Audio.

    Args:
        audio: AudioSamples to play
        autoplay: Whether to start playing automatically
        normalize: Whether to normalize audio to prevent clipping

    Raises:
        ImportError: If IPython is not available
    """
    if not IPYTHON_AVAILABLE:
        raise ImportError("IPython is required for Jupyter audio playback")

    # Convert to numpy array with float32 for web compatibility
    audio_data = audio.numpy(dtype="float32", copy=True)

    # Normalize if requested
    if normalize:
        max_val = abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val

    # Create Audio widget
    audio_widget = Audio(
        data=audio_data.T,  # IPython expects channels as rows
        rate=audio.sample_rate,
        autoplay=autoplay,
        normalize=False,  # We handle normalization ourselves
    )

    display(audio_widget)


def save_and_play(
    audio: AudioSamples,
    filename: Optional[str] = None,
    format: str = "wav",
    autoplay: bool = False,
    keep_file: bool = False,
) -> str:
    """Save audio to temporary file and play in Jupyter.

    Args:
        audio: AudioSamples to save and play
        filename: Optional filename (will create temp file if None)
        format: Audio format (wav, mp3, etc.)
        autoplay: Whether to start playing automatically
        keep_file: Whether to keep the temporary file

    Returns:
        Path to the created audio file

    Raises:
        ImportError: If IPython is not available
    """
    if not IPYTHON_AVAILABLE:
        raise ImportError("IPython is required for Jupyter audio playback")

    # Create temporary file if no filename provided
    if filename is None:
        temp_fd, temp_path = tempfile.mkstemp(suffix=f".{format}")
        os.close(temp_fd)  # Close the file descriptor, we just need the path
        filepath = temp_path
    else:
        filepath = filename

    # Save audio using the helper function
    from ._helpers import save

    save(filepath, audio)

    # Play using file path
    audio_widget = Audio(filename=filepath, autoplay=autoplay)

    display(audio_widget)

    # Clean up temporary file if requested
    if not keep_file and filename is None:
        try:
            os.unlink(filepath)
        except OSError:
            pass  # File might already be deleted

    return filepath


def quick_plot(
    audio: AudioSamples, plot_type: str = "waveform", **plot_options
) -> None:
    """Quick plotting function for Jupyter notebooks.

    Args:
        audio: AudioSamples to plot
        plot_type: Type of plot ("waveform", "spectrogram", or "both")
        **plot_options: Additional options passed to plotting functions
    """
    from .display import (
        waveform,
        spectrogram,
        WaveformPlotOptions,
        SpectrogramPlotOptions,
    )

    if plot_type == "waveform":
        options = WaveformPlotOptions(**plot_options)
        waveform(audio, options)
    elif plot_type == "spectrogram":
        options = SpectrogramPlotOptions(**plot_options)
        spectrogram(audio, options)
    elif plot_type == "both":
        import matplotlib.pyplot as plt

        # Create subplot layout
        figsize = plot_options.get("figsize", (14, 10))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot waveform on top
        waveform_opts = WaveformPlotOptions(
            figsize=(figsize[0], figsize[1] // 2), **plot_options
        )
        # Temporarily override save_path and display options for subplot
        waveform_opts.save_path = None

        # Plot spectrogram on bottom
        spec_opts = SpectrogramPlotOptions(
            figsize=(figsize[0], figsize[1] // 2), **plot_options
        )
        spec_opts.save_path = None

        # Note: This is a simplified version. For full subplot control,
        # users should use the individual plotting functions directly.
        print(
            "For 'both' plots, use waveform() and spectrogram() separately for better control."
        )
        waveform(audio, waveform_opts)
        spectrogram(audio, spec_opts)
    else:
        raise ValueError(
            f"Unknown plot_type: {plot_type}. Use 'waveform', 'spectrogram', or 'both'."
        )


def audio_info(audio: AudioSamples) -> None:
    """Display formatted audio information in Jupyter.

    Args:
        audio: AudioSamples to analyze
    """
    if IPYTHON_AVAILABLE:
        from IPython.display import display, HTML

        # Create formatted HTML table
        html = f"""
        <div style="
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            border: 1px solid #ccc;
            padding: 16px 20px;
            border-radius: 8px;
            background-color: #fefefe;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            max-width: 480px;
            color: #333;
        ">
            <h3 style="
                margin: 0 0 16px;
                font-size: 16px;
                font-weight: 600;
                color: #555;
                border-bottom: 1px solid #eee;
                padding-bottom: 4px;
                letter-spacing: 0.5px;
            ">
                🎧 Audio Information
            </h3>
            <table style="
                border-collapse: collapse;
                width: 100%;
                font-size: 14px;
            ">
                <tbody>
                    <tr style="background-color: #fafafa;">
                        <td style='padding: 6px 0; font-weight: 600;'>Sample Rate:</td>
                        <td style='text-align: right;'>{audio.sample_rate:,} Hz</td>
                    </tr>
                    <tr>
                        <td style='padding: 6px 0; font-weight: 600;'>Channels:</td>
                        <td style='text-align: right;'>{audio.num_channels}</td>
                    </tr>
                    <tr style="background-color: #fafafa;">
                        <td style='padding: 6px 0; font-weight: 600;'>Duration:</td>
                        <td style='text-align: right;'>{audio.duration:.3f} seconds</td>
                    </tr>
                    <tr>
                        <td style='padding: 6px 0; font-weight: 600;'>Samples per Channel:</td>
                        <td style='text-align: right;'>{len(audio.extract_channel(0).numpy()):,}</td>
                    </tr>
                    <tr style="background-color: #fafafa;">
                        <td style='padding: 6px 0; font-weight: 600;'>Data Type:</td>
                        <td style='text-align: right;'>{audio.dtype}</td>
                    </tr>
                    <tr>
                        <td style='padding: 6px 0; font-weight: 600;'>Total Samples:</td>
                        <td style='text-align: right;'>{len(audio.extract_channel(0).numpy()) * audio.num_channels:,}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """

        display(HTML(html))
    else:
        # Fallback to plain text
        print("Audio Information:")
        print(f"  Sample Rate: {audio.sample_rate} Hz")
        print(f"  Channels: {audio.num_channels}")
        print(f"  Duration: {audio.duration:.3f} seconds")
        print(f"  Samples per Channel: {len(audio.extract_channel(0).numpy()):,}")
        print(f"  Data Type: {audio.dtype}")
        print(
            f"  Total Samples: {len(audio.extract_channel(0).numpy()) * audio.num_channels:,}"
        )


def is_notebook() -> bool:
    """Check if running in a Jupyter notebook.

    Returns:
        True if running in Jupyter notebook, False otherwise
    """
    if not IPYTHON_AVAILABLE:
        return False

    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except Exception:
        return False  # Probably standard Python interpreter


# Convenience functions for common workflows
def load_and_play(filepath: str, **load_options) -> AudioSamples:
    """Load audio file and immediately play it in Jupyter.

    Args:
        filepath: Path to audio file
        **load_options: Options passed to load() function

    Returns:
        Loaded AudioSamples
    """
    from ._helpers import load

    audio = load(filepath, **load_options)
    play_audio(audio)
    return audio


def load_plot_play(
    filepath: str, plot_type: str = "waveform", **options
) -> AudioSamples:
    """Load audio, plot it, and play it - all in one go.

    Args:
        filepath: Path to audio file
        plot_type: Type of plot ("waveform", "spectrogram", or "both")
        **options: Options for loading and plotting

    Returns:
        Loaded AudioSamples
    """
    from ._helpers import load

    # Separate load and plot options
    load_opts = {
        k: v
        for k, v in options.items()
        if k
        in [
            "dtype",
            "start_time",
            "stop_time",
            "duration",
            "start_frame",
            "stop_frame",
            "frames",
        ]
    }
    plot_opts = {k: v for k, v in options.items() if k not in load_opts}

    audio = load(filepath, **load_opts)
    audio_info(audio)
    quick_plot(audio, plot_type, **plot_opts)
    play_audio(audio)

    return audio
