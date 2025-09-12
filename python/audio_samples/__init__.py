from .audio_samples import *
from ._helpers import load, save
from .display import (
    waveform, WaveformPlotOptions, 
    spectrogram, SpectrogramPlotOptions,
    compare_waveforms, ComparisonPlotOptions,
    plot_difference
)

# Jupyter integration - import only if IPython is available
try:
    from . import jupyter
except ImportError:
    # IPython not available, skip jupyter integration
    pass
