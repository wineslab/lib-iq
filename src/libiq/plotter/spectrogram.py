import matplotlib

from libiq.utils.logger import logger

try:
    import tkinter
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.close()
except Exception as e:
    logger.warning(f"TkAgg not available or usable: {e}. Falling back to Agg.")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from typing import Sequence

import numpy as np

try:
    import scienceplots
    plt.style.use(["science", "no-latex"])
except (ImportError, OSError) as e:
    logger.warning(f"Matplotlib style 'science' not found. Using default style. ({e})")
    plt.style.use("default")
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
rcParams["font.size"] = 14
rcParams["legend.fontsize"] = "medium"
rcParams["axes.grid"] = False

TICK_LABEL_SIZE = 8
AXIS_LABEL_SIZE = 10
TITLE_FONT_SIZE = 12

def get_frequency_scale(freqs: Sequence[float]) -> tuple[float, str]:
    """
    Determine the appropriate frequency scale and unit based on the frequency values.

    Args:
        freqs (Sequence[float]): List or array of frequencies.

    Returns:
        tuple[float, str]: A scaling factor and its unit as a string.
    """
    freq = freqs[0]
    if freq >= 1e9:
        return 1e9, "[GHz]"
    elif freq >= 1e6:
        return 1e6, "[MHz]"
    elif freq >= 1e3:
        return 1e3, "[kHz]"
    else:
        return 1, "[Hz]"


def get_window_size_scale(times: Sequence[float]) -> tuple[float, str]:
    """
    Determine the appropriate scale and unit for time windows.

    Args:
        times (Sequence[float]): Sequence of time values or window indices.

    Returns:
        tuple[float, str]: A scaling factor and its unit as a string.
    """
    t = times[-1] if len(times) > 0 else 0
    if t >= 1e9:
        return 1e9, "[x10⁹]"
    elif t >= 1e6:
        return 1e6, "[x10⁶]"
    elif t >= 1e3:
        return 1e3, "[x10³]"
    else:
        return 1, ""


def update_x_labels(ax: plt.Axes, num_windows: int) -> None:
    """
    Update X-axis labels of a spectrogram to display time window numbers
    spanning the entire range [0 … num_windows–1], not just the zoomed region.
    """
    num_ticks = 7

    x_ticks = np.linspace(0, num_windows - 1, num_ticks)

    x_indices = np.round(x_ticks).astype(int)

    time_scale, time_unit = get_window_size_scale(x_indices)
    x_labels = [f"{(idx / time_scale):.2f}" for idx in x_indices]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel(f"Window number {time_unit}", fontsize=AXIS_LABEL_SIZE)


def update_y_labels(
    ax: plt.Axes, num_freqs: int, sample_rate: float, center_frequency: float
) -> None:
    """
    Update Y-axis labels to show full frequency range (not just zoomed region).
    """
    num_y_ticks = 7
    freqs = np.linspace(-sample_rate / 2, sample_rate / 2, num_freqs) + center_frequency
    freq_scale, freq_unit = get_frequency_scale(freqs)

    y_ticks = np.linspace(0, num_freqs - 1, num_y_ticks).astype(int)
    y_labels = [f"{freqs[i] / freq_scale:.2f}" for i in y_ticks]

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=TICK_LABEL_SIZE)
    ax.set_ylabel(f"Frequency {freq_unit}", fontsize=AXIS_LABEL_SIZE)


def on_zoom(
    event,
    ax: plt.Axes,
    num_freqs: int,
    sample_rate: float,
    center_frequency: float,
    num_windows: int,
) -> None:
    """
    Callback for zooming on the spectrogram. Updates axis labels accordingly.

    Args:
        event: Matplotlib zoom event (unused).
        ax (plt.Axes): The matplotlib Axes object.
        num_freqs (int): Number of frequency bins.
        sample_rate (float): Sampling rate in Hz.
        center_frequency (float): Center frequency in Hz.
        num_windows (int): Number of time windows.

    Returns:
        None
    """
    update_y_labels(ax, num_freqs, sample_rate, center_frequency)
    update_x_labels(ax, num_windows)


def spectrogram(
    spectrogram_data: Sequence[Sequence[float]],
    sample_rate: float,
    center_frequency: float,
    interactive_plots: bool = False,
    path: str = "",
) -> None:
    """
    Display a spectrogram image with proper axis labeling for time and frequency.

    Args:
        spectrogram_data (Sequence[Sequence[float]]): 2D array or list of dB values [freq x time].
        sample_rate (float): Sampling rate in Hz.
        center_frequency (float): Center frequency in Hz.

    Returns:
        None
    """
    spectrogram_data = np.array(spectrogram_data).T
    logger.debug(
        f"There are {spectrogram_data.shape[1]} windows of size {spectrogram_data.shape[0]}"
    )

    fig, ax = plt.subplots(dpi=300)

    max_power_db = np.max(spectrogram_data)
    min_power_db = np.min(spectrogram_data)
    img = ax.imshow(
        spectrogram_data,
        aspect="auto",
        cmap="jet",
        origin="lower",
        vmin=min_power_db,
        vmax=max_power_db,
    )

    num_windows = spectrogram_data.shape[1]
    num_freqs = spectrogram_data.shape[0]

    update_y_labels(ax, num_freqs, sample_rate, center_frequency)

    x_ticks = np.linspace(0, num_windows - 1, min(num_windows, 7))
    x_labels = np.round(x_ticks).astype(int)
    time_scale, time_unit = get_window_size_scale(x_labels)
    x_labels = x_labels / time_scale
    x_labels = [f"{x:.2f}" for x in x_labels]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel(f"Window number {time_unit}", fontsize=AXIS_LABEL_SIZE)

    cbar = fig.colorbar(img, shrink=0.8, aspect=20)
    cbar.set_label("Power [dB]", fontsize=AXIS_LABEL_SIZE)
    cbar.locator = MaxNLocator(nbins=9)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)

    ax.callbacks.connect(
        "xlim_changed",
        lambda evt: on_zoom(
            evt,
            ax,
            num_freqs,
            sample_rate,
            center_frequency,
            num_windows,
        ),
    )
    ax.callbacks.connect(
        "ylim_changed",
        lambda evt: on_zoom(
            evt,
            ax,
            num_freqs,
            sample_rate,
            center_frequency,
            num_windows,
        ),
    )

    if interactive_plots:
        plt.show()
    else:
        if path != "":
            plt.savefig(path, format="pdf")
            plt.close()
        else:
            raise ValueError(
                "The path to save the plot is empty. Provide a valid path or set INTERACTIVE_PLOTS to 'interactive'."
            )
