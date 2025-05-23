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


def calculate_window_duration_ms(
    num_samples: int, num_windows: int, sample_rate: float
) -> float:
    """
    Calculate the duration of each time window in milliseconds.

    Args:
        num_samples (int): Total number of samples.
        num_windows (int): Number of windows.
        sample_rate (float): Sampling rate in Hz.

    Returns:
        float: Window duration in milliseconds.
    """
    total_duration_s = num_samples / sample_rate
    window_duration_s = total_duration_s / num_windows
    window_duration_ms = window_duration_s * 1000
    return window_duration_ms


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
        return 1e9, "(GHz)"
    elif freq >= 1e6:
        return 1e6, "(MHz)"
    elif freq >= 1e3:
        return 1e3, "(kHz)"
    else:
        return 1, "(Hz)"


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
        return 1e9, "(10⁹)"
    elif t >= 1e6:
        return 1e6, "(10⁶)"
    elif t >= 1e3:
        return 1e3, "(10³)"
    else:
        return 1, ""


def update_y_labels(
    ax: plt.Axes, num_freqs: int, sample_rate: float, center_frequency: float
) -> None:
    """
    Update Y-axis labels of a spectrogram to display frequency values.

    Args:
        ax (plt.Axes): The matplotlib Axes object.
        num_freqs (int): Number of frequency bins.
        sample_rate (float): Sampling rate in Hz.
        center_frequency (float): Center frequency in Hz.

    Returns:
        None
    """
    y_lims = ax.get_ylim()
    freqs = np.linspace(-sample_rate / 2, sample_rate / 2, num_freqs) + center_frequency
    y_lims = np.clip(y_lims, 0, num_freqs - 1)
    visible_freqs = freqs[int(y_lims[0]) : int(y_lims[1])]
    if len(visible_freqs) == 0:
        return
    freq_scale, freq_unit = get_frequency_scale(visible_freqs)
    num_ticks = 11
    if (y_lims[1] - y_lims[0]) > 0:
        y_ticks = np.linspace(int(y_lims[0]), int(y_lims[1]), num_ticks)
    else:
        y_ticks = np.linspace(0, num_freqs - 1, min(num_freqs, num_ticks))
    valid_ticks = [int(y) for y in y_ticks if 0 <= int(y) < num_freqs]
    y_labels = [f"{freqs[idx] / freq_scale:.2f}" for idx in valid_ticks]
    ax.set_yticks(valid_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel(f"Frequency {freq_unit}")
    plt.draw()


def update_x_labels(ax: plt.Axes, num_windows: int, window_duration_ms: float) -> None:
    """
    Update X-axis labels of a spectrogram to display time window numbers.

    Args:
        ax (plt.Axes): The matplotlib Axes object.
        num_windows (int): Number of windows.
        window_duration_ms (float): Duration of each window in milliseconds.

    Returns:
        None
    """
    x_lims = ax.get_xlim()
    x_lims = np.clip(x_lims, 0, num_windows - 1)
    num_ticks = 7
    if (x_lims[1] - x_lims[0]) > 0:
        x_ticks = np.linspace(int(x_lims[0]), int(x_lims[1]), num_ticks)
    else:
        x_ticks = np.linspace(0, num_windows - 1, min(num_windows, num_ticks))
    x_labels = np.round(x_ticks).astype(int)

    time_scale, time_unit = get_window_size_scale(x_labels)
    x_labels = x_labels / time_scale
    x_labels = [f"{x:.2f}" for x in x_labels]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(f"Window number {time_unit}")
    plt.draw()


def on_zoom(
    event,
    ax: plt.Axes,
    num_freqs: int,
    sample_rate: float,
    center_frequency: float,
    num_windows: int,
    window_duration_ms: float,
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
        window_duration_ms (float): Duration of each window in ms.

    Returns:
        None
    """
    update_y_labels(ax, num_freqs, sample_rate, center_frequency)
    update_x_labels(ax, num_windows, window_duration_ms)


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

    freqs = np.linspace(-sample_rate / 2, sample_rate / 2, num_freqs) + center_frequency
    y_ticks = np.linspace(0, num_freqs - 1, min(num_freqs, 11))
    freq_scale, freq_unit = get_frequency_scale(freqs)
    y_labels = [f"{freqs[int(y)] / freq_scale:.2f}" for y in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel(f"Frequency {freq_unit}")

    window_duration_ms = calculate_window_duration_ms(
        num_freqs * num_windows, num_windows, sample_rate
    )
    x_ticks = np.linspace(0, num_windows - 1, min(num_windows, 7))
    x_labels = np.round(x_ticks).astype(int)
    time_scale, time_unit = get_window_size_scale(x_labels)
    x_labels = x_labels / time_scale
    x_labels = [f"{x:.2f}" for x in x_labels]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(f"Window number {time_unit}")

    fig.colorbar(img, label="Power (dB)")
    fig.subplots_adjust(left=0.2)

    plt.text(
        1,
        1.05,
        f"Each window is {window_duration_ms:.2f} ms",
        transform=ax.transAxes,
        ha="right",
    )

    ax.callbacks.connect(
        "xlim_changed",
        lambda evt: on_zoom(
            evt,
            ax,
            num_freqs,
            sample_rate,
            center_frequency,
            num_windows,
            window_duration_ms,
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
            window_duration_ms,
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
