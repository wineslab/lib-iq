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

import math
from typing import Literal, Sequence

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

IQSample = Sequence[tuple[float, float]]
DataFormat = Literal["real-imag", "magnitude-phase"]

def to_superscript(num: int) -> str:
    superscript_map = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return str(num).translate(superscript_map)

def get_scale_suffix(value: float) -> tuple[float, str]:
    """
    Return scale and suffix for axis labels.
    """
    if value == 0:
        return 1.0, "×10⁰"
    exponent = int(math.floor(math.log10(abs(value))))
    scale = 10 ** (exponent // 3 * 3)
    superscript = to_superscript(int(math.log10(scale)))
    suffix = f"×10{superscript}" if scale != 1 else ""
    return scale, suffix or "×10⁰"


def process_data(
    iq_sample: IQSample, data_format: DataFormat
) -> tuple[list[float], list[float]]:
    """
    Process I/Q data into either real-imaginary or magnitude-phase format.

    Args:
        iq_sample (Sequence[tuple[float, float]]): List of I/Q sample pairs.
        data_format (Literal["real-imag", "magnitude-phase"]): Format to convert the I/Q data into.

    Returns:
        tuple[list[float], list[float]]: Two lists representing I and Q or phase and magnitude.
    """
    real = [x[0] for x in iq_sample]
    imag = [x[1] for x in iq_sample]
    if data_format == "real-imag":
        return real, imag
    elif data_format == "magnitude-phase":
        magnitude = []
        phase = []
        for i in range(len(real)):
            magnitude.append(math.sqrt(real[i] ** 2 + imag[i] ** 2))
            phase.append(math.atan2(imag[i], real[i]))
        return phase, magnitude


def scatterplot(iq: IQSample, data_format: DataFormat, grids: bool = False, interactive_plots: bool = False,
    path: str = "",) -> None:
    """
    Plot a static scatterplot of I/Q samples in real/imag or magnitude/phase format.

    Args:
        iq (Sequence[tuple[float, float]]): List of I/Q sample pairs.
        data_format (Literal["real-imag", "magnitude-phase"]): Format for plotting.
        grids (bool): Whether to show gridlines.

    Returns:
        None
    """
    a_data, b_data = process_data(iq, data_format)

    background_color = "white"
    text_color = "black"

    fig, ax = plt.subplots(dpi=300)
    fig.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.tick_params(axis="both", colors=text_color, pad=5)
    labels_font_size = 10
    title_font_size = 12
    tick_font_size = 8

    if data_format == "real-imag":
        max_val = max(max(abs(x) for x in a_data), max(abs(y) for y in b_data))
        scale, suffix = get_scale_suffix(max_val)

        a_scaled = np.array(a_data) / scale
        b_scaled = np.array(b_data) / scale

        ax.scatter(a_scaled, b_scaled, s=0.1, c=text_color)
        ax.tick_params(axis="both", labelsize=tick_font_size, colors=text_color, pad=5)

        x_min = a_scaled.min() * 1.1 - 2
        x_max = a_scaled.max() * 1.1 + 2
        y_min = b_scaled.min() * 1.1 - 2
        y_max = b_scaled.max() * 1.1 + 2

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        def round_to_pretty_step(val):
            for step in [1, 2, 5, 10, 20, 50, 100]:
                if val <= step:
                    return step
            return val

        max_x_ticks = 9
        max_y_ticks = 7
        x_range = x_max - x_min
        y_range = y_max - y_min

        x_tick_step = round_to_pretty_step(round(x_range / (max_x_ticks - 1)))
        y_tick_step = round_to_pretty_step(round(y_range / (max_y_ticks - 1)))

        x_ticks = np.arange(
            math.floor(x_min / x_tick_step) * x_tick_step,
            math.ceil(x_max / x_tick_step) * x_tick_step + x_tick_step,
            x_tick_step,
        )
        y_ticks = np.arange(
            math.floor(y_min / y_tick_step) * y_tick_step,
            math.ceil(y_max / y_tick_step) * y_tick_step + y_tick_step,
            y_tick_step,
        )

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        if grids:
            ax.grid(True, color="gray", linestyle="--", linewidth=0.5)

        ax.set_title("I/Q Scatter Plot (Real-Imag)", color=text_color, y=1.07, fontsize=title_font_size)
        ax.set_xlabel(f"I [{suffix}]", color=text_color, fontsize=labels_font_size)
        ax.set_ylabel(f"Q [{suffix}]", color=text_color, fontsize=labels_font_size)

    elif data_format == "magnitude-phase":
        max_val = max(abs(m) for m in b_data)
        scale, suffix = get_scale_suffix(max_val)
        b_scaled = np.array(b_data) / scale
        a_array = np.array(a_data)

        ax.scatter(a_array, b_scaled, s=0.1, c=text_color)
        ax.tick_params(axis="both", labelsize=tick_font_size, colors=text_color, pad=5)

        ax.set_xlim(-math.pi, math.pi)
        y_min = 0
        y_max = b_scaled.max() * 1.1
        ax.set_ylim(y_min, y_max)

        x_ticks = np.arange(-4, 5, 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{int(round(x))}" for x in x_ticks], color=text_color)

        y_tick_step = 10
        y_ticks = np.arange(
            math.floor(y_min / y_tick_step) * y_tick_step,
            math.ceil(y_max / y_tick_step) * y_tick_step + y_tick_step,
            y_tick_step,
        )
        ax.set_yticks(y_ticks)

        if grids:
            ax.grid(True, color="gray", linestyle="--", linewidth=0.5)

        ax.set_title("I/Q Scatter Plot (Magnitude-Phase)", color=text_color, y=1.07, fontsize=title_font_size)
        ax.set_xlabel("Phase [rad]", color=text_color, fontsize=labels_font_size)
        ax.set_ylabel(f"Magnitude [{suffix}]", color=text_color, fontsize=labels_font_size)

    plt.tight_layout()
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
