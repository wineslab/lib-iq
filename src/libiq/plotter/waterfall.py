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

import os
from typing import Union

import numpy as np
import pandas as pd

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

def plot_waterfall(
    data_input: Union[str, np.ndarray],
    interactive_plots: str = "interactive",
    fft_size: int = 1536,
    path: str = "",
    signed_data: bool = True,
) -> None:
    """
    Plots the waterfall of a precomputed FFT signal derived from IQ samples.

    Parameters:
        data_input: One of the following:
            - A string representing a path to a binary file with 16-bit data arranged
              as pairs (Real, Imag) for each FFT sample.
            - A string representing a path to a CSV file containing at least "Real"
              and "Imaginary" columns. The CSV may also contain "Phase" and "Magnitude,"
              but they are not required for this plot.
            - A NumPy array containing the FFT data directly. If the array is 1D, its
              length must be a multiple of fft_size (the FFT window width). If it is 2D, it
              must have fft_size columns.
        interactive_plots: If set to '' or 'interactive', the plot is displayed interactively.
                    Otherwise, it is assumed that this parameter is the file path
                    where the plot will be saved.
        signed_data: If True, the binary file is read as int16; if False, as uint16.
                     This applies only if a binary file is provided and matters if
                     the IQ data can be negative (often the case for raw IQ data).

    Notes:
        - Each FFT window is composed of fft_size complex samples. Each complex sample
          is stored as two 16-bit values: (Real, Imag).
        - Therefore, one FFT window = fft_size * 2 16-bit values.
        - The waterfall is plotted from the first window (top) to the last window (bottom).
        - When reading from CSV, only the "Real" and "Imaginary" columns are used to
          reconstruct the complex samples. The rest of the columns are ignored.
    """
    plt.close("all")

    if isinstance(data_input, str):
        if not os.path.exists(data_input):
            raise FileNotFoundError(f"File not found: {data_input}")

        if data_input.lower().endswith(".csv"):
            df = pd.read_csv(data_input)
            if "Real" not in df.columns or "Imaginary" not in df.columns:
                raise ValueError(
                    "CSV file must contain at least 'Real' and 'Imaginary' columns."
                )
            real_data = df["Real"].to_numpy(dtype=np.float32)
            imag_data = df["Imaginary"].to_numpy(dtype=np.float32)
            fft_data = real_data + 1j * imag_data
        else:
            dtype_to_use = np.int16 if signed_data else np.uint16
            raw_data = np.fromfile(data_input, dtype=dtype_to_use)

            if raw_data.size % 2 != 0:
                raise ValueError(
                    "The binary file does not contain an even number of 16-bit elements."
                )
            if raw_data.size % (fft_size * 2) != 0:
                raise ValueError(
                    f"The file size ({raw_data.size} elements) is not a multiple of fft_size.\n"
                    "Each FFT window must have fft_size complex samples (fft_size 16-bit values)."
                )

            complex_pairs = raw_data.reshape(-1, 2)
            fft_data = complex_pairs[:, 0].astype(np.float32) + 1j * complex_pairs[
                :, 1
            ].astype(np.float32)

    elif isinstance(data_input, np.ndarray):
        fft_data = data_input
    else:
        raise TypeError("data_input must be a string (file path) or a NumPy array.")

    if fft_data.ndim == 1:
        if fft_data.size % fft_size != 0:
            raise ValueError("FFT data length is not a multiple of fft_size.")
        waterfall = fft_data.reshape(-1, fft_size)
    elif fft_data.ndim == 2:
        if fft_data.shape[1] != fft_size:
            raise ValueError(
                "The FFT array must have fft_size elements per window (columns)."
            )
        waterfall = fft_data
    else:
        raise ValueError("FFT data must be a 1D or 2D array.")

    waterfall = np.flip(waterfall, axis=1)

    magnitude = np.abs(waterfall)
    with np.errstate(divide="ignore"):
        magnitude_dB = 20 * np.log10(magnitude)
    magnitude_dB[np.isneginf(magnitude_dB)] = 0

    plt.figure(figsize=(10, 8))
    im = plt.imshow(
        magnitude_dB,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[0, fft_size, waterfall.shape[0], 0],
        cmap="viridis",
        vmin=0,
        vmax=60,
    )
    plt.colorbar(im, label="Magnitude (dB)")
    plt.xlabel("FFT Bin")
    plt.ylabel("Time Window")

    if interactive_plots:
        plt.show()
    else:
        if path != "":
            plt.savefig(path, format="pdf", dpi=1000)
            plt.close()
        else:
            raise ValueError(
                "Il path per salvare il plot è vuoto. Fornisci un path valido o imposta INTERACTIVE_PLOTS a 'interactive'."
            )
