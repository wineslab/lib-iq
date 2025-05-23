from typing import Tuple

import numpy as np
from libiq.utils.logger import logger


def energy_detector(
    data_matrix: np.ndarray, extraction_window: int, moving_avg_window: int = 5
) -> Tuple[int, np.ndarray]:
    """
    Applies an energy detector to a 2D matrix of IQ samples.
    The matrix is assumed to have shape (n_rows, 1536), where each row represents an FFT snapshot.
    The function computes the energy (sum of squared magnitudes) for each column, smooths this energy vector,
    and identifies the column with maximum energy. Then it extracts a contiguous block of columns (of width extraction_window)
    centered around that peak using circular (wrap-around) indexing.

    Additionally, regardless of the peak detection, columns from index 400 to 500 are removed before further processing.

    Parameters:
        data_matrix: 2D NumPy array of complex numbers.
        extraction_window: The number of columns to extract.
        moving_avg_window: The window size for smoothing the energy vector.

    Returns:
        A tuple containing:
          - The total number of samples in the extracted matrix.
          - A submatrix of shape (n_rows, extraction_window) containing the selected columns with the peak centered.
    """
    n_rows, n_cols = data_matrix.shape

    data_matrix = data_matrix[:, 80:-30]

    n_cols = data_matrix.shape[1]

    if n_cols <= extraction_window:
        total_samples = data_matrix.shape[0] * data_matrix.shape[1]
        return total_samples, data_matrix

    energy_per_column = np.sum(np.abs(data_matrix) ** 2, axis=0)
    kernel = np.ones(moving_avg_window) / moving_avg_window
    smoothed_energy = np.convolve(energy_per_column, kernel, mode="same")

    peak_index = np.argmax(smoothed_energy)
    half_window = extraction_window // 2

    indices = np.mod(
        np.arange(
            peak_index - half_window, peak_index - half_window + extraction_window
        ),
        n_cols,
    )

    cropped_matrix = data_matrix[:, indices]
    total_samples = cropped_matrix.shape[0] * cropped_matrix.shape[1]
    data_flat = cropped_matrix.flatten(order="C")
    return total_samples, data_flat
