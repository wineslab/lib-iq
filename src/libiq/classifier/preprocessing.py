import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from libiq.utils.constants import RANDOM_STATE
from libiq.utils.logger import logger


def load_csv(file_path: str, columns: List[str]) -> pd.DataFrame:
    """
    Reads the specified CSV file and returns a DataFrame containing only the given columns.
    Raises a FileNotFoundError if the file does not exist.

    Parameters:
        file_path (str): Path to the CSV file.
        columns (List[str]): List of columns to read from the CSV.

    Returns:
        pd.DataFrame: The DataFrame containing the specified columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    return pd.read_csv(file_path, usecols=columns)


def normalize(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the input data.

    This is a placeholder function that currently returns the data unchanged.

    Parameters:
        data (pd.DataFrame): The DataFrame to be normalized.

    Returns:
        pd.DataFrame: The normalized DataFrame (currently identical to the input).
    """
    return data


def preprocess_data(
    csv_file_path: str,
    test_size: float,
    random_state: int = RANDOM_STATE,
    report: bool = False,
    report_path: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares the data for CNN training:

      1. Reads the CSV file using load_csv (expects columns: 'File', 'Real', 'Imaginary', 'Phase', 'Magnitude', 'Labels').
      2. Applies normalization (currently an identity function).
      3. Optionally generates a profiling report using ydata_profiling if 'report' is True and a valid 'report_path' is provided.
      4. Groups the data by the 'File' column, preserving the original order of samples.
      5. For each timeseries (group), extracts the 4 features (Real, Imaginary, Phase, Magnitude) as a NumPy array.
      6. Extracts the corresponding label for each timeseries (assuming the label is constant within the group).
      7. Splits the data into training and testing sets using train_test_split.

    Parameters:
        csv_file_path (str): Path to the CSV file.
        test_size (float): Fraction of the data to be used for testing.
        random_state (int): Random state for reproducibility.
        report (bool): If True, a profiling report is generated.
        report_path (str): Directory where the profiling report will be saved.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            x_train: Training set features with shape (number of files, samples per file, 4)
            x_test: Testing set features with the same structure as x_train.
            y_train: Training set labels.
            y_test: Testing set labels.
    """

    cols = ["File", "Real", "Imaginary", "Phase", "Magnitude", "Labels"]
    df = load_csv(csv_file_path, cols)
    df = normalize(df)

    if report and report_path:
        try:
            from ydata_profiling import ProfileReport

            pr = ProfileReport(df, title="Profiling Report", explorative=True)
            pr.to_file(report_path)
            logger.info(f"Profiling report saved to: {report_path}")
        except ImportError:
            raise ImportError(
                "Optional dependency 'ydata-profiling' is not installed. "
                "You can install it with: pip install libiq[profile]"
            ) from None

    grouped = df.groupby("File", sort=False)

    X_list = []
    y_list = []

    for file_name, group in grouped:
        ts = group[["Real", "Imaginary", "Phase", "Magnitude"]].to_numpy()
        X_list.append(ts)
        y_list.append(group["Labels"].iloc[0])

    X = np.array(X_list)
    y = np.array(y_list)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return x_train, x_test, y_train, y_test
