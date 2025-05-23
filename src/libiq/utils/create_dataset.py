import csv
import multiprocessing
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from libiq.classifier.energy_detector import energy_detector
from libiq.utils.logger import logger


def rename_duplicate_files(files: Dict[str, int]) -> Dict[str, int]:
    """
    Checks if there are duplicate files in the input (i.e., files with the same filename, which is the last part of the path).
    In such cases, it physically renames the file by appending a numeric suffix, ensuring each file has a unique name.

    Args:
        files: A dictionary mapping file paths to their labels.

    Returns:
        A new dictionary with updated file paths (each unique).
    """
    seen = {}
    updated_files = {}
    for file_path, label in files.items():
        directory, filename = os.path.split(file_path)

        if filename in seen:
            count = seen[filename]
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_{count}{ext}"
            new_file_path = os.path.join(directory, new_filename)

            while os.path.exists(new_file_path):
                count += 1
                new_filename = f"{base}_{count}{ext}"
                new_file_path = os.path.join(directory, new_filename)
            os.rename(file_path, new_file_path)
            seen[filename] = count + 1
            updated_files[new_file_path] = label
        else:
            seen[filename] = 1
            updated_files[file_path] = label
    return updated_files


def delete_csv_files(directory: str) -> None:
    """
    Deletes all CSV files in the specified directory.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If non-CSV files are found in the directory.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The specified directory does not exist: {directory}")

    files = os.listdir(directory)
    non_csv_files = [f for f in files if not f.endswith(".csv")]
    if non_csv_files:
        raise ValueError(f"Non-CSV files found in the directory: {non_csv_files}")

    for file in files:
        if file.endswith(".csv"):
            os.remove(os.path.join(directory, file))

    logger.debug(f"All CSV files in '{directory}' have been deleted.")


def combine_csv_files(csv_files: List[str], output_file: str) -> None:
    """
    Combines multiple CSV files into a single CSV file.
    A new column 'File' is inserted to record the source file name.

    Raises:
        FileNotFoundError: If any of the CSV files is not found.
        ValueError: If there are no CSV files to combine.
    """
    dataframes = []
    for file_path in csv_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path)
        df.insert(0, "File", os.path.basename(file_path))
        dataframes.append(df)

    if not dataframes:
        raise ValueError("No CSV files to combine.")

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    logger.info(f"All CSV files have been combined into '{output_file}'.")


def read_binary_data(
    file_path: str, dtype: np.dtype, max_rows: int = None
) -> np.ndarray:
    """
    Reads a binary file and converts it into a NumPy array of complex numbers.
    Only the number of FFT rows specified by max_rows is read (if provided).

    Parameters:
        file_path: Path to the binary file.
        dtype: Data type for reading the binary data.
        max_rows: (Optional) Maximum number of FFT rows to read.
                  Each FFT row contains 1536 complex numbers (i.e., 1536*2 values).

    Returns:
        NumPy array of complex numbers.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the binary file does not contain an even number of elements.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified path does not exist: {file_path}")

    try:
        count = None
        n_iq_data = 1536
        if max_rows is not None:
            count = max_rows * n_iq_data * 2
        data = np.fromfile(file_path, dtype=dtype, count=count)
        if data.size % 2 != 0:
            raise ValueError(
                "The binary file does not contain an even number of elements."
            )
        data = data.reshape(-1, 2)
        complex_data = data[:, 0] + 1j * data[:, 1]
        return complex_data
    except Exception as e:
        raise ValueError(f"Error reading binary data: {e}") from None


def process_samples_vectorized(data: np.ndarray, ground_truth: int) -> pd.DataFrame:
    """
    Converts an array of complex numbers into a DataFrame with columns in the desired format.

    Parameters:
        data: Array of complex numbers.
        ground_truth: Label associated with the data.

    Returns:
        A pandas DataFrame containing the processed data.
    """
    df = pd.DataFrame()
    df["Real"] = np.real(data)
    df["Imaginary"] = np.imag(data)
    df["Phase"] = np.angle(data)
    magnitude = np.abs(data)
    with np.errstate(divide="ignore"):
        magnitude_dB = 20 * np.log10(magnitude)
    magnitude_dB[np.isneginf(magnitude_dB)] = 0
    df["Magnitude"] = magnitude_dB
    df["Labels"] = ground_truth
    return df


def split_dataframe(
    df: pd.DataFrame, n_samples: int, num_files: int
) -> Dict[int, pd.DataFrame]:
    """
    Splits a DataFrame into multiple DataFrames, each containing n_samples rows.
    A maximum of num_files splits are created.

    Returns:
        A dictionary where the key is the file index (starting from 1) and the value is the corresponding DataFrame.
    """
    dfs = {}
    max_samples = n_samples * num_files
    df = df.iloc[:max_samples]
    for i in range(num_files):
        start = i * n_samples
        end = start + n_samples
        chunk = df.iloc[start:end]
        if chunk.empty:
            break
        dfs[i + 1] = chunk
    return dfs


def save_dataframes_to_csv(
    dataframes: Dict[int, pd.DataFrame], base_filename: str, output_path: str
) -> List[Tuple[str, int]]:
    """
    Saves each DataFrame in the dictionary to a CSV file.
    The file names are constructed based on the base filename and the file index.

    Returns:
        A list of tuples (file_path, number_of_rows).
    """
    saved_files = []
    base_name = os.path.splitext(os.path.basename(base_filename))[0]
    for file_index, df in dataframes.items():
        file_name = f"{base_name}_{file_index}.csv"
        file_path = os.path.join(output_path, file_name)
        df.to_csv(file_path, index=False)
        saved_files.append((file_path, len(df)))
    return saved_files


def process_binary_file(
    file_path: str,
    ground_truth: int,
    output_path: str,
    input_vector: int,
    num_files: int,
    dtype: np.dtype,
    extraction_window: int = 600,
    moving_avg_window: int = 5,
) -> Tuple[List[str], int]:
    """
    Processes a binary file by:
      - Reading only the first 'input_vector' FFT rows from the binary file.
      - Determining the total number of FFT rows read.
      - Reshaping the data into a matrix of shape (total_rows, 1536).
      - Applying the energy detector for horizontal cropping and unpacking the results.
      - Converting the flattened data into a DataFrame.
      - Splitting the DataFrame into chunks and saving each chunk as a CSV file.

    Returns:
        A tuple containing:
          - A list of CSV file paths that were created.
          - The number of samples (updated_n_samples) expected in each file.
    """
    n_iq_data = 1536

    data = read_binary_data(file_path, dtype, max_rows=input_vector)

    total_rows = len(data) // n_iq_data
    if total_rows == 0:
        raise ValueError("No complete FFT rows were read from the file.")

    data_matrix = data[: total_rows * n_iq_data].reshape(total_rows, n_iq_data)

    updated_n_samples, cropped_data = energy_detector(
        data_matrix, extraction_window, moving_avg_window
    )

    df = process_samples_vectorized(cropped_data, ground_truth)

    df_chunks = split_dataframe(df, updated_n_samples, num_files)
    saved_files_info = save_dataframes_to_csv(df_chunks, file_path, output_path)

    for file_name, rows in saved_files_info:
        if rows != updated_n_samples:
            logger.warning(
                f"Warning: The file '{file_name}' contains {rows} samples instead of the expected {updated_n_samples}."
            )

    return ([file_name for file_name, _ in saved_files_info], updated_n_samples)


def combine_csv_files_with_check(
    csv_files: List[str], output_file: str, n_samples: int
) -> None:
    """
    Combines multiple CSV files into a single CSV file with an additional 'File' column.
    Checks that each file contains exactly n_samples rows.

    Raises:
        FileNotFoundError: If any CSV file is not found.
        ValueError: If a CSV file does not contain the expected number of rows.
    """
    logger.info("Starting combining files")
    seen_base_names = set()
    header_written = False
    total_count_appended = 0

    with open(output_file, "w", newline="") as outfile:
        writer = None

        for file_path in csv_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            base_name = os.path.basename(file_path)
            if base_name in seen_base_names:
                logger.warning(
                    f"Warning: Duplicate base file name detected: '{base_name}'. Skipping file '{file_path}'."
                )
                continue
            seen_base_names.add(base_name)

            with open(file_path, "r", newline="") as infile:
                reader = csv.reader(infile)
                try:
                    file_header = next(reader)
                except StopIteration:
                    raise ValueError(f"File '{file_path}' is empty.") from None

                if not header_written:
                    writer = csv.writer(outfile)
                    writer.writerow(["File"] + file_header)
                    header_written = True

                count = 0
                for row in reader:
                    writer.writerow([base_name] + row)
                    count += 1

                if count != n_samples:
                    raise ValueError(
                        f"The file '{file_path}' contains {count} samples instead of {n_samples}."
                    )
                total_count_appended += count

    logger.info(f"All CSV files have been merged into '{output_file}'.")


def create_dataset_from_bin(
    files: Dict[str, int],
    num_files: int,
    output_path: str,
    combined_output_path: str,
    input_vector: int,
    extraction_window: int = 1536,
    moving_avg_window: int = 5,
    dtype: np.dtype = np.int16,
) -> None:
    """
    Converts a set of binary files into CSV files (with a size check) and then combines all CSV files into one.

    Parameters:
        files: A dictionary mapping binary file paths to their labels.
        num_files: Maximum number of CSV files to create per binary file.
        output_path: Directory where the individual CSV files will be saved.
        combined_output_path: Path for the combined CSV file.
        input_vector: Number of FFT rows (input vectors) to read from each binary file.
        dtype: Data type for reading the binary data.
        extraction_window: Number of columns to extract around the energy peak (default 1536).
        moving_avg_window: Size of the moving average window for smoothing (default 5).
    """
    files = rename_duplicate_files(files)

    delete_csv_files(output_path)
    csv_files = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = []
    updated_samples_list = []

    for file_path, ground_truth in files.items():
        result = pool.apply_async(
            process_binary_file,
            args=(
                file_path,
                ground_truth,
                output_path,
                input_vector,
                num_files,
                dtype,
                extraction_window,
                moving_avg_window,
            ),
        )
        results.append(result)
    pool.close()
    pool.join()
    for res in results:
        file_names, updated_n_samples = res.get()
        csv_files.extend(file_names)
        updated_samples_list.append(updated_n_samples)

    if not csv_files:
        raise FileNotFoundError("No CSV file was created.")

    if not all(x == updated_samples_list[0] for x in updated_samples_list):
        logger.warning(
            "Warning: Not all binary files produced the same number of samples after cropping."
        )

    combine_csv_files_with_check(
        csv_files, combined_output_path, updated_samples_list[0]
    )


def create_dataset_from_csv(
    files: Dict[str, int], num_files: int, output_path: str, combined_csv_file_path: str
) -> None:
    """
    Combines the CSV files (already created) into a single CSV file.

    Parameters:
        files: A dictionary mapping the original binary file paths to their labels.
        num_files: Maximum number of CSV files per binary file (used to reconstruct file names).
        output_path: Directory containing the CSV files.
        combined_csv_file_path: Path for the combined CSV file.
    """
    files = rename_duplicate_files(files)

    csv_files = []
    for file in files.keys():
        base_name = os.path.splitext(os.path.basename(file))[0]
        for i in range(1, num_files + 1):
            csv_file = os.path.join(output_path, f"{base_name}_{i}.csv")
            if os.path.exists(csv_file):
                csv_files.append(csv_file)

    if not csv_files:
        raise FileNotFoundError("No specified CSV files were found.")

    combine_csv_files(csv_files, combined_csv_file_path)
