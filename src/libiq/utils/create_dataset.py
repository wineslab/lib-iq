import os
import multiprocessing
import numpy as np
import pandas as pd
import cmath
import multiprocessing
from typing import List, Dict, Tuple

def delete_csv_files(directory: str) -> None:
    """
    Deletes all CSV files in the specified directory.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If there are non-CSV files in the directory.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The specified directory does not exist: {directory}")

    files = os.listdir(directory)
    non_csv_files = [f for f in files if not f.endswith('.csv')]
    if non_csv_files:
        raise ValueError(f"Non-CSV files found in the directory: {non_csv_files}")

    for file in files:
        if file.endswith('.csv'):
            os.remove(os.path.join(directory, file))
    
    print(f"All CSV files in '{directory}' have been deleted.")


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
        df.insert(0, 'File', os.path.basename(file_path))
        dataframes.append(df)
    
    if not dataframes:
        raise ValueError("No CSV files to combine.")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"All CSV files have been combined into '{output_file}'.")


def read_binary_data(file_path: str, dtype: np.dtype) -> np.ndarray:
    """
    Reads a binary file and converts it into a NumPy array of complex numbers.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the binary file does not contain an even number of elements.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified path does not exist: {file_path}")
    
    try:
        data = np.fromfile(file_path, dtype=dtype)
        if data.size % 2 != 0:
            raise ValueError("The binary file does not contain an even number of elements.")
        data = data.reshape(-1, 2)
        complex_data = data[:, 0] + 1j * data[:, 1]
        return complex_data
    except Exception as e:
        raise ValueError(f"Error reading binary data: {e}")


def apply_fft_filter(data: np.ndarray, fft_filter: Tuple[float, float]) -> np.ndarray:
    """
    Applies a frequency-domain filter to the data.
    
    Parameters:
        data: Array of complex numbers.
        fft_filter: Tuple (freq_min, freq_max). If (-inf, inf), the data is returned unchanged.
    
    Returns:
        The filtered data obtained via IFFT.
    """
    freq_min, freq_max = fft_filter
    if freq_min == -float('inf') and freq_max == float('inf'):
        return data

    n = len(data)
    fft_data = np.fft.fft(data)
    filtered_fft = np.zeros_like(fft_data)

    freq_min = int(max(0, freq_min))
    freq_max = int(min(n // 2, freq_max))

    filtered_fft[freq_min:freq_max] = fft_data[freq_min:freq_max]

    if freq_min != 0:
        filtered_fft[-freq_max:-freq_min] = fft_data[-freq_max:-freq_min]
    filtered_data = np.fft.ifft(filtered_fft)
    return filtered_data


def process_samples_vectorized(data: np.ndarray, ground_truth: int) -> pd.DataFrame:
    """
    Converts an array of complex numbers into a DataFrame with columns according to the desired format.

    Parameters:
        data: Array of complex numbers.
        ground_truth: Label associated with the data.
    
    Returns:
        A pandas DataFrame containing the processed data.
    """
    df = pd.DataFrame()
    df['Real'] = np.real(data)
    df['Imaginary'] = np.imag(data)
    df['Phase'] = np.angle(data)
    magnitude = np.abs(data)
    with np.errstate(divide='ignore'):
        magnitude_dB = 20 * np.log10(magnitude)
    magnitude_dB[np.isneginf(magnitude_dB)] = 0
    df['Magnitude'] = magnitude_dB
    df['Labels'] = ground_truth
    
    return df


def split_dataframe(df: pd.DataFrame, n_samples: int, num_files: int) -> Dict[int, pd.DataFrame]:
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


def save_dataframes_to_csv(dataframes: Dict[int, pd.DataFrame], base_filename: str, output_path: str) -> List[Tuple[str, int]]:
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


def process_binary_file(file_path: str,
                        ground_truth: int,
                        output_path: str,
                        n_samples: int,
                        num_files: int,
                        dtype: np.dtype,
                        fft_filter: Tuple[float, float] = (-float('inf'), float('inf'))
                        ) -> List[str]:
    """
    Processes a binary file:
      1. Reads the entire file and converts it to complex numbers.
      2. Applies an optional FFT filter.
      3. Converts the samples into the desired format.
      4. Splits the data into multiple CSV files (each containing n_samples samples, up to num_files files).

    Returns:
        A list of file paths for the saved CSV files.
    """
    data = read_binary_data(file_path, dtype)
    
    data = apply_fft_filter(data, fft_filter)
    
    df = process_samples_vectorized(data, ground_truth)
    
    df_chunks = split_dataframe(df, n_samples, num_files)
    
    saved_files_info = save_dataframes_to_csv(df_chunks, file_path, output_path)
    
    for file_name, num_rows in saved_files_info:
        if num_rows != n_samples:
            print(f"The file '{file_name}' contains {num_rows} samples instead of {n_samples}.")
    
    return [file_name for file_name, _ in saved_files_info]


def create_dataset_from_bin(files: Dict[str, int],
                            num_files: int,
                            output_path: str,
                            combined_output_path: str,
                            n_samples: int,
                            dtype: np.dtype = np.int16,
                            fft_filter: Tuple[float, float] = (-float('inf'), float('inf')),
                            ) -> None:
    """
    Converts a set of binary files into CSV files (up to num_files per binary file),
    optionally applying a frequency-domain filter, and finally combines all CSV files into one.

    Parameters:
        files: Dictionary mapping the binary file path to its label.
        num_files: Maximum number of CSV files to create per binary file.
        output_path: Directory where the CSV files will be saved.
        combined_output_path: Path for the combined CSV file.
        n_samples: Number of samples per CSV file.
        fft_filter: Frequency filter to apply (default: no filter).
    """
    delete_csv_files(output_path)
    
    csv_files = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = []
        for file_path, ground_truth in files.items():
            result = pool.apply_async(
                process_binary_file,
                args=(file_path, ground_truth, output_path, n_samples, num_files, dtype, fft_filter)
            )
            results.append(result)
        pool.close()
        pool.join()
        
        for res in results:
            csv_files.extend(res.get())
    
    if not csv_files:
        raise FileNotFoundError("No CSV file was created.")
    
    combine_csv_files(csv_files, combined_output_path)


def create_dataset_from_csv(files: Dict[str, int],
                            num_files: int,
                            output_path: str,
                            combined_csv_file_path: str) -> None:
    """
    Combines the CSV files (already created) into a single CSV file.
    
    Parameters:
        files: Dictionary mapping the original binary file path to its label.
        num_files: Maximum number of CSV files per binary file (used to reconstruct file names).
        output_path: Directory containing the CSV files.
        combined_csv_file_path: Path for the combined CSV file.
    """
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
