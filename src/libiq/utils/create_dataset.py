import numpy as np
import pandas as pd
import os
import cmath
import multiprocessing
from typing import Generator, List, Dict
from libiq.utils.constants import DTYPE, DATA_FORMAT, CHUNK_SIZE, N_JOBS, COLUMNS_LIST, DATA_FORMAT_OPTIONS

def columns_name(data_format: str) -> List[str]:
    """
    Retrieve column names based on the data format.

    Args:
        data_format (str): The format of the data (e.g., 'real-imag', 'phase-magnitude').

    Returns:
        List[str]: Column names corresponding to the specified data format.

    Raises:
        ValueError: If the specified data format is not recognized.
    """

    try:
        if data_format not in COLUMNS_LIST:
            raise ValueError(f"Unknown value for data format: {data_format}")
        
        return COLUMNS_LIST[data_format]
    
    except ValueError as e:
        raise e

def delete_csv_files(output_path: str) -> None:
    """
    Delete all CSV files in a specified directory.

    This function checks if the specified directory exists, verifies that all files are CSVs,
    and deletes them. If the directory does not exist or contains non-CSV files, it raises
    an error.

    Args:
        output_path (str): Directory path where CSV files are located.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If any non-CSV files are found in the directory.
    """

    try:
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"The specified path does not exist: {output_path}")

        files = os.listdir(output_path)
        non_csv_files = [f for f in files if not f.endswith('.csv')]
        if non_csv_files:
            raise ValueError(f"There are non-.csv files in the directory: {non_csv_files}")

        for file in files:
            file_path = os.path.join(output_path, file)
            if file.endswith('.csv'):
                os.remove(file_path)
        
        print(f"\nAll CSV files in {output_path} have been deleted.")
    
    except (FileNotFoundError, ValueError) as e:
        raise e

def read_data(file_path: str, dtype: np.dtype) -> Generator[np.ndarray, None, None]:
    """
    Read binary data from a file in chunks.

    This function opens a binary file and reads data of the specified dtype in chunks,
    yielding each chunk as a numpy array until the file is fully read.

    Args:
        file_path (str): Path to the binary file.
        dtype (np.dtype): Data type to read the binary file (e.g., np.float32).

    Yields:
        np.ndarray: Chunk of data read as a numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If an error occurs while reading data.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file path does not exist: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            while True:
                data = np.fromfile(f, dtype=dtype, count=2)
                if not data.size:
                    break
                yield data
    except Exception as e:
        raise ValueError(f"An error occurred while reading the binary data: {e}")

def process(data: np.ndarray, data_format: str, ground_truth: int) -> List:
    """
    Process complex data into a specified format.

    Based on the specified format, the function extracts real, imaginary, phase, and magnitude
    information from a complex number, and appends the ground truth label.

    Args:
        data (np.ndarray): Array containing complex numbers.
        data_format (str): Desired data format ('real-imag', 'phase-magnitude', 'complex', or 'all').
        ground_truth (int): Label representing ground truth.

    Returns:
        List: Processed data with the specified format and ground truth label.

    Raises:
        ValueError: If the data format is unrecognized or the input array is empty.
    """

    try:
        if len(data) == 0:
            raise ValueError("The input data array is empty.")
        
        complex_num = data[0]
        if data_format == 'real-imag':
            return [complex_num.real, complex_num.imag, ground_truth]
        elif data_format == 'phase-magnitude':
            phase = cmath.phase(complex_num)
            magnitude = abs(complex_num)
            magnitude_dB = 20 * np.log10(magnitude) if magnitude > 0 else -120
            return [phase, magnitude_dB, ground_truth]
        elif data_format == 'complex':
            return [complex_num, ground_truth]
        elif data_format == 'all':
            phase = cmath.phase(complex_num)
            magnitude = abs(complex_num)
            magnitude_dB = 20 * np.log10(magnitude) if magnitude > 0 else -120
            return [complex_num.real, complex_num.imag, phase, magnitude_dB, ground_truth]
        else:
            raise ValueError(f"Unknown value for data format: {data_format}")
    
    except ValueError as e:
        raise e

def handle_message(file_path: str, ground_truth: int, n_samples: float, output_path: str, num_files: int) -> None:
    """
    Process binary data and save it to CSV files in chunks.

    Reads binary data from a file, processes it according to a specified format, and writes
    processed data to multiple CSV files in a specified output directory.

    Args:
        file_path (str): Path to the binary file.
        ground_truth (int): Ground truth label for data.
        n_samples (float): Number of samples to process.
        output_path (str): Directory for saving CSV files.
        num_files (int): Number of CSV files to generate.

    Raises:
        FileNotFoundError: If the binary file is not found.
        ValueError: If there is an issue with data processing or CSV file writing.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file path does not exist: {file_path}")

    print(f"Converting {file_path} into corresponding .csv")
    counter = 0
    message: List[List] = []
    file_counter = 1

    def save_to_csv(message: List[List], file_counter: int) -> None:
        """
        Save the message list to a CSV file.

        Args:
            message (List[List]): The list of processed data rows.
            file_counter (int): The counter for the CSV file name.

        Raises:
            ValueError: If there is an error writing to the CSV file.
        """
        if message:
            file_name = os.path.join(output_path, f'{os.path.splitext(os.path.basename(file_path))[0]}_{file_counter}.csv')
            df = pd.DataFrame(message, columns=columns_name(DATA_FORMAT))
            try:
                df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
            except Exception as e:
                raise ValueError(f"Error writing to CSV file {file_name}: {e}")

    try:
        for data in read_data(file_path, DTYPE):
            if file_counter > num_files:
                break

            tmp = np.frombuffer(data, dtype=DTYPE)
            csv_row = process(tmp, DATA_FORMAT, ground_truth)
            message.append(csv_row)

            counter += 1

            if counter % CHUNK_SIZE == 0 or counter == n_samples:
                save_to_csv(message, file_counter)
                message = []
                if counter == n_samples:
                    counter = 0
                    file_counter += 1

        save_to_csv(message, file_counter)
    except Exception as e:
        raise ValueError(f"Error processing file {file_path}: {e}")

def combine_csv(csv_files: List[str], combined_csv_file_path: str) -> None:
    """
    Merge multiple CSV files into one.

    Reads each CSV file in chunks, adds a column indicating the source file, and appends
    each chunk to a single, combined CSV file.

    Args:
        csv_files (List[str]): List of paths to CSV files.
        combined_csv_file_path (str): Path to the output combined CSV file.

    Raises:
        FileNotFoundError: If any of the specified files do not exist.
        ValueError: If there is an error during reading or writing CSV data.
    """

    print("Combining all .csv into one")
    try:
        for file_path in csv_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            chunk_container = pd.read_csv(file_path, chunksize=CHUNK_SIZE)
            for chunk in chunk_container:
                chunk_copy = chunk.copy()
                chunk_copy.insert(0, 'File', os.path.basename(file_path))
                chunk_copy.to_csv(combined_csv_file_path, mode='a', header=not os.path.exists(combined_csv_file_path), index=False)
        
        print(f"All CSV files have been combined into {combined_csv_file_path}.")
    
    except (FileNotFoundError, ValueError) as e:
        raise e
    except Exception as e:
        raise ValueError(f"Error processing file {file_path}: {e}")

def create_dataset_from_bin(files: Dict[str, int], num_files: int, output_path: str, combined_output_path: str, n_samples: float = float('inf')) -> None:
    """
    Convert binary files to CSV format and combine them into one dataset.

    Reads, processes, and saves binary files as multiple CSV files. It combines these CSVs
    into a single dataset CSV file. Uses multiprocessing for parallel processing of multiple files.

    Args:
        files (Dict[str, int]): Dictionary with binary file paths as keys and labels as values.
        num_files (int): Number of CSV files to create for each binary file.
        output_path (str): Directory for saving individual CSV files.
        combined_output_path (str): Path for the combined dataset CSV file.
        n_samples (float): Number of samples per binary file to process. Default is all samples.

    Raises:
        ValueError: If an unrecognized data format is specified.
        FileNotFoundError: If no CSV files are generated from the binary files.
    """

    try:
        if DATA_FORMAT not in DATA_FORMAT_OPTIONS:
            raise ValueError("Data format not available")
        
        file_names = list(files.keys())
        ground_truths = list(files.values())

        delete_csv_files(output_path)

        with multiprocessing.Pool(processes=N_JOBS) as pool:
            pool.starmap(handle_message, zip(file_names, ground_truths, [n_samples] * len(file_names), [output_path] * len(file_names), [num_files] * len(file_names)))

        csv_files = [
            f"{output_path}{os.path.splitext(os.path.basename(file))[0]}_{i}.csv"
            for file in files
                for i in range(1, num_files + 1)
                    if os.path.exists(f"{output_path}{os.path.splitext(os.path.basename(file))[0]}_{i}.csv")
        ]

        if not csv_files:
            raise FileNotFoundError("None of the specified CSV files were found.")

        combine_csv(csv_files, combined_output_path)
    
    except (ValueError, FileNotFoundError) as e:
        raise e
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")

def create_dataset_from_csv(files: Dict[str, int], num_files: int, output_path: str, combined_csv_file_path: str) -> None:
    """
    Combine multiple pre-existing CSV files into one dataset.

    Reads and combines CSV files created from previous binary file processing. Adds a 'File' column 
    to indicate the source CSV file in the combined dataset.

    Args:
        files (Dict[str, int]): Dictionary with CSV file paths as keys and labels as values.
        num_files (int): Number of output files to create per input file.
        output_path (str): Directory containing the individual CSV files.
        combined_csv_file_path (str): Path for the combined CSV dataset.

    Raises:
        FileNotFoundError: If none of the specified CSV files are found.
    """

    try:
        csv_files = [
            f"{output_path}{os.path.splitext(os.path.basename(file))[0]}_{i}.csv"
            for file in files
                for i in range(1, num_files + 1)
                    if os.path.exists(f"{output_path}{os.path.splitext(os.path.basename(file))[0]}_{i}.csv")
        ]

        if not csv_files:
            raise FileNotFoundError("None of the specified CSV files were found.")

        combine_csv(csv_files, combined_csv_file_path)
    
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")