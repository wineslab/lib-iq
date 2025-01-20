import pandas as pd
import numpy as np
from typing import List, Union, Tuple, Dict
from libiq.utils.constants import STATIC_LABELS, COMBINED_CSV_FILE_PATH, ORIGINAL_COMBINED_CSV_FILE_PATH, LABELS, MODE
from libiq.iq_clustering.preprocessing import load_csv, aggregate_columns
import shutil
from scipy.stats import skew, kurtosis
from scipy.signal import periodogram


'''
def calculate_energy(complex_signal: np.ndarray) -> float:
    """
    Calculates the energy of a complex signal using the Fast Fourier Transform (FFT).

    Args:
        complex_signal (np.ndarray): A NumPy array containing the complex signal.

    Returns:
        float: The calculated energy of the signal.
    
    Raises:
        TypeError: If complex_signal is not a numpy array.
        ValueError: If complex_signal is empty.
    """

    try:
        if complex_signal.size == 0:
            raise ValueError("complex_signal cannot be empty.")
        
        fft_result = np.fft.fft(complex_signal)
        energy = np.mean(np.abs(fft_result) ** 2)
        return energy
    
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

def calculate_threshold(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Calculates the energy threshold based on the average and minimum energy values
    for each group of files in the DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing 'File', 'Real', and 'Imaginary' columns.

    Returns:
        Tuple[float, float, float]: A tuple containing the calculated threshold, the average energy, and the minimum energy.
    
    Raises:
        TypeError: If df is not a DataFrame.
        ValueError: If required columns are not present in the DataFrame.
    """

    try:        
        grouped = df.groupby('File')
        energies = []

        for name, group in grouped:
            complex_signal = group['Real'].values + 1j * group['Imaginary'].values
            energy = calculate_energy(complex_signal)
            energies.append(energy)

        average_energy = np.mean(energies)
        min_energy = np.min(energies)
        threshold = (average_energy + min_energy) / 2
        return threshold, average_energy, min_energy
    
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

def energy_detection(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Assigns labels based on the calculated energy for each group of files in the DataFrame
    and updates the 'Labels' column.

    Args:
        df (pd.DataFrame): A DataFrame containing 'File', 'Real', 'Imaginary', and 'Labels' columns.
        threshold (float): The energy threshold used to determine the label.

    Returns:
        pd.DataFrame: An updated DataFrame with assigned labels in the 'Labels' column.
    
    Raises:
        TypeError: If df is not a DataFrame or if threshold is not a float.
        ValueError: If required columns are not present in the DataFrame.
    """

    try:
        if not all(col in df.columns for col in ['File', 'Real', 'Imaginary', 'Labels']):
            raise ValueError("DataFrame must contain 'File', 'Real', 'Imaginary', and 'Labels' columns.")
        
        for name, group in df.groupby('File'):
            complex_signal = group['Real'].to_numpy() + 1j * group['Imaginary'].to_numpy()
            energy = calculate_energy(complex_signal)
            label = LABELS['WIFI'] if energy > threshold else LABELS['Noise']
            df.loc[group.index, 'Labels'] = label

        return df
    
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

def update_csv(df: pd.DataFrame, file_path: str, backup_path: str) -> None:
    """
    Updates the existing CSV file with new labels and creates a backup of the original file.

    Args:
        df (pd.DataFrame): A DataFrame containing updated labels.
        file_path (str): The path to the existing CSV file.
        backup_path (str): The path where the backup of the original file will be saved.

    Raises:
        TypeError: If df is not a DataFrame or if file_path or backup_path are not strings.
        ValueError: If the required 'File' and 'Labels' columns are not present in the CSV file.
    """

    try:        
        shutil.copy(file_path, backup_path)
        
        combined_df = pd.read_csv(file_path)
        
        if 'File' not in combined_df.columns or 'Labels' not in combined_df.columns:
            raise ValueError("The columns 'File' and 'Labels' must be present in the CSV.")
        
        df_labels_dict = df.set_index('File')['Labels'].to_dict()

        def update_label(row):
            if row['File'] in df_labels_dict and row['Labels'] != df_labels_dict[row['File']]:
                return df_labels_dict[row['File']]
            else:
                return row['Labels']

        combined_df['Labels'] = combined_df.apply(update_label, axis=1)

        combined_df.to_csv(file_path, index=False)
    
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

def noise_detector(file: str, backup_path: str) -> None:
    """
    Detects noise in signals from a CSV file, calculates the energy threshold, and assigns labels.

    Args:
        file (str): The path to the CSV file containing the data.
        backup_path (str): The path where the backup of the original file will be saved.
        MODE (Union[str, None], optional): The MODE of column selection ('PCA', 'Magnitude', 'Magnitude-PCA', 'all'). Defaults to None.

    Raises:
        TypeError: If file or backup_path are not strings.
        ValueError: If an error occurs during preprocessing or if the file extension is not '.csv'.
    """
    
    try:        
        extension = (file.split("/")[-1]).split(".")[-1]
        if extension != "csv":
            raise ValueError(f"Unexpected file extension, .csv required, got .{extension}")
        
        columns = []
        if MODE is None:
            columns = ['File', 'Real', 'Imaginary', 'Labels']
        elif MODE == 'Magnitude':
            return
        elif MODE == 'all':
            columns = ['File', 'Real', 'Imaginary', 'Magnitude', 'Labels']
        else:
            raise ValueError(f"Mode {MODE} not supported")

        df = load_csv(file, columns)
        if df is None:
            raise ValueError("Failed to load the CSV file for x_df.")

        df = df[df['Labels'] == LABELS['WIFI']]

        if df.empty:
            return
        
        threshold, average_energy, min_energy = calculate_threshold(df)
        print(f"\nAverage energy: {average_energy}")
        print(f"Minimum energy: {min_energy}")
        print(f"Calculated threshold: {threshold}")

        labeled_df = energy_detection(df, threshold)
        label_counts = labeled_df['Labels'].value_counts()

        print("Number of occurrences for each label in the 'Labels' column:")
        print(label_counts)

        #update_csv(labeled_df, file, backup_path)

    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        raise e
'''









def calculate_energy(complex_signal: np.ndarray) -> float:
    """
    Calculates the energy of a complex signal using the Fast Fourier Transform (FFT).

    Args:
        complex_signal (np.ndarray): A NumPy array containing the complex signal.

    Returns:
        float: The calculated energy of the signal.
    
    Raises:
        TypeError: If complex_signal is not a numpy array.
        ValueError: If complex_signal is empty.
    """

    try:
        if complex_signal.size == 0:
            raise ValueError("complex_signal cannot be empty.")
        
        fft_result = np.fft.fft(complex_signal)
        energy = np.mean(np.abs(fft_result) ** 2)
        return energy
    
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

def energy_detection(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """
    Assign labels based on calculated energy thresholds for each file in the DataFrame,
    updating the 'Labels' column based on specific threshold criteria.

    Args:
        df (pd.DataFrame): DataFrame containing 'File', 'Real', 'Imaginary', and 'Labels' columns.
        thresholds (Dict[str, float]): Dictionary with energy thresholds used for label assignment.

    Returns:
        pd.DataFrame: Updated DataFrame with assigned labels in the 'Labels' column.

    Raises:
        ValueError: If DataFrame does not contain required columns or if threshold values are invalid.
    """

    try:
        if not all(col in df.columns for col in ['File', 'Real', 'Imaginary', 'Labels']):
            raise ValueError("DataFrame must contain 'File', 'Real', 'Imaginary', and 'Labels' columns.")
        
        unique_files = df['File'].unique()
        for file in unique_files:
            energy_metrics = calculate_energy_statistics(df[df['File'] == file])
            counter = 0
            if energy_metrics['mean_energy_freq'] > thresholds['mean_energy_freq']:
                counter += 1
            if energy_metrics['max_energy_freq'] > thresholds['max_energy_freq']:
                counter += 1
            if energy_metrics['var_energy_freq'] > thresholds['var_energy_freq']:
                counter += 1
            if energy_metrics['time_domain_energy'] > thresholds['time_domain_energy']:
                counter += 1
            if energy_metrics['rms'] > thresholds['rms']:
                counter += 1
            if energy_metrics['peak_value'] > thresholds['peak_value']:
                counter += 1

            if counter == 6:
                df.loc[df['File'] == file, 'Labels'] = LABELS['WIFI']
            else:
                df.loc[df['File'] == file, 'Labels'] = LABELS['Noise']

        return df
    
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

def update_csv(df: pd.DataFrame, file_path: str, backup_path: str) -> None:
    """
    Updates the existing CSV file with new labels and creates a backup of the original file.

    Args:
        df (pd.DataFrame): A DataFrame containing updated labels.
        file_path (str): The path to the existing CSV file.
        backup_path (str): The path where the backup of the original file will be saved.

    Raises:
        TypeError: If df is not a DataFrame or if file_path or backup_path are not strings.
        ValueError: If the required 'File' and 'Labels' columns are not present in the CSV file.
    """

    try:        
        shutil.copy(file_path, backup_path)
        
        combined_df = pd.read_csv(file_path)
        
        if 'File' not in combined_df.columns or 'Labels' not in combined_df.columns:
            raise ValueError("The columns 'File' and 'Labels' must be present in the CSV.")
        
        df_labels_dict = df.set_index('File')['Labels'].to_dict()

        def update_label(row):
            if row['File'] in df_labels_dict and row['Labels'] != df_labels_dict[row['File']]:
                return df_labels_dict[row['File']]
            else:
                return row['Labels']

        combined_df['Labels'] = combined_df.apply(update_label, axis=1)

        combined_df.to_csv(file_path, index=False)
    
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

def calculate_energy_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate a range of energy-related statistics for a complex signal from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'Real' and 'Imaginary' columns with complex signal data.

    Returns:
        Dict[str, float]: Dictionary of calculated energy metrics, including min, max, mean, variance, 
                          RMS, peak values, skewness, kurtosis, and other key indicators.

    Raises:
        ValueError: If required columns are missing in the DataFrame.
    """

    real = df.iloc[:, 1]
    imaginary = df.iloc[:, 2]

    complex_signal = real + 1j * imaginary

    fft_result = np.fft.fft(complex_signal)

    energy_freq = np.abs(fft_result) ** 2

    min_energy = np.min(energy_freq)
    mean_energy = np.mean(energy_freq)
    max_energy = np.max(energy_freq)
    std_energy = np.std(energy_freq)
    var_energy = np.var(energy_freq)

    time_domain_energy = np.mean(np.abs(complex_signal) ** 2)

    mean_power = np.mean(np.abs(complex_signal) ** 2)

    rms = np.sqrt(mean_power)

    peak_value = np.max(np.abs(complex_signal))

    papr = peak_value / rms

    crest_factor = peak_value / rms

    real_part = np.real(complex_signal)
    imag_part = np.imag(complex_signal)
    amplitude = np.abs(complex_signal)
    phase = np.angle(complex_signal)

    skewness_real = skew(real_part)
    kurtosis_real = kurtosis(real_part)

    skewness_imag = skew(imag_part)
    kurtosis_imag = kurtosis(imag_part)

    skewness_amplitude = skew(amplitude)
    kurtosis_amplitude = kurtosis(amplitude)

    variance_amplitude = np.var(amplitude)

    stats = {
        'min_energy_freq': min_energy,
        'mean_energy_freq': mean_energy,
        'max_energy_freq': max_energy,
        'std_energy_freq': std_energy,
        'var_energy_freq': var_energy,
        'time_domain_energy': time_domain_energy,
        'mean_power': mean_power,
        'rms': rms,
        'peak_value': peak_value,
        'papr': papr,
        'crest_factor': crest_factor,
        'skewness_real': skewness_real,
        'kurtosis_real': kurtosis_real,
        'skewness_imag': skewness_imag,
        'kurtosis_imag': kurtosis_imag,
        'skewness_amplitude': skewness_amplitude,
        'kurtosis_amplitude': kurtosis_amplitude,
        'variance_amplitude': variance_amplitude
    }

    return stats

def noise_detector(file: str, backup_path: str) -> None:
    """
    Detect noise in a CSV file, calculate energy thresholds, and assign labels for noise detection.

    Args:
        file (str): Path to the CSV file containing the data.
        backup_path (str): Path to save a backup of the original file.
    
    Raises:
        ValueError: If file format is not CSV or required data is missing.
        Exception: If other unexpected errors occur during processing.
    """
    
    try:        
        extension = (file.split("/")[-1]).split(".")[-1]
        if extension != "csv":
            raise ValueError(f"Unexpected file extension, .csv required, got .{extension}")
        
        columns = []
        if MODE is None:
            columns = ['File', 'Real', 'Imaginary', 'Labels']
        elif MODE == 'Magnitude':
            return
        elif MODE == 'all':
            columns = ['File', 'Real', 'Imaginary', 'Magnitude', 'Labels']
        else:
            raise ValueError(f"Mode {MODE} not supported")

        df = load_csv(file, columns)
        if df is None:
            raise ValueError("Failed to load the CSV file for x_df.")

        df = df[df['Labels'] == LABELS['Noise']]

        if df.empty:
            print("No noise data found for threshold calculation.")
            return

        unique_files = df['File'].unique()

        thresholds = {
            'mean_energy_freq': 0.0,
            'max_energy_freq': 0.0,
            'var_energy_freq': 0.0,
            'time_domain_energy': 0.0,
            'rms': 0.0,
            'peak_value': 0.0
        }

        for unique_file in unique_files:
            noise_energy_stats = calculate_energy_statistics(df[df['File'] == unique_file])
            if thresholds['mean_energy_freq'] < noise_energy_stats['mean_energy_freq']:
                thresholds['mean_energy_freq'] = noise_energy_stats['mean_energy_freq']
            if thresholds['max_energy_freq'] < noise_energy_stats['max_energy_freq']:
                thresholds['max_energy_freq'] = noise_energy_stats['max_energy_freq']
            if thresholds['var_energy_freq'] < noise_energy_stats['var_energy_freq']:
                thresholds['var_energy_freq'] = noise_energy_stats['var_energy_freq']
            if thresholds['time_domain_energy'] < noise_energy_stats['time_domain_energy']:
                thresholds['time_domain_energy'] = noise_energy_stats['time_domain_energy']
            if thresholds['rms'] < noise_energy_stats['rms']:
                thresholds['rms'] = noise_energy_stats['rms']
            if thresholds['peak_value'] < noise_energy_stats['peak_value']:
                thresholds['peak_value'] = noise_energy_stats['peak_value']

        df = load_csv(file, columns)
        df = df[df['Labels'] == LABELS['WIFI']]

        if df.empty:
            return
    
        print('Mean_energy_freq: ', thresholds['mean_energy_freq'])
        print('Max_energy_freq: ', thresholds['max_energy_freq'])
        print('Var_energy_freq: ', thresholds['var_energy_freq'])
        print('Time_domain_energy: ', thresholds['time_domain_energy'])
        print('Rms: ', thresholds['rms'])
        print('Peak_value: ', thresholds['peak_value'])

        labeled_df = energy_detection(df, thresholds)
        
        label_counts = df.drop_duplicates(subset='File')['Labels'].map(STATIC_LABELS).value_counts()

        print("Number of occurrences for each label in the 'Labels' column:")
        print(label_counts)

        update_csv(labeled_df, file, backup_path)

    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        raise e