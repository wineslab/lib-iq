import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import matplotlib
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from typing import List, Union, Tuple
import os
from libiq.utils.constants import NORMALIZATION_TYPE, MODE

def explode_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Explode specified columns in the DataFrame, transforming each list-like element in the specified
    columns into a separate row.

    Args:
        df (pd.DataFrame): Input DataFrame with columns to be exploded.
        cols (List[str]): List of column names to explode.

    Returns:
        pd.DataFrame: DataFrame with exploded columns.

    Raises:
        KeyError: If any specified columns do not exist in the DataFrame.
        ValueError: If an error occurs during column explosion.
    """

    try:
        for col in cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' does not exist in the DataFrame.")
            df = df.explode(col)
        return df
    except KeyError as e:
        raise e
    except Exception as e:
        raise ValueError(f"An error occurred while exploding columns: {e}")

def report(x_df: pd.DataFrame, y_df: pd.DataFrame, title: str, report_path: str, file_name: str, explode: bool = True, explorative: bool = True) -> None:
    """
    Generate and save an exploratory data analysis (EDA) report by merging and optionally exploding columns.

    Args:
        x_df (pd.DataFrame): Main DataFrame for analysis.
        y_df (pd.DataFrame): DataFrame with 'File' and 'Labels' columns for merging.
        title (str): Title of the report.
        report_path (str): Directory path to save the report.
        file_name (str): Name of the output report file.
        explode (bool, optional): Whether to explode the columns in the DataFrame. Default is True.
        explorative (bool, optional): Whether to enable exploratory settings in the report. Default is True.

    Raises:
        FileNotFoundError: If the specified report path does not exist.
        ValueError: If an error occurs during report generation.
    """

    try:
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"The specified report path does not exist: {report_path}")

        matplotlib.use('Agg')

        merged_df = pd.merge(pd.DataFrame(x_df), y_df[['File', 'Labels']], on='File', how='left').drop(columns=['File'])

        if explode:
            columns_to_explode = list(merged_df.columns)
            columns_to_explode.remove('Labels')

            if columns_to_explode:
                if len(columns_to_explode) > 1:
                    combined_df = merged_df[columns_to_explode].apply(lambda row: list(zip(*row)), axis=1).explode()
                    combined_df = pd.DataFrame(combined_df.tolist(), index=combined_df.index).rename(columns=lambda x: columns_to_explode[x])
                    exploded_df = merged_df.drop(columns=columns_to_explode).join(combined_df)
                else:
                    exploded_df = explode_columns(merged_df, columns_to_explode)
            else:
                exploded_df = merged_df
        else:
            exploded_df = merged_df

        profile = ProfileReport(exploded_df, title=title, explorative=explorative)
        profile.to_file(f"{report_path}{file_name}")

        print(f"Report saved as {report_path}{file_name}")

        matplotlib.use('TkAgg')
        plt.close('all')
    except Exception as e:
        raise ValueError(f"An error occurred while generating the report: {e}")

def load_csv(csv_file_path: str, columns: List[str]) -> pd.DataFrame:
    """
    Load a CSV file with the specified columns.

    Args:
        csv_file_path (str): The path to the CSV file.
        columns (List[str]): List of columns to load.

    Returns:
        Optional[pd.DataFrame]: The loaded DataFrame or None if an error occurs.

    Raises:
        FileNotFoundError: If the specified CSV file path does not exist.
        ValueError: If there is an error reading the CSV file.
    """

    try:
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"The specified file path does not exist: {csv_file_path}")

        x_df = pd.read_csv(csv_file_path, usecols=columns)
        return x_df
    except Exception as e:
        raise ValueError(f"An error occurred while loading the CSV file: {e}")

def convert_nested_lists_to_arrays(data: List[List[List[float]]]) -> np.ndarray:
    """
    Convert nested lists to numpy arrays.

    Args:
        data (List[List[List[float]]]): The input nested list of floats.

    Returns:
        np.ndarray: Numpy array converted from the nested lists.

    Raises:
        ValueError: If there is an error in converting the nested lists to numpy arrays.
    """

    try:
        return np.array([np.array([np.array(sublist, dtype=float) for sublist in row]) for row in data])
    except Exception as e:
        raise ValueError(f"An error occurred while converting nested lists to arrays: {e}")

def normalize_data(x_df: pd.DataFrame, columns: List[str], normalization_type: str) -> pd.DataFrame:
    """
    Normalize specified columns in a DataFrame using a specified normalization technique.

    Args:
        x_df (pd.DataFrame): Input DataFrame containing the data to normalize.
        columns (List[str]): List of columns to normalize.
        normalization_type (str): Type of normalization ('RobustScaler', 'MinMax', 'TimeSeriesScalerMeanVariance').

    Returns:
        pd.DataFrame: DataFrame with normalized columns.

    Raises:
        ValueError: If the normalization type is not recognized or if an error occurs during normalization.
    """

    try:
        if normalization_type is None:
            return x_df

        if normalization_type not in {'RobustScaler', 'MinMax', 'TimeSeriesScalerMeanVariance'}:
            raise ValueError(f"Unknown normalization type: {normalization_type}")

        if 'Real' in x_df.columns and 'Imaginary' in x_df.columns:
            return _normalize_complex(x_df, columns, normalization_type)
        elif 'Magnitude' in x_df.columns:
            return _normalize_magnitude(x_df, columns, normalization_type)
        else:
            raise ValueError(f"Columns {x_df.columns} not supported for normalization")
    except Exception as e:
        raise ValueError(f"An error occurred during normalization: {e}")

def _normalize_complex(x_df: pd.DataFrame, columns: List[str], normalization_type: str) -> pd.DataFrame:
    """
    Normalize complex data (Real and Imaginary parts) using the specified normalization type.

    Args:
        x_df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of columns to normalize.
        normalization_type (str): The type of normalization to apply.

    Returns:
        pd.DataFrame: DataFrame with normalized complex columns.

    Raises:
        ValueError: If an error occurs during normalization.
    """

    try:
        if normalization_type == 'RobustScaler':
            return _apply_robust_scaler(x_df, columns)
        elif normalization_type == 'MinMax':
            return _apply_minmax_scaler(x_df, columns)
        elif normalization_type == 'TimeSeriesScalerMeanVariance':
            return _apply_timeseries_scaler(x_df, columns)
    except Exception as e:
        raise ValueError(f"Error in _normalize_complex with {normalization_type}: {e}")

def _normalize_magnitude(x_df: pd.DataFrame, columns: List[str], normalization_type: str) -> pd.DataFrame:
    """
    Normalize magnitude data using the specified normalization type.

    Args:
        x_df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of columns to normalize.
        normalization_type (str): The type of normalization to apply.

    Returns:
        pd.DataFrame: DataFrame with normalized magnitude column.

    Raises:
        ValueError: If an error occurs during normalization.
    """

    try:
        if normalization_type == 'RobustScaler':
            return _apply_robust_scaler_magnitude(x_df, columns)
        elif normalization_type == 'MinMax':
            return _apply_minmax_scaler_magnitude(x_df, columns)
        elif normalization_type == 'TimeSeriesScalerMeanVariance':
            return _apply_timeseries_scaler_magnitude(x_df, columns)
    except Exception as e:
        raise ValueError(f"Error in _normalize_magnitude with {normalization_type}: {e}")

def _apply_robust_scaler(x_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply RobustScaler normalization to the Real and Imaginary parts of the data.

    Args:
        x_df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of columns to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized Real and Imaginary columns.

    Raises:
        ValueError: If the input DataFrame or columns are invalid.
    """

    try:
        normalized_real, normalized_imaginary = [], []
        for index, row in x_df.iterrows():
            real_part = np.nan_to_num(np.array(row[columns[0]]), nan=0.0)
            imaginary_part = np.nan_to_num(np.array(row[columns[1]]), nan=0.0)
            normalized_real.append(RobustScaler().fit_transform(real_part.reshape(-1, 1)).flatten().tolist())
            normalized_imaginary.append(RobustScaler().fit_transform(imaginary_part.reshape(-1, 1)).flatten().tolist())
        
        x_df[columns[0]] = normalized_real
        x_df[columns[1]] = normalized_imaginary
        return x_df
    
    except ValueError as e:
        raise e

def _apply_minmax_scaler(x_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply MinMaxScaler normalization to the Real and Imaginary parts of the data.

    Args:
        x_df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of columns to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized Real and Imaginary columns.
    
    Raises:
        TypeError: If x_df is not a DataFrame or if columns is not a list.
        ValueError: If the input DataFrame is empty or columns are not found.
    """

    try:
        if x_df.empty:
            raise ValueError("The input DataFrame is empty.")
        
        for column in columns:
            if column not in x_df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")

        normalized_real, normalized_imaginary = [], []
        
        for index, row in x_df.iterrows():
            real_part = np.nan_to_num(np.array(row[columns[0]]), nan=0.0)
            imaginary_part = np.nan_to_num(np.array(row[columns[1]]), nan=0.0)
            normalized_real.append(MinMaxScaler().fit_transform(real_part.reshape(-1, 1)).flatten().tolist())
            normalized_imaginary.append(MinMaxScaler().fit_transform(imaginary_part.reshape(-1, 1)).flatten().tolist())
        
        x_df[columns[0]] = normalized_real
        x_df[columns[1]] = normalized_imaginary
    
        return x_df
    except (TypeError, ValueError) as e:
        raise e
    except Exception as e:
        print(f"An error occurred during normalization: {e}")
        raise e
    
def _apply_timeseries_scaler(x_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply TimeSeriesScalerMeanVariance normalization to the Real and Imaginary parts of the data.

    Args:
        x_df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of columns to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized Real and Imaginary columns.
    
    Raises:
        TypeError: If x_df is not a DataFrame or if columns is not a list.
        ValueError: If the input DataFrame is empty or columns are not found.
    """

    try:
        if x_df.empty:
            raise ValueError("The input DataFrame is empty.")
        
        for column in columns:
            if column not in x_df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")

        combined_data = []
        for index, row in x_df.iterrows():
            real_part = np.nan_to_num(np.array(row[columns[0]]), nan=0.0)
            imaginary_part = np.nan_to_num(np.array(row[columns[1]]), nan=0.0)
            combined_data.append(np.array([real_part, imaginary_part]).T)
        
        normalized_data = TimeSeriesScalerMeanVariance(mu=0, std=1).fit_transform(np.array(combined_data))
        x_df[columns[0]] = normalized_data[:, :, 0].tolist()
        x_df[columns[1]] = normalized_data[:, :, 1].tolist()
    
        return x_df
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during normalization: {e}")
        raise e

def _apply_robust_scaler_magnitude(x_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply RobustScaler normalization to the Magnitude part of the data.

    Args:
        x_df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of columns to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized Magnitude column.
    
    Raises:
        TypeError: If x_df is not a DataFrame or if columns is not a list.
        ValueError: If the input DataFrame is empty or columns are not found.
    """

    try:
        if x_df.empty:
            raise ValueError("The input DataFrame is empty.")
        
        for column in columns:
            if column not in x_df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")

        normalized_magnitude = []
        for index, row in x_df.iterrows():
            magnitude_part = np.nan_to_num(np.array(row[columns[0]]), nan=0.0)
            normalized_magnitude.append(RobustScaler().fit_transform(magnitude_part.reshape(-1, 1)).flatten().tolist())
        
        x_df[columns[0]] = normalized_magnitude
    
        return x_df
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during normalization: {e}")
        raise e

def _apply_minmax_scaler_magnitude(x_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply MinMaxScaler normalization to the Magnitude part of the data.

    Args:
        x_df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of columns to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized Magnitude column.
    
    Raises:
        TypeError: If x_df is not a DataFrame or if columns is not a list.
        ValueError: If the input DataFrame is empty or columns are not found.
    """

    try:
        if x_df.empty:
            raise ValueError("The input DataFrame is empty.")
        
        for column in columns:
            if column not in x_df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")

        normalized_magnitude = []
        for index, row in x_df.iterrows():
            magnitude_part = np.nan_to_num(np.array(row[columns[0]]), nan=0.0)
            normalized_magnitude.append(MinMaxScaler().fit_transform(magnitude_part.reshape(-1, 1)).flatten().tolist())
        
        x_df[columns[0]] = normalized_magnitude
    
        return x_df
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during normalization: {e}")
        raise e

def _apply_timeseries_scaler_magnitude(x_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply TimeSeriesScalerMeanVariance normalization to the Magnitude part of the data.

    Args:
        x_df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of columns to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized Magnitude column.
    
    Raises:
        TypeError: If x_df is not a DataFrame or if columns is not a list.
        ValueError: If the input DataFrame is empty or columns are not found.
    """

    try:
        if x_df.empty:
            raise ValueError("The input DataFrame is empty.")
        
        for column in columns:
            if column not in x_df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")

        x_df['Magnitude'] = x_df['Magnitude'].apply(lambda x: np.array(x))
        scaler = TimeSeriesScalerMeanVariance(mu=0, std=1)
        normalized_array = scaler.fit_transform(list(x_df['Magnitude']))
        normalized_flattened = [arr.flatten() for arr in normalized_array]
        normalized_df = pd.DataFrame({
            'File': x_df['File'],
            'Magnitude': normalized_flattened
        })
    
        return normalized_df
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during normalization: {e}")
        raise e

def equalize_dataframe(x_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Equalize the length of sequences in the specified columns of a DataFrame by padding shorter sequences
    and group data based on the 'File' column, maintaining the order of occurrence.

    Args:
        x_df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of columns with sequences to equalize.

    Returns:
        pd.DataFrame: DataFrame with equalized sequence lengths in the specified columns and grouped by 'File'.
    
    Raises:
        TypeError: If x_df is not a DataFrame or if columns is not a list.
        ValueError: If the input DataFrame is empty or columns are not found.
    """

    try:
        if x_df.empty:
            raise ValueError("The input DataFrame is empty.")
        
        equalized_data = {}
        target_length = max(x_df[col].count() for col in columns)
        
        for col in columns:
            sequences = x_df[col].tolist()
            if len(sequences) < target_length:
                equalized_seq = np.pad(sequences, (0, target_length - len(sequences)), 'constant')
            else:
                equalized_seq = sequences
            equalized_data[col] = equalized_seq

        x_df = pd.DataFrame.from_dict(equalized_data)

        grouped_df = x_df

        if MODE == 'all':
            grouped_df = x_df.groupby('File').agg({
                'Real': list,
                'Imaginary': list,
                'Phase': list,
                'Magnitude': list
            }).reset_index()

        return grouped_df
    
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred while equalizing the DataFrame: {e}")
        raise e

def preprocess_time_series(x_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Preprocess time series data by concatenating or alternating 'Real' and 'Imaginary' columns.

    Args:
        x_df (pd.DataFrame): The input DataFrame containing 'File', 'Real', and 'Imaginary' columns.
        mode (str): The mode of preprocessing ('concatenate' or 'alternate').

    Returns:
        pd.DataFrame: A new DataFrame with 'File' and 'Values' columns.
    
    Raises:
        TypeError: If x_df is not a DataFrame or if mode is not a string.
        ValueError: If the mode is not 'concatenate' or 'alternate'.
    """

    def concatenate(row: pd.Series) -> np.ndarray:
        """
        Concatenate 'Real' and 'Imaginary' columns for a given row.

        Args:
            row (pd.Series): A row of the DataFrame.

        Returns:
            np.ndarray: Concatenated array of 'Real' and 'Imaginary' values.
        """
        return np.concatenate([row['Real'], row['Imaginary']])
    
    def alternate(row: pd.Series) -> np.ndarray:
        """
        Alternate 'Real' and 'Imaginary' columns for a given row.

        Args:
            row (pd.Series): A row of the DataFrame.

        Returns:
            np.ndarray: Alternated array of 'Real' and 'Imaginary' values.
        """
        return np.ravel(np.column_stack((row['Real'], row['Imaginary'])))
    
    try:
        file_names = x_df['File']

        if mode == 'concatenate':
            concatenated = x_df.apply(concatenate, axis=1)
        elif mode == 'alternate':
            concatenated = x_df.apply(alternate, axis=1)
        else:
            raise ValueError("Invalid mode. Choose 'concatenate' or 'alternate'.")
        
        result_df = pd.DataFrame({'File': file_names, 'Values': concatenated})
        
        return result_df
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        raise e

def validate_dataset(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    """
    Validate the dataset by checking for empty arrays and NaN values.

    Args:
        x_train (np.ndarray): Training data features.
        x_test (np.ndarray): Testing data features.
        y_train (np.ndarray): Training data labels.
        y_test (np.ndarray): Testing data labels.

    Raises:
        TypeError: If any of the inputs are not numpy arrays.
        ValueError: If the dataset is not valid.
    """

    try:        
        if len(y_train) <= 0 or len(y_test) <= 0:
            raise ValueError("Dataset not valid: Labels are empty.")
        if x_train.size == 0 or x_test.size == 0:
            raise ValueError("Dataset not valid: DataFrame is empty.")
        
        x_train_flat = np.array([item for sublist in x_train for item in sublist])
        x_test_flat = np.array([item for sublist in x_test for item in sublist])
        
        if np.any(np.isnan(x_train_flat)) or np.any(np.isnan(x_test_flat)):
            raise ValueError("Dataset not valid: DataFrame contains NaN values.")
    
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during validation: {e}")
        raise e

def aggregate_columns(x_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Aggregates columns in the DataFrame based on the 'File' column.

    Args:
        x_df (pd.DataFrame): The input DataFrame containing the data.
        columns (List[str]): The list of columns to be aggregated.

    Returns:
        pd.DataFrame: The aggregated DataFrame.
    
    Raises:
        TypeError: If x_df is not a DataFrame or if columns is not a list.
        ValueError: If any of the specified columns do not exist in the DataFrame.
    """

    try:        
        for col in columns:
            if col not in x_df.columns:
                raise ValueError(f"Column {col} does not exist in DataFrame")
        
        if len(columns) == 2:
            x_df = x_df.groupby('File').agg({columns[1]: list}).reset_index()
        elif len(columns) == 3:
            x_df = x_df.groupby('File').agg({columns[1]: list, columns[2]: list}).reset_index()
        elif len(columns) == 4:
            x_df = x_df.groupby('File').agg({columns[1]: list, columns[2]: list, columns[3]: list}).reset_index()
        
        return x_df
    
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during aggregation: {e}")
        raise e

def preprocess_data(file: str, test_size: float, random_state: Union[int, None], report_path: str, mode: Union[str, None] = None, reports: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """
    Preprocess the data from a CSV file, splitting it for training and testing.

    Args:
        file (str): Path to the CSV file containing the dataset.
        test_size (float): Proportion of the dataset for the test split.
        random_state (Union[int, None]): Random seed for shuffling data during split.
        report_path (str): Path to save the generated reports.
        mode (Union[str, None], optional): Processing mode ('Magnitude', 'all', or None for default). Default is None.
        reports (bool, optional): Whether to generate reports. Default is False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, list, list]: Training and test data, and their corresponding labels.

    Raises:
        TypeError: If the type of any argument is not as expected.
        ValueError: If an error occurs during preprocessing.
    """
    
    try:        
        extension = (file.split("/")[-1]).split(".")[-1]
        if extension != "csv":
            raise ValueError(f"Unexpected file extension, .csv required, got .{extension}")
        
        columns = []
        if mode is None:
            columns = ['File', 'Real', 'Imaginary']
        elif mode == 'Magnitude':
            columns = ['File', 'Magnitude']
        elif mode == 'all':
            columns = ['File', 'Real', 'Imaginary', 'Phase', 'Magnitude']
        else:
            raise ValueError(f"Mode {mode} not supported")

        x_df = load_csv(file, columns)
        if x_df is None:
            raise ValueError("Failed to load the CSV file for x_df.")
        
        y_df = load_csv(file, ['File', 'Labels'])
        if y_df is None:
            raise ValueError("Failed to load the CSV file for y_df.")

        y_df = y_df.groupby('File').agg({'Labels': 'first'}).reset_index()
        
        x_df = aggregate_columns(x_df, columns)
        x_df = x_df.sort_values(by='File').reset_index(drop=True)
        
        x_df = equalize_dataframe(x_df, columns)

        if reports:
            report(x_df, y_df, 'Pandas Profiling Report equalization', report_path, '1-report_equalization.html')
        
        columns_to_normalize = x_df.columns.difference(['File'])
        x_df = normalize_data(x_df, columns_to_normalize, NORMALIZATION_TYPE)

        if reports:
            report(x_df, y_df, 'Pandas Profiling Report normalization', report_path, '2-report_normalization.html')
        
        if mode is None:
            x_df = np.stack((x_df['Real'], x_df['Imaginary']), axis=-1)
            x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=test_size, random_state=random_state)
            x_train = convert_nested_lists_to_arrays(x_train)
            x_test = convert_nested_lists_to_arrays(x_test)
            x_train = np.transpose(x_train, (0, 2, 1))
            x_test = np.transpose(x_test, (0, 2, 1))
        elif mode == 'Magnitude':
            x_train, x_test, y_train, y_test = train_test_split(x_df['Magnitude'].tolist(), y_df, test_size=test_size, random_state=random_state)
            x_train = np.array(x_train)
            x_test = np.array(x_test)
        elif mode == 'all':
            x_train, x_test, y_train, y_test = train_test_split(x_df[['Real', 'Imaginary', 'Phase', 'Magnitude']], y_df, test_size=test_size, random_state=random_state)
        else:
            raise ValueError("Invalid mode. Choose None.")

        return x_train, x_test, y_train['Labels'].tolist(), y_test['Labels'].tolist()
    
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        raise e