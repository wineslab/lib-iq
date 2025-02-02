import numpy as np
import pandas as pd
import os
import cmath
import multiprocessing
from typing import Generator, List, Dict, Tuple
from libiq.utils.constants import DTYPE, DATA_FORMAT, CHUNK_SIZE, N_JOBS, COLUMNS_LIST, DATA_FORMAT_OPTIONS

# =============================================================================
# 1) Funzione per ricavare i nomi delle colonne dal DATA_FORMAT
# =============================================================================
def columns_name(data_format: str) -> List[str]:
    if data_format not in COLUMNS_LIST:
        raise ValueError(f"Unknown value for data format: {data_format}")
    return COLUMNS_LIST[data_format]

# =============================================================================
# 2) Eliminazione dei file CSV pre-esistenti in una directory
# =============================================================================
def delete_csv_files(output_path: str) -> None:
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

# =============================================================================
# 3) Combina più file CSV in un unico file
# =============================================================================
def combine_csv(csv_files: List[str], combined_csv_file_path: str) -> None:
    print("Combining all .csv into one")
    try:
        for file_path in csv_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Legge il CSV a chunk
            chunk_container = pd.read_csv(file_path, chunksize=CHUNK_SIZE)
            for chunk in chunk_container:
                chunk_copy = chunk.copy()
                chunk_copy.insert(0, 'File', os.path.basename(file_path))
                chunk_copy.to_csv(
                    combined_csv_file_path,
                    mode='a',
                    header=not os.path.exists(combined_csv_file_path),
                    index=False
                )
        
        print(f"All CSV files have been combined into {combined_csv_file_path}.")
    
    except (FileNotFoundError, ValueError) as e:
        raise e
    except Exception as e:
        raise ValueError(f"Error processing file {file_path}: {e}")

# =============================================================================
# 4) Lettura dati binari (Approccio B: chunk di (re, im) interlacciati)
# =============================================================================
def read_data(file_path: str, dtype: np.dtype) -> Generator[np.ndarray, None, None]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file path does not exist: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            while True:
                data = np.fromfile(f, dtype=dtype, count=CHUNK_SIZE * 2)
                if data.size == 0:
                    break
                data = data.reshape(-1, 2)
                # Converte in numeri complessi (float32)
                complex_data = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)
                yield complex_data
    except Exception as e:
        raise ValueError(f"An error occurred while reading the binary data: {e}")

# =============================================================================
# 5) Processamento di un singolo campione complesso
# =============================================================================
def process_sample(complex_num: complex, data_format: str, ground_truth: int) -> List:
    if data_format == 'real-imag':
        return [complex_num.real, complex_num.imag, ground_truth]
    elif data_format == 'phase-magnitude':
        phase = cmath.phase(complex_num)
        magnitude = abs(complex_num)
        magnitude_dB = 20 * np.log10(magnitude) if magnitude > 0 else 0
        return [phase, magnitude_dB, ground_truth]
    elif data_format == 'all':
        phase = cmath.phase(complex_num)
        magnitude = abs(complex_num)
        magnitude_dB = 20 * np.log10(magnitude) if magnitude > 0 else 0
        return [complex_num.real, complex_num.imag, phase, magnitude_dB, ground_truth]
    elif data_format == 'complex':
        return [complex_num, ground_truth]
    else:
        raise ValueError(f"Unknown value for data format: {data_format}")

# =============================================================================
# 6) Gestione di un singolo file binario -> produce file CSV
# =============================================================================
def handle_message(file_path: str,
                   ground_truth: int,
                   n_samples: float,
                   output_path: str,
                   num_files: int,
                   fft_filter: Tuple[float, float] = (-float('inf'), float('inf'))
                   ) -> None:
    """
    Legge i dati binari a blocchi, applica eventualmente il filtro in dominio della frequenza,
    processa ciascun campione e scrive i risultati su più CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file path does not exist: {file_path}")

    counter = 0         # campioni accumulati nel CSV corrente
    file_counter = 1    # indice del CSV corrente
    message: List[List] = []

    def save_to_csv(msg: List[List], f_counter: int) -> None:
        if not msg:
            return
        file_name = os.path.join(output_path, f'{os.path.splitext(os.path.basename(file_path))[0]}_{f_counter}.csv')
        df = pd.DataFrame(msg, columns=columns_name(DATA_FORMAT))
        try:
            df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
        except Exception as e:
            raise ValueError(f"Error writing to CSV file {file_name}: {e}")

    try:
        total_samples_written = 0  # campioni totali processati

        for chunk_complex_data in read_data(file_path, DTYPE):
            # APPLICAZIONE DEL FILTRO IN DOMINIO DELLA FREQUENZA (se specificato)
            if fft_filter != (-float('inf'), float('inf')):
                freq_min, freq_max = fft_filter
                freq_min = int(freq_min)
                freq_max = int(freq_max)
                n = len(chunk_complex_data)
                # Assicuriamoci che gli indici siano nei limiti del chunk
                if freq_min < 0:
                    freq_min = 0
                if freq_max > n:
                    freq_max = n
                fft_data = np.fft.fft(chunk_complex_data)
                #codice subito sotto per azzerare i valori nel range specificato
                    #filtered_fft = np.zeros_like(fft_data, dtype=complex)
                    #filtered_fft[freq_min:freq_max] = fft_data[freq_min:freq_max]
                    #if freq_min > 0:
                    #    filtered_fft[-freq_max:-freq_min] = fft_data[-freq_max:-freq_min]
                    # Ricostruisco il segnale nel dominio del tempo
                    #chunk_complex_data = np.fft.ifft(filtered_fft)

                selected_positive = fft_data[freq_min:freq_max]
                selected_negative = fft_data[-freq_max:-freq_min]
                selected_bins = np.concatenate((selected_positive, selected_negative))
                new_length = selected_bins.size
                chunk_complex_data = np.fft.ifft(selected_bins, n=new_length)

            # Processa ciascun campione nel chunk (già filtrato, se richiesto)
            for c in chunk_complex_data:
                if file_counter > num_files:
                    break  # raggiunto il numero massimo di CSV

                csv_row = process_sample(c, DATA_FORMAT, ground_truth)
                message.append(csv_row)
                counter += 1
                total_samples_written += 1

                # Quando abbiamo accumulato CHUNK_SIZE righe, salviamo su CSV
                if counter == CHUNK_SIZE:
                    save_to_csv(message, file_counter)
                    message = []
                    counter = 0

                    if total_samples_written >= n_samples:
                        file_counter += 1

                if total_samples_written == n_samples:
                    save_to_csv(message, file_counter)
                    message = []
                    counter = 0
                    file_counter += 1
                    if file_counter > num_files:
                        break

            if file_counter > num_files:
                break

        # Salva eventuali righe rimanenti se non abbiamo raggiunto num_files
        if message and file_counter <= num_files:
            save_to_csv(message, file_counter)

    except Exception as e:
        raise ValueError(f"Error processing file {file_path}: {e}")

# =============================================================================
# 7) Creazione dataset da file binari
# =============================================================================
def create_dataset_from_bin(files: Dict[str, int],
                            num_files: int,
                            output_path: str,
                            combined_output_path: str,
                            n_samples: float = float('inf'),
                            fft_filter: Tuple[float, float] = (-float('inf'), float('inf'))
                            ) -> None:
    """
    Converte i file binari in CSV (fino a num_files per ciascun binario),
    applicando eventualmente un filtro in dominio della frequenza, e poi combina
    tutti i CSV in un unico file.
    """
    try:
        if DATA_FORMAT not in DATA_FORMAT_OPTIONS:
            raise ValueError("Data format not available")
        
        file_names = list(files.keys())
        ground_truths = list(files.values())

        # Elimina eventuali CSV pre-esistenti nella cartella di output
        delete_csv_files(output_path)

        # Esecuzione in parallelo di handle_message, propagando fft_filter
        with multiprocessing.Pool(processes=N_JOBS) as pool:
            pool.starmap(
                handle_message,
                zip(
                    file_names,
                    ground_truths,
                    [n_samples] * len(file_names),
                    [output_path] * len(file_names),
                    [num_files] * len(file_names),
                    [fft_filter] * len(file_names)  # Propagazione del filtro
                )
            )

        # Recupera tutti i file CSV generati
        csv_files = [
            os.path.join(output_path, f"{os.path.splitext(os.path.basename(file))[0]}_{i}.csv")
            for file in files
            for i in range(1, num_files + 1)
            if os.path.exists(os.path.join(output_path, f"{os.path.splitext(os.path.basename(file))[0]}_{i}.csv"))
        ]

        if not csv_files:
            raise FileNotFoundError("None of the specified CSV files were found.")

        combine_csv(csv_files, combined_output_path)
    
    except (ValueError, FileNotFoundError) as e:
        raise e
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")

# =============================================================================
# 8) Creazione dataset da file CSV esistenti (se serve)
# =============================================================================
def create_dataset_from_csv(files: Dict[str, int],
                            num_files: int,
                            output_path: str,
                            combined_csv_file_path: str) -> None:
    try:
        csv_files = [
            os.path.join(output_path, f"{os.path.splitext(os.path.basename(file))[0]}_{i}.csv")
            for file in files
            for i in range(1, num_files + 1)
            if os.path.exists(os.path.join(output_path, f"{os.path.splitext(os.path.basename(file))[0]}_{i}.csv"))
        ]

        if not csv_files:
            raise FileNotFoundError("None of the specified CSV files were found.")

        combine_csv(csv_files, combined_csv_file_path)
    
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
