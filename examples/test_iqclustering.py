import os
import random
import numpy as np
import libiq
from libiq.utils.constants import LABELS, PLOTS_PATH, OUTPUT_PATH_TO_LABEL, CNN_MODEL_PATH, CAPTURES_PATH, COMBINED_CSV_FILE_PATH_LABELED, COMBINED_CSV_FILE_PATH, CLUSTER_MODEL_PATH, ORIGINAL_COMBINED_CSV_FILE_PATH_TO_LABEL, ORIGINAL_COMBINED_CSV_FILE_PATH, FILES, TEST_SIZE, RANDOM_STATE, REPORT_PATH, MODE, N_JOBS, NUM_FILES, N_INSTANTS, REPORTS, PLOTS, GRID_SEARCH, SCATTERPLOT_MAGNITUDE, SCATTERPLOT_TIME, FILES_TO_LABEL, OUTPUT_PATH, COMBINED_CSV_FILE_PATH, COMBINED_CSV_FILE_PATH_TO_LABEL
from libiq.utils.create_dataset import create_dataset_from_bin, create_dataset_from_csv
from libiq.iq_clustering.preprocessing import preprocess_data, convert_nested_lists_to_arrays
from libiq.iq_clustering.clustering_models import k_mean_dba_magnitude_test_cross_validation, k_mean_dba, k_mean_dba_magnitude_train, k_mean_dba_magnitude_test, print_scatterplot_3d
from libiq.iq_clustering.signal_processing import noise_detector
from libiq.iq_clustering.signal_processing import noise_detector
import libiq.iq_clustering.cnn as custom_cnn
from libiq.iq_clustering.cnn import Classifier # Importa la classe aggiornata

def build_datasets(
    captures_path,
    train_count,
    to_label_count,
    random_choice=False
):


    # Cartelle all’interno di captures_path
    subfolders = [
        d for d in os.listdir(captures_path)
        if os.path.isdir(os.path.join(captures_path, d))
    ]
    
    files = {}
    files_to_label = {}

    for folder_name in subfolders:
        folder_path = os.path.join(captures_path, folder_name)
        
        # Prendi la label intera dal mapping LABELS (default -1 se non esiste)
        label = LABELS.get(folder_name, -1)
        
        # Filtra i file che iniziano con "iqs"
        all_files = [f for f in os.listdir(folder_path) if f.startswith("iqs")]
        
        if random_choice:
            random.shuffle(all_files)
        else:
            all_files.sort()

        selected_for_files = all_files[:train_count]
        selected_for_files_to_label = all_files[train_count:train_count + to_label_count]

        # Aggiungi i file a `files` con label intera
        for f in selected_for_files:
            full_path = os.path.join(folder_path, f)
            files[full_path] = label
        
        # Aggiungi i file a `files_to_label` con label intera
        for f in selected_for_files_to_label:
            full_path = os.path.join(folder_path, f)
            files_to_label[full_path] = label

    return files, files_to_label



def load_captures():
    CAPTURES_PATH = "/home/user/Desktop/catture"  # Modifica in base al tuo percorso
    TRAIN_COUNT = 600
    TO_LABEL_COUNT = 200
    random_choice = False

    files, files_to_label = build_datasets(
        captures_path=CAPTURES_PATH,
        train_count=TRAIN_COUNT,
        to_label_count=TO_LABEL_COUNT,
        random_choice=random_choice
    )

    return files, files_to_label


def create_directories(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory created successfully: {path}")
        else:
            print(f"The directory already exists: {path}")

def main():

    directories = [
        CAPTURES_PATH,
        OUTPUT_PATH,
        REPORT_PATH,
        CNN_MODEL_PATH,
        OUTPUT_PATH_TO_LABEL,
        PLOTS_PATH,
    ]

    create_directories(directories)

    files = None
    files_to_label = None

    files, files_to_label = load_captures()

    classification_model = Classifier(timeseries_size=1536, model_path=f"{CNN_MODEL_PATH}best_model.keras")

    for n_files in [1]:
        for n_samples in [1536]:
            create_dataset_from_bin(files, n_files, OUTPUT_PATH, f"{COMBINED_CSV_FILE_PATH}combined_output.csv", n_samples)                

            x_train, x_test, y_train, y_test = preprocess_data(f"{COMBINED_CSV_FILE_PATH}combined_output.csv", TEST_SIZE, RANDOM_STATE, REPORT_PATH, mode=MODE, reports=REPORTS)

            classification_model.cnn_train(x_train, y_train)

            create_dataset_from_bin(files_to_label , 1, OUTPUT_PATH_TO_LABEL, f"{COMBINED_CSV_FILE_PATH_TO_LABEL}combined_output.csv", n_samples)
            
            classification_model.cnn_test(f"{COMBINED_CSV_FILE_PATH_TO_LABEL}combined_output.csv")
            
if __name__ == '__main__':
    main() 