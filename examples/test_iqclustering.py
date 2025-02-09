import os
import random
import numpy as np
import libiq
from libiq.utils.constants import LABELS, PLOTS_PATH, OUTPUT_PATH_TO_LABEL, CNN_MODEL_PATH, CAPTURES_PATH, COMBINED_CSV_FILE_PATH_LABELED, COMBINED_CSV_FILE_PATH, CLUSTER_MODEL_PATH, ORIGINAL_COMBINED_CSV_FILE_PATH_TO_LABEL, ORIGINAL_COMBINED_CSV_FILE_PATH, FILES, TEST_SIZE, RANDOM_STATE, REPORT_PATH, MODE, NUM_FILES, N_INSTANTS, REPORTS, PLOTS, GRID_SEARCH, SCATTERPLOT_MAGNITUDE, SCATTERPLOT_TIME, FILES_TO_LABEL, OUTPUT_PATH, COMBINED_CSV_FILE_PATH, COMBINED_CSV_FILE_PATH_TO_LABEL
from libiq.utils.create_dataset import create_dataset_from_bin, create_dataset_from_csv
from libiq.iq_clustering.preprocessing import preprocess_data
from libiq.iq_clustering.clustering_models import k_mean_dba_magnitude_test_cross_validation, k_mean_dba, k_mean_dba_magnitude_train, k_mean_dba_magnitude_test, print_scatterplot_3d
from libiq.iq_clustering.signal_processing import noise_detector
import libiq.iq_clustering.cnn as custom_cnn
from libiq.iq_clustering.cnn import Classifier

def build_datasets(
    captures_path,
    train_count,
    random_choice=False
):

    subfolders = [
        d for d in os.listdir(captures_path)
        if os.path.isdir(os.path.join(captures_path, d))
    ]
    
    files = {}

    for folder_name in subfolders:
        folder_path = os.path.join(captures_path, folder_name)
        
        label = LABELS.get(folder_name, -1)
        
        all_files = [f for f in os.listdir(folder_path) if f.startswith("iqs") and f.endswith(".bin")]
        
        if random_choice:
            random.shuffle(all_files)
        else:
            all_files.sort()

        selected_for_files = all_files[:train_count]

        for f in selected_for_files:
            full_path = os.path.join(folder_path, f)
            files[full_path] = label

    return files



def load_captures():
    CAPTURES_PATH = "/home/user/Desktop/catture"
    TRAIN_COUNT = 800
    random_choice = False

    files = build_datasets(
        captures_path=CAPTURES_PATH,
        train_count=TRAIN_COUNT,
    )

    return files


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

    files = load_captures()

    classification_model = Classifier(timeseries_size=1536*1)

    for n_files in [1]:
        for n_samples in [classification_model.timeseries_size]:
            create_dataset_from_bin(files, n_files, OUTPUT_PATH, f"{COMBINED_CSV_FILE_PATH}combined_output.csv", n_samples)                

            x_train, x_test, y_train, y_test = preprocess_data(f"{COMBINED_CSV_FILE_PATH}combined_output.csv", TEST_SIZE)

            classification_model.cnn_train(x_train, y_train)

            classification_model.load_model(f"{CNN_MODEL_PATH}best_model.keras")
            
            classification_model.cnn_test(x_test, y_test)
            
if __name__ == '__main__':
    main() 