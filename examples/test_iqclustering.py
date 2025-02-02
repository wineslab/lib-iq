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

def build_datasets(
    captures_path,
    train_count,
    to_label_count,
    random_choice=False
):
    """
    - captures_path: path che contiene le cartelle "Noise", "Triangular", "Sine", ecc.
    - train_count: quanti file usare per 'files'
    - to_label_count: quanti file usare per 'files_to_label'
    - random_choice: se True, pesca i file a caso; se False, in ordine alfabetico
    """

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

    '''
    print("Lunghezza files:", len(files))
    print("Lunghezza files_to_label:", len(files_to_label))
    # Esempio di ispezione di un elemento
    if files:
        sample_key = list(files.keys())[0]
        print("Esempio files:", sample_key, "->", files[sample_key])

    if files_to_label:
        sample_key_label = list(files_to_label.keys())[0]
        print("Esempio files_to_label:", sample_key_label, "->", files_to_label[sample_key_label])
    '''
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


    for n_files in [1]:
        for n_samples in [1536]:
            if files is None:
                create_dataset_from_bin(FILES, n_files, OUTPUT_PATH, f"{COMBINED_CSV_FILE_PATH}combined_output.csv", n_samples)
            else:
                create_dataset_from_bin(files, n_files, OUTPUT_PATH, f"{COMBINED_CSV_FILE_PATH}combined_output.csv", n_samples)                

            #noise_detector(f"{COMBINED_CSV_FILE_PATH}combined_output.csv", f"{ORIGINAL_COMBINED_CSV_FILE_PATH}combined_output_original.csv")

            x_train, x_test, y_train, y_test = preprocess_data(f"{COMBINED_CSV_FILE_PATH}combined_output.csv", TEST_SIZE, RANDOM_STATE, REPORT_PATH, mode=MODE, reports=REPORTS)

            custom_cnn.cnn_train(x_train, y_train)

            if files_to_label is None:
                create_dataset_from_bin(FILES_TO_LABEL , 1, OUTPUT_PATH_TO_LABEL, f"{COMBINED_CSV_FILE_PATH_TO_LABEL}combined_output.csv", n_samples)
            else:
                create_dataset_from_bin(files_to_label , 1, OUTPUT_PATH_TO_LABEL, f"{COMBINED_CSV_FILE_PATH_TO_LABEL}combined_output.csv", n_samples)
            
            #noise_detector(f"{COMBINED_CSV_FILE_PATH_TO_LABEL}combined_output.csv", f"{ORIGINAL_COMBINED_CSV_FILE_PATH_TO_LABEL}combined_output_original.csv")

            custom_cnn.cnn_test(f"{COMBINED_CSV_FILE_PATH_TO_LABEL}combined_output.csv")

            '''
            if MODE is None:
                pass #see if we want to remove this one because is useless and wrong
                #y_pred_train, y_pred_test = k_mean_dba(x_train, x_test, y_train, y_test, RANDOM_STATE, N_JOBS, grid_search=GRID_SEARCH, plots=PLOTS)
            elif MODE == 'Magnitude': #without noise detector because there are no real and imaginary columns
                pass
                y_pred_train = k_mean_dba_magnitude_train(x_train, y_train, CLUSTER_MODEL_PATH, RANDOM_STATE, N_JOBS, grid_search=GRID_SEARCH, plots=PLOTS)
                y_pred_test = k_mean_dba_magnitude_test(x_test, y_test, CLUSTER_MODEL_PATH, plots=PLOTS)
                # y_pred_train, y_pred_test = k_mean_dba_magnitude(x_train, x_test, y_train, y_test, RANDOM_STATE, N_JOBS, grid_search=GRID_SEARCH, plots=PLOTS)
            elif MODE == 'all':
                y_pred_train = None
                y_pred_test = None
                y_pred_train = k_mean_dba_magnitude_train(np.array(x_train['Magnitude'].tolist()), y_train, CLUSTER_MODEL_PATH, RANDOM_STATE, N_JOBS, grid_search=GRID_SEARCH, plots=PLOTS)
                y_pred_test = k_mean_dba_magnitude_test(np.array(x_test['Magnitude'].tolist()), y_test, CLUSTER_MODEL_PATH, plots=PLOTS)
                
                y_pred_test = k_mean_dba_magnitude_test_cross_validation(f"{COMBINED_CSV_FILE_PATH_LABELED}combined_output_labeled.csv", CLUSTER_MODEL_PATH, plots=PLOTS)

                if y_pred_train is not None and y_pred_test is not None and GRID_SEARCH == False:
                    if SCATTERPLOT_MAGNITUDE == True or SCATTERPLOT_TIME == True:
                        x_tr = np.stack((x_train['Real'], x_train['Imaginary']), axis=-1)
                        x_te = np.stack((x_test['Real'], x_test['Imaginary']), axis=-1)
                        x_tr = convert_nested_lists_to_arrays(x_tr)
                        x_te = convert_nested_lists_to_arrays(x_te)
                        x_tr = np.transpose(x_tr, (0, 2, 1))
                        x_te = np.transpose(x_te, (0, 2, 1))
                    if SCATTERPLOT_MAGNITUDE == True:
                        print_scatterplot_3d(x_te, y_test, x_magnitude = np.array(x_train['Magnitude'].tolist()))
                        print_scatterplot_3d(x_te, y_pred_test,x_magnitude = np.array(x_train['Magnitude'].tolist()))
                    if SCATTERPLOT_TIME == True:
                        print_scatterplot_3d(x_tr, y_pred_train)
                        print_scatterplot_3d(x_te, y_pred_test)
            else:
                raise ValueError(f"Mode {MODE} not aveable")
            '''

            
if __name__ == '__main__':
    main() 