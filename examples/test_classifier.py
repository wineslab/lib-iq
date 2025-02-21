import os
import random
import numpy as np
import libiq
from libiq.utils.constants import LABELS, PLOTS_PATH, OUTPUT_PATH_TO_LABEL, CNN_MODEL_PATH, CAPTURES_PATH, COMBINED_CSV_FILE_PATH_LABELED, COMBINED_CSV_FILE_PATH, CLUSTER_MODEL_PATH, ORIGINAL_COMBINED_CSV_FILE_PATH_TO_LABEL, ORIGINAL_COMBINED_CSV_FILE_PATH, TEST_SIZE, RANDOM_STATE, REPORT_PATH, REPORTS, PLOTS, OUTPUT_PATH, COMBINED_CSV_FILE_PATH, COMBINED_CSV_FILE_PATH_TO_LABEL
from libiq.utils.create_dataset import create_dataset_from_bin, create_dataset_from_csv
from libiq.classifier.preprocessing import preprocess_data
import libiq.classifier.cnn as custom_cnn
from libiq.classifier.cnn import Classifier
import libiq.plotter.waterfall as wf

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



def load_captures(captures_path, train_count, random_choice):
    files = build_datasets(
        captures_path=captures_path,
        train_count=train_count,
        random_choice=random_choice
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
    x_train = None
    x_test = None

    files = load_captures("/home/user/Desktop/train", 3500, False)
    files_test = load_captures("/home/user/Desktop/test", 3500, False)

    extraction_window = 600#1536-110
    moving_avg_window = 30
    K = 5
    classification_model = Classifier(time_window=K, input_vector=extraction_window)

    #start training
    create_dataset_from_bin(files, 1, OUTPUT_PATH, f"{COMBINED_CSV_FILE_PATH}combined_output.csv", classification_model.time_window, classification_model.input_vector, moving_avg_window)                
    x_train, x_test, y_train, y_test = preprocess_data(f"{COMBINED_CSV_FILE_PATH}combined_output.csv", 0.0001)
    print(x_train.shape)
    print(x_test.shape)

    wf.plot_waterfall("/home/user/Desktop/libiq/examples/test_results/iq_samples/csv/combined_output.csv", plots_mode="", fft_size=extraction_window, path=f'{PLOTS_PATH}waterfall_train.pdf')

    classification_model.cnn_train(x_train, y_train)
    x_train = None
    x_test = None
    #end training

    #start test
    classification_model.load_model(f"{CNN_MODEL_PATH}best_model.keras")
    
    create_dataset_from_bin(files_test, 1, OUTPUT_PATH, f"{COMBINED_CSV_FILE_PATH}combined_output.csv", classification_model.time_window, classification_model.input_vector, moving_avg_window)                
    x_train, x_test, y_train, y_test = preprocess_data(f"{COMBINED_CSV_FILE_PATH}combined_output.csv", 0.99)
    print(x_train.shape)
    print(x_test.shape)
    wf.plot_waterfall("/home/user/Desktop/libiq/examples/test_results/iq_samples/csv/combined_output.csv", plots_mode="", fft_size=extraction_window, path=f'{PLOTS_PATH}waterfall_test.pdf')
    classification_model.cnn_test(x_test, y_test)
    #end test

if __name__ == '__main__':
    main() 