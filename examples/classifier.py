import os
import random

import libiq.plotter.waterfall as wf
from libiq.classifier.cnn import Classifier
from libiq.classifier.preprocessing import preprocess_data
from libiq.utils.constants import (
    CNN_MODEL_PATH,
    CSV_FILE_PATH,
    LABELS,
    PLOTS_PATH,
    REPORT_PATH,
)
from libiq.utils.create_dataset import create_dataset_from_bin
from libiq.utils.logger import logger


def build_datasets(captures_path, train_count, random_choice=False):
    subfolders = [
        d
        for d in os.listdir(captures_path)
        if os.path.isdir(os.path.join(captures_path, d))
    ]

    files = {}

    for folder_name in subfolders:
        folder_path = os.path.join(captures_path, folder_name)

        label = LABELS.get(folder_name, -1)

        all_files = [
            f
            for f in os.listdir(folder_path)
            if f.startswith("iqs") and f.endswith(".bin")
        ]

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
        random_choice=random_choice,
    )

    return files


def create_directories(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            logger.debug(f"Directory created successfully: {path}")


def main():
    directories = [
        CSV_FILE_PATH,
        REPORT_PATH,
        CNN_MODEL_PATH,
        PLOTS_PATH,
    ]

    create_directories(directories)

    files = None
    x_train = None
    x_test = None
    report = False
    plots = True
    interactive_plots = False
    test_percentage = 0.2
    n_files_to_load = 60
    n_files_from_single_capture = 1

    files = load_captures("docs/sample_data/train", n_files_to_load, False)
    files_test = load_captures("docs/sample_data/test", n_files_to_load, False)

    classification_model = Classifier(
        time_window=1,
        input_vector=1536,
        moving_avg_window=30,
        extraction_window=600,
        epochs=7,
        batch_size=32,
        plots=plots,
        interactive_plots=interactive_plots,
    )

    create_dataset_from_bin(
        files,
        n_files_from_single_capture,
        CSV_FILE_PATH,
        f"{CSV_FILE_PATH}combined_output.csv",
        classification_model.time_window,
        classification_model.input_vector,
        classification_model.moving_avg_window,
    )

    x_train, x_test, y_train, y_test = preprocess_data(
        f"{CSV_FILE_PATH}combined_output.csv",
        test_percentage,
        report=report,
        report_path=f"{REPORT_PATH}train.html",
    )

    logger.debug(x_train.shape)
    logger.debug(x_test.shape)

    if classification_model.plots:
        wf.plot_waterfall(
            f"{CSV_FILE_PATH}combined_output.csv",
            interactive_plots=classification_model.interactive_plots,
            fft_size=classification_model.extraction_window,
            path=f"{PLOTS_PATH}waterfall_train.pdf",
        )

    classification_model.cnn_train(x_train, y_train)
    x_train = None
    x_test = None

    classification_model.load_model(f"{CNN_MODEL_PATH}best_model.keras")

    create_dataset_from_bin(
        files_test,
        n_files_from_single_capture,
        CSV_FILE_PATH,
        f"{CSV_FILE_PATH}combined_output.csv",
        classification_model.time_window,
        classification_model.input_vector,
        classification_model.moving_avg_window,
    )

    test_percentage = 0.8

    x_train, x_test, y_train, y_test = preprocess_data(
        f"{CSV_FILE_PATH}combined_output.csv",
        test_percentage,
        report=report,
        report_path=f"{REPORT_PATH}test.html",
    )

    logger.debug(x_train.shape)
    logger.debug(x_test.shape)

    if classification_model.plots:
        wf.plot_waterfall(
            f"{CSV_FILE_PATH}combined_output.csv",
            interactive_plots=classification_model.interactive_plots,
            fft_size=classification_model.extraction_window,
            path=f"{PLOTS_PATH}waterfall_test.pdf",
        )

    classification_model.cnn_test(x_test, y_test)


if __name__ == "__main__":
    main()
