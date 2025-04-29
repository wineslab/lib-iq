import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import Counter
from typing import List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import scienceplots
from libiq.plotter.loss_curve import plot_loss_curve
from libiq.plotter.confusion_matrix import plot_confusion_matrix
from libiq.classifier.energy_detector import energy_detector

plt.style.use(['science', 'no-latex'])
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.size'] = 14
rcParams['legend.fontsize'] = "medium"
rcParams['axes.grid'] = False
plt.tight_layout(pad=0.05)

from libiq.utils.constants import (
    PLOTS_PATH, 
    CNN_MODEL_PATH, 
    PLOT_LABELS, 
    STATIC_LABELS,
    RANDOM_STATE,
    N_EPOCHS
)

os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# To force CPU-only execution, uncomment the line below
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


class Classifier:
    def __init__(self, time_window: int = 1, input_vector: int = 1536, moving_avg_window: int = 30, extraction_window: int = 600, model_path: str = None):
        self.time_window = time_window
        self.input_vector = input_vector
        self.moving_avg_window = moving_avg_window
        self.extraction_window = extraction_window
        self.max_window = 1536

        if input_vector != extraction_window:
            self.input_vector = extraction_window

        if self.time_window > 1:
            self.buffer = np.empty((0, 2))
        else:
            self.buffer = None

        if model_path is not None:
            self.model = keras.models.load_model(model_path)
        else:
            self.model = None

        self.last_prediction = None

        @tf.function
        def fast_predict(x):
            return self.model(x, training=False)
        self.fast_predict = fast_predict

    def load_model(self, model_path: str = None):
        if model_path is not None:
            self.model = keras.models.load_model(model_path)
        else:
            self.model = None

    def apply_energy_detector_to_data(self, iq_data) -> np.ndarray:
        if iq_data.ndim == 1:
            iq_data = iq_data.reshape(-1, 2)
        complex_data = iq_data[:, 0] + 1j * iq_data[:, 1]
        data_matrix = complex_data.reshape(self.time_window, self.max_window)
        updated_n_samples, cropped_data = energy_detector(
            data_matrix, 
            extraction_window=self.extraction_window, 
            moving_avg_window=self.moving_avg_window
        )
        return cropped_data

    def predict(self, iq_data):
        if self.buffer is not None:
            iq_data_arr = np.array(iq_data).reshape(-1, 2)
            self.buffer = np.concatenate((self.buffer, iq_data_arr), axis=0)

            if self.buffer.shape[0] < self.time_window * self.max_window:
                return self.last_prediction

            if self.buffer.shape[0] == self.time_window * self.max_window:
                data_to_predict = self.buffer
                self.buffer = np.empty((0, 2))
            else:
                data_to_predict = self.buffer[:self.time_window * self.max_window]
                self.buffer = self.buffer[self.time_window * self.max_window:]
            
            cropped_data = self.apply_energy_detector_to_data(data_to_predict)
            preprocessed_data = self.preprocessing(cropped_data)
            result = self.cnn_test_dapp(preprocessed_data)
            self.last_prediction = result
            return result

        else:
            processed_data = self.apply_energy_detector_to_data(iq_data)
            preprocessed_data = self.preprocessing(processed_data)
            result = self.cnn_test_dapp(preprocessed_data)
            self.last_prediction = result
            return result

    def cnn_metrics(self, y_true: List[int], y_pred: List[int], path: str = '') -> Tuple[float, float, float, float]:
        try:
            if len(y_true) != len(y_pred):
                raise ValueError("The lists y_true and y_pred must have the same length.")

            print("\nModel Metrics:")
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_true, y_pred)

            print("\tAccuracy:", acc)
            print("\tPrecision:", precision)
            print("\tRecall:", recall)
            print("\tF1 Score:", f1)

            plot_confusion_matrix(cm, PLOT_LABELS, path)

            return acc, precision, recall, f1
        except ValueError as e:
            print(f"Error calculating metrics: {e}")
            raise e

    def modify_file_path(self, file: str) -> str:
        try:
            file_path = Path(file)
            if not file_path.exists():
                raise ValueError("The file path does not exist.")
            new_file_path = file_path.with_name(file_path.stem + '_labeled' + file_path.suffix)
            return str(new_file_path)
        except (TypeError, ValueError) as e:
            raise e

    def make_model(self, num_classes: int, input_shape: tuple) -> keras.models.Model:
        try:
            if len(input_shape) == 0:
                raise ValueError("input_shape must be a non-empty tuple.")

            input_layer = keras.layers.Input(input_shape)

            conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.ReLU()(conv1)

            conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.ReLU()(conv2)

            conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.ReLU()(conv3)

            gap = keras.layers.GlobalAveragePooling1D()(conv3)

            output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

            return keras.models.Model(inputs=input_layer, outputs=output_layer)
        except (TypeError, ValueError) as e:
            raise e

    def cnn_train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = N_EPOCHS, batch_size: int = 32) -> None:
        print("\nStarting CNN model training...")
        try:
            if x_train is None or len(x_train) == 0:
                raise ValueError("The input time series is empty or None.")

            model = self.make_model(7, input_shape=(self.time_window * self.input_vector, 4))
            keras.utils.plot_model(model, show_shapes=True, to_file=f'{CNN_MODEL_PATH}model.pdf')

            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    f'{CNN_MODEL_PATH}best_model.keras', save_best_only=True, monitor="val_loss"
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
                ),
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
            ]

            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["sparse_categorical_accuracy"],
            )

            history = model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_split=0.2,
                verbose=1,
            )

            plot_loss_curve(history.history, path=f'{PLOTS_PATH}loss_curve_train.pdf')

            y_train_pred = np.argmax(model.predict(x_train), axis=1)
            self.cnn_metrics(y_train, y_train_pred, f'{PLOTS_PATH}confusion_matrix_train.pdf')

        except Exception as e:
            raise e

    def cnn_test(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        try:
            if self.model is None:
                raise ValueError("The model was not loaded correctly.")

            if x_test is None or len(x_test) == 0:
                raise ValueError("The input time series is empty or None.")

            y_pred = self.model.predict(x_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            self.cnn_metrics(y_test, y_pred_classes, f'{PLOTS_PATH}confusion_matrix_test.pdf')

        except Exception as e:
            raise e

    def cnn_test_dapp(self, x: np.ndarray) -> int:
        if self.model is None:
            raise ValueError("The model was not loaded correctly.")

        if x is None or len(x) == 0:
            raise ValueError("The input time series is empty or None.")

        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)
        elif x.ndim == 2:
            if x.shape[-1] != 4:
                raise ValueError(f"Expected 4 channels; received {x.shape[-1]}.")

        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)
        elif x.ndim == 3 and x.shape[0] != 1:
            x = x[0:1, :, :]


        x = tf.convert_to_tensor(x, dtype=tf.float32)
        predictions = self.fast_predict(x)

        if predictions.ndim == 2:
            y_pred_classes = np.argmax(predictions, axis=1)
        else:
            predictions = predictions.reshape(-1, predictions.shape[-1])
            y_pred_classes = np.argmax(predictions, axis=1)

        final_label = Counter(y_pred_classes).most_common(1)[0][0]
        return STATIC_LABELS[final_label]

    def preprocessing(self, iq_data):
        iq_array = np.array(iq_data).reshape(-1)
        real_vals = np.real(iq_array)
        imag_vals = np.imag(iq_array)
        
        mag_vals = np.sqrt(np.clip(real_vals**2 + imag_vals**2, a_min=0, a_max=None))
        eps = np.finfo(float).eps
        mag_dB = 20 * np.log10(np.maximum(mag_vals, eps))
        mag_dB[mag_vals == 0] = 0
        phase_vals = np.arctan2(imag_vals, real_vals)
        
        x = np.stack((real_vals, imag_vals, phase_vals, mag_dB), axis=1)
        return x