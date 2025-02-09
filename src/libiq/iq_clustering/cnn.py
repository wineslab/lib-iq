import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from libiq.utils.constants import (
    PLOTS_PATH, 
    PLOTS_MODE, 
    N_LABELS, 
    CNN_MODEL_PATH, 
    LABELS, 
    PLOT_LABELS, 
    PLOT_CONFUSION_MATRIX, 
    STATIC_LABELS,
    RANDOM_STATE,
    N_EPOCHS
)

os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
#to force the execution to CPU only decomment below line
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)



class Classifier:
    def __init__(self, timeseries_size: int = 1536, model_path: str = None):
        self.timeseries_size = timeseries_size
        if model_path is not None:
            self.model = keras.models.load_model(model_path)
        else:
            self.model = None

    def load_model(self, model_path: str = None):
        if model_path is not None:
            self.model = keras.models.load_model(model_path)
        else:
            self.model = None

    def predict(self, iq_data):
        preprocessed_data = self.preprocessing(iq_data)
        result = self.cnn_test_dapp(preprocessed_data, self.timeseries_size)
        return result

    def plot_confusion_matrix(self, cm: List[List[int]], class_names: List[str], path: str = ''):
        try:
            cm = np.array(cm)
            if cm.size == 0:
                raise ValueError("La matrice di confusione è vuota. Fornisci una matrice valida.")

            row_sums = cm.sum(axis=1, keepdims=True)
            cm_normalized = cm / row_sums
            cm_normalized = np.nan_to_num(cm_normalized)

            plt.figure(figsize=(10, 7))
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Real')
            plt.title('Normalized Confusion Matrix by Row')

            if PLOTS_MODE == 'interactive':
                plt.show()
            else:
                if path != '':
                    plt.savefig(path, format='pdf')
                    plt.close()
                else:
                    raise ValueError("Il path per salvare il plot è vuoto. Fornisci un path valido o imposta PLOTS_MODE a 'interactive'.")
        except ValueError as ve:
            print(f"Errore: {ve}")
        except Exception as e:
            print(f"Errore imprevisto: {e}")

    def plot_loss_curve(self, history: dict, path: str = ''):
        try:
            if not history or 'loss' not in history or 'val_loss' not in history:
                raise ValueError("Il dizionario history deve contenere le chiavi 'loss' e 'val_loss'.")

            epochs = range(1, len(history['loss']) + 1)

            plt.figure(figsize=(8, 6))
            plt.plot(epochs, history['loss'], 'bo-', label='Training Loss')
            plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
            plt.title("Training e Validation Loss")
            plt.xlabel("Epoche")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

            if PLOTS_MODE == 'interactive':
                plt.show()
            else:
                if path != '':
                    plt.savefig(path, format='pdf')
                    plt.close()
                else:
                    raise ValueError("Il path per salvare il plot è vuoto. Fornisci un path valido o imposta PLOTS_MODE a 'interactive'.")
        except ValueError as ve:
            print(f"Errore: {ve}")
        except Exception as e:
            print(f"Errore imprevisto: {e}")


    def cnn_metrics(self, y_true: List[int], y_pred: List[int], path: str = '') -> Tuple[float, float, float, float]:
        try:
            if len(y_true) != len(y_pred):
                raise ValueError("Le liste y_true e y_pred devono avere la stessa lunghezza.")

            print("\nLe metriche del modello sono:")
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_true, y_pred)

            print("\tAccuracy:", acc)
            print("\tPrecision:", precision)
            print("\tRecall:", recall)
            print("\tF1 Score:", f1)

            self.plot_confusion_matrix(cm, PLOT_LABELS, path)

            return acc, precision, recall, f1
        except ValueError as e:
            print(f"Errore nel calcolo delle metriche: {e}")
            raise e

    def modify_file_path(self, file: str) -> str:
        try:
            file_path = Path(file)
            if not file_path.exists():
                raise ValueError("Il path del file non esiste.")
            new_file_path = file_path.with_name(file_path.stem + '_labeled' + file_path.suffix)
            return str(new_file_path)
        except (TypeError, ValueError) as e:
            raise e

    def make_model(self, num_classes: int, input_shape: tuple) -> keras.models.Model:
        try:
            if len(input_shape) == 0:
                raise ValueError("input_shape deve essere una tupla non vuota.")

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
                raise ValueError("La time series di input è vuota o None.")

            model = self.make_model(N_LABELS, input_shape=(self.timeseries_size,4))
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

            self.plot_loss_curve(history.history, path=f'{PLOTS_PATH}loss_curve_train.pdf')

            y_train_pred = np.argmax(model.predict(x_train), axis=1)
            self.cnn_metrics(y_train, y_train_pred, f'{PLOTS_PATH}confusion_matrix_train.pdf')

        except Exception as e:
            raise e

    def cnn_test(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        try:

            if self.model is None:
                raise ValueError("The model was not loaded correctly.")

            if x_test is None or len(x_test) == 0:
                raise ValueError("La time series di input è vuota o None.")

            y_pred = self.model.predict(x_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            self.cnn_metrics(y_test, y_pred_classes, f'{PLOTS_PATH}confusion_matrix_test.pdf')

        except Exception as e:
            raise e

    def cnn_test_dapp(self, x: np.ndarray, timeseries_size: int) -> int:

        if self.model is None:
            raise ValueError("The model was not loaded correctly.")

        if x is None or len(x) == 0:
            raise ValueError("La time series di input è vuota o None.")

        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)
        elif x.ndim == 2:
            if x.shape[-1] != 4:
                raise ValueError(f"Ci dovrebbero essere 4 canali; ricevuti {x.shape[-1]}.")

        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)
        elif x.ndim == 3 and x.shape[0] != 1:
            x = x[0:1, :, :]

        predictions = self.model.predict(x, verbose=0)
        if predictions.ndim == 2:
            y_pred_classes = np.argmax(predictions, axis=1)
        else:
            predictions = predictions.reshape(-1, predictions.shape[-1])
            y_pred_classes = np.argmax(predictions, axis=1)

        final_label = Counter(y_pred_classes).most_common(1)[0][0]
        return STATIC_LABELS[final_label]

    def preprocessing(self, iq_data):
        iq_array = np.array(iq_data).reshape(-1, 2)
        real_vals = iq_array[:, 0]
        imag_vals = iq_array[:, 1]

        mag_vals = np.sqrt(np.clip(real_vals**2 + imag_vals**2, a_min=0, a_max=None))
        eps = np.finfo(float).eps
        mag_dB = 20 * np.log10(np.maximum(mag_vals, eps))
        mag_dB[mag_vals == 0] = 0

        phase_vals = np.arctan2(imag_vals, real_vals)
        x = np.stack((real_vals, imag_vals, phase_vals, mag_dB), axis=1).astype(float)
        return x