import os
# 0 = mostra tutti i log TF (default)
# 1 = nasconde i messaggi INFO
# 2 = nasconde anche i WARNING
# 3 = nasconde anche i messaggi di errore (sconsigliato)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow import keras

# Assicurati che queste costanti siano definite nel modulo specificato
from libiq.utils.constants import (
    PLOTS_PATH, 
    PLOTS_MODE, 
    N_LABELS, 
    CNN_MODEL_PATH, 
    LABELS, 
    PLOT_LABELS, 
    PLOT_CONFUSION_MATRIX, 
    STATIC_LABELS
)


class Classifier:
    def __init__(self, timeseries_size: int = 1536, model_path: str = CNN_MODEL_PATH):
        """
        Inizializza la classe con la lunghezza attesa della time series e il path del modello CNN addestrato.
        """
        self.timeseries_size = timeseries_size
        self.cnn_trained_model = model_path
        self.model = keras.models.load_model(model_path)

    def predict(self, iq_data):
        """
        Effettua la predizione su dati IQ grezzi:
          1. Applica il preprocessing per ottenere l'array con i 4 canali.
          2. Utilizza il metodo cnn_test_dapp per ottenere la classe predetta.
        
        Args:
            iq_data: Dati IQ in ingresso.
        
        Returns:
            La classe predetta (tramite STATIC_LABELS).
        """
        preprocessed_data = self.preprocessing(iq_data)
        result = self.cnn_test_dapp(preprocessed_data, self.timeseries_size)
        return result

    def plot_confusion_matrix(self, cm: List[List[int]], class_names: List[str], path: str = ''):
        """
        Plotta la matrice di confusione normalizzata per riga (ogni riga somma a 1).

        Args:
            cm (List[List[int]]): Matrice di confusione (2D list).
            class_names (List[str]): Lista dei nomi delle classi.
            path (str): Path dove salvare il plot (se vuoto, il plot viene mostrato interattivamente).
        """
        try:
            # Converte in array NumPy
            cm = np.array(cm)
            if cm.size == 0:
                raise ValueError("La matrice di confusione è vuota. Fornisci una matrice valida.")

            # Normalizzazione per riga
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_normalized = cm / row_sums
            cm_normalized = np.nan_to_num(cm_normalized)

            plt.figure(figsize=(10, 7))
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predetto')
            plt.ylabel('Reale')
            plt.title('Matrice di Confusione Normalizzata per Riga')

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
        """
        Calcola e stampa le metriche di classificazione (accuracy, precision, recall, F1 score) e plotta la matrice di confusione.

        Args:
            y_true (List[int]): Etichette reali.
            y_pred (List[int]): Etichette predette.
            path (str): Path dove salvare il plot della matrice di confusione.

        Returns:
            Tuple contenente (accuracy, precision, recall, F1 score).
        """
        try:
            if len(y_true) != len(y_pred):
                raise ValueError("Le liste y_true e y_pred devono avere la stessa lunghezza.")

            print("\nLe metriche del modello sono:")
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
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
        """
        Modifica un path di file aggiungendo '_labeled' prima dell'estensione.

        Args:
            file (str): Path originale del file.

        Returns:
            str: Nuovo path del file.
        """
        try:
            file_path = Path(file)
            if not file_path.exists():
                raise ValueError("Il path del file non esiste.")
            new_file_path = file_path.with_name(file_path.stem + '_labeled' + file_path.suffix)
            return str(new_file_path)
        except (TypeError, ValueError) as e:
            raise e

    def make_model(self, num_classes: int, input_shape: tuple) -> keras.models.Model:
        """
        Crea un modello CNN.

        Args:
            num_classes (int): Numero di classi in output.
            input_shape (tuple): Shape dei dati di input.

        Returns:
            keras.models.Model: Il modello CNN creato.
        """
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

    def cnn_train(self, x_train, y_train, epochs: int = 5, batch_size: int = 32) -> None:
        """
        Addestra il modello CNN sui dati di training.

        Se i dati sono un DataFrame, vengono cercate le colonne:
        'Real', 'Imaginary', 'Phase', 'Magnitude' oppure 'Values' (come fallback).
        """
        print("\nStarting CNN model training...")
        try:
            # Preparazione dei dati
            if isinstance(x_train, pd.DataFrame):
                required_cols = ['Real', 'Imaginary', 'Phase', 'Magnitude']
                if set(required_cols).issubset(x_train.columns):
                    x_train_array = np.stack([
                        np.stack((row['Real'], row['Imaginary'], row['Phase'], row['Magnitude']), axis=-1)
                        for idx, row in x_train.iterrows()
                    ])
                elif 'Values' in x_train.columns:
                    x_train_array = np.stack(x_train['Values'].values)
                else:
                    x_train_array = np.stack(x_train.iloc[:, 0].values)
            elif isinstance(x_train, pd.Series):
                x_train_array = np.stack(x_train.values)
            else:
                x_train_array = x_train

            x_train_array = x_train_array.astype(float)

            # Se l'array è 2D, aggiunge la dimensione canale
            if x_train_array.ndim == 2:
                x_train_array = np.expand_dims(x_train_array, axis=-1)

            y_train_array = np.array(y_train)

            # Creazione del modello (input shape preso da x_train_array)
            model = self.make_model(N_LABELS, input_shape=x_train_array.shape[1:])
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
                x_train_array,
                y_train_array,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_split=0.2,
                verbose=1,
            )

            # Valutazione sui dati di training
            y_train_pred = np.argmax(model.predict(x_train_array), axis=1)
            self.cnn_metrics(y_train_array, y_train_pred, f'{PLOTS_PATH}confusion_matrix_train.pdf')

        except Exception as e:
            raise e

    def cnn_test(self, file: str, n_samples: int = None) -> None:
        """
        Testa il modello CNN sui dati contenuti in un file CSV.

        Se il CSV contiene la colonna 'Values' viene usata direttamente;
        altrimenti, si raggruppano i dati per file per ottenere le time series.
        """
        try:
            print("\n")
            x = pd.read_csv(file)

            if 'Values' in x.columns:
                x_series = x['Values']
                x_array = np.stack(x_series.values)
            else:
                grouped = x.groupby('File').agg({
                    'Real': list,
                    'Imaginary': list,
                    'Phase': list,
                    'Magnitude': list,
                    'Labels': 'first'
                }).reset_index()
                x_array = np.stack([
                    np.stack((row['Real'], row['Imaginary'], row['Phase'], row['Magnitude']), axis=-1)
                    for index, row in grouped.iterrows()
                ])
                y_array = grouped['Labels'].to_numpy()

            if 'Values' in x.columns:
                y_grouped = x.groupby('File')['Labels'].first().reset_index()
                y_array = y_grouped['Labels'].to_numpy()

            print("x shape before adding channel:", x_array.shape)
            if x_array.ndim == 2:
                x_array = np.expand_dims(x_array, axis=-1)

            print("x shape:", x_array.shape)
            print("y shape:", y_array.shape)

            model = keras.models.load_model(f'{CNN_MODEL_PATH}best_model.keras')
            test_loss, test_acc = model.evaluate(x_array, y_array)
            print("Test accuracy:", test_acc)
            print("Test loss:", test_loss)

            y_pred = model.predict(x_array)
            y_pred_classes = np.argmax(y_pred, axis=1)
            self.cnn_metrics(y_array, y_pred_classes, f'{PLOTS_PATH}confusion_matrix_test.pdf')

        except Exception as e:
            raise e

    def cnn_test_dapp(self, x: np.ndarray, timeseries_size: int) -> int:
        """
        Carica un modello CNN, esegue la predizione su una singola time series `x` e restituisce l'etichetta finale.
        
        Si assume che in modalità "all" l'input x abbia forma (sequence_length, 4).
        
        Args:
            x (np.ndarray): Time series di input.
            timeseries_size (int): Lunghezza attesa della time series.
            model_path (str, optional): Path del modello salvato. Se None, viene usato self.cnn_trained_model.
        
        Returns:
            int: Etichetta finale (convertita tramite STATIC_LABELS).
        """

        if self.model is None:
            raise ValueError("The model was not loaded correctly.")

        if x is None or len(x) == 0:
            raise ValueError("La time series di input è vuota o None.")

        # Controlla la dimensionalità e aggiusta se necessario
        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)
        elif x.ndim == 2:
            if x.shape[-1] != 4:
                raise ValueError(f"Per mode 'all', ci si aspetta 4 canali; ricevuti {x.shape[-1]}.")

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

        # Se il modello restituisce più predizioni, si usa il majority voting
        final_label = Counter(y_pred_classes).most_common(1)[0][0]
        return STATIC_LABELS[final_label]

    def preprocessing(self, iq_data):
        """
        Preprocessa i dati IQ per ottenere una matrice con 4 colonne:
        Real, Imaginary, Phase e Magnitude in dB.
        
        Args:
            iq_data: Dati in input (es. lista o array) che vengono convertiti in array 2D.
            
        Returns:
            np.ndarray: Array di shape (n_samples, 4)
        """
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
