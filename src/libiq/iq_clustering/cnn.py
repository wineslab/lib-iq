import os

# 0 = mostra tutti i log TF (default)
# 1 = nasconde i messaggi INFO
# 2 = nasconde anche i WARNING
# 3 = nasconde anche i messaggi di errore (sconsigliato)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import numpy as np
import pandas as pd
from libiq.utils.constants import PLOTS_PATH, PLOTS_MODE, N_LABELS, CNN_MODEL_PATH, LABELS, PLOT_LABELS, PLOT_CONFUSION_MATRIX, STATIC_LABELS
import shutil
from typing import List, Tuple
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], path: str = ''):
    """
    Plots a confusion matrix normalized per row (class-level normalization), 
    ensuring that the values in each row represent proportions.

    Args:
        cm (List[List[int]]): Confusion matrix as a 2D list of integers.
        class_names (List[str]): List of class names corresponding to matrix indices.
        path (str): Path to save the plot (optional). If empty, the plot is only displayed interactively.
    """
    try:
        # Convert to numpy array for easier manipulation
        cm = np.array(cm)

        # Check if the confusion matrix is empty
        if cm.size == 0:
            raise ValueError("Confusion matrix is empty. Please provide a valid matrix.")

        # Normalization per row (proportion for each real class)
        row_sums = cm.sum(axis=1, keepdims=True)  # Sum for each row
        cm_normalized = cm / row_sums  # Normalize each row

        # Handle division by zero (if a row has no elements)
        cm_normalized = np.nan_to_num(cm_normalized)

        # Plot the normalized confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predetto')
        plt.ylabel('Reale')
        plt.title('Matrice di Confusione Normalizzata per Riga')
    
        # Save or display the plot
        if PLOTS_MODE == 'interactive':
            plt.show()
        else:
            if path != '':
                plt.savefig(path, format='pdf')
                plt.close()
            else:
                raise ValueError("Path to save the plot is empty. Provide a valid path or set PLOTS_MODE to 'interactive'.")

    except ValueError as ve:
        print(f"Errore: {ve}")
    except Exception as e:
        print(f"Errore imprevisto: {e}")

def cnn_metrics(y_true: List[int], y_pred: List[int], path: str = '') -> Tuple[float, float, float, float]:
    """
    Calculate and print classification metrics, including accuracy, precision, recall, and F1 score.
    Also, plot the confusion matrix.

    Args:
        y_true (List[int]): List of true labels.
        y_pred (List[int]): List of predicted labels.
        path (str, optional): Path to save the confusion matrix plot. Defaults to ''.

    Returns:
        Tuple[float, float, float, float]: Accuracy, precision, recall, and F1 score.

    Raises:
        ValueError: If the lengths of y_true and y_pred are not the same.
    """

    try:
        if len(y_true) != len(y_pred):
            raise ValueError("Length of y_true and y_pred must be the same.")
        
        print("\nThe metrics of the model are:")
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        
        print("\tAccuracy:", acc)
        print("\tPrecision:", precision)
        print("\tRecall:", recall)
        print("\tF1 Score:", f1)
        
        plot_confusion_matrix(cm, PLOT_LABELS, path)

        return acc, precision, recall, f1
    
    except ValueError as e:
        print(f"Error in metrics calculation: {e}")
        raise e

def majority_voting(labels: List[int], n_samples: int) -> List[int]:
    """
    Perform majority voting on a list of labels within specified windows.

    Args:
        labels (List[int]): List of labels.
        n_samples (int): Number of samples for each voting window.

    Returns:
        List[int]: List of labels resulting from the majority voting for each window.

    Raises:
        ValueError: If n_samples is less than or equal to zero.
        TypeError: If labels is not a list of integers or if n_samples is not an integer.
    """

    try:
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")
        
        majority_labels = []
        
        for i in range(0, len(labels), n_samples):
            window = labels[i:i + n_samples]
            
            label_counts = Counter(window).most_common()
            
            max_count = label_counts[0][1]
            tied_labels = [label for label, count in label_counts if count == max_count]
            
            if len(tied_labels) > 1:
                most_common_label = -1
            else:
                most_common_label = label_counts[0][0]
            
            majority_labels.append(most_common_label)
        
        return majority_labels
    
    except (TypeError, ValueError) as e:
        raise e

def modify_file_path(file: str) -> str:
    """
    Modify a file path by appending '_labeled' to the file name.

    Args:
        file (str): Original file path as a string.

    Returns:
        str: Modified file path as a string with '_labeled' added before the extension.

    Raises:
        TypeError: If the input file is not a string.
        ValueError: If the file path does not exist.
    """

    try:        
        file_path = Path(file)
        
        if not file_path.exists():
            raise ValueError("The file path does not exist.")
        
        new_file_path = file_path.with_name(file_path.stem + '_labeled' + file_path.suffix)
        
        return str(new_file_path)
    
    except (TypeError, ValueError) as e:
        raise e

def label_data(backup_file: str, predicted_labels: List[int], n_samples: int) -> List[int]:
    """
    Label data by adding a 'predicted_labels' column to a CSV file.

    Args:
        backup_file (str): Path to the CSV file.
        predicted_labels (List[int]): List of predicted labels.
        n_samples (int): Number of samples to consider for each voting window.

    Returns:
        List[int]: List of processed predicted labels after majority voting.

    Raises:
        TypeError: If the input types are incorrect.
        ValueError: If n_samples is less than or equal to zero or if the file path is invalid.
    """

    try:
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")
        
        file = modify_file_path(backup_file)
        shutil.copy(backup_file, file)
        x = pd.read_csv(file)
        
        predicted_labels = majority_voting(predicted_labels, n_samples)
        
        predicted_labels = np.repeat(predicted_labels, n_samples).tolist()

        x['predicted_labels'] = predicted_labels
        
        x.to_csv(file, index=False)

        print(f"File updated with 'predicted_labels' column saved to: {file}")
        return predicted_labels
    
    except (TypeError, ValueError) as e:
        raise e

def make_model(num_classes: int, input_shape: tuple) -> keras.models.Model:
    """
    Create a convolutional neural network (CNN) model.

    Args:
        num_classes (int): Number of output classes.
        input_shape (tuple): Shape of the input data.

    Returns:
        keras.models.Model: Keras CNN model instance.

    Raises:
        TypeError: If num_classes is not an integer or if input_shape is not a tuple.
        ValueError: If num_classes is less than or equal to zero or if input_shape is empty.
    """

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

def cnn_train(x_train, y_train, epochs: int = 10, batch_size: int = 32) -> None:
    """
    Addestra un modello CNN sui dati in cui ogni esempio è una time series completa.
    Se il mode è 'all', si assumono presenti le colonne:
    'Real', 'Imaginary', 'Phase' e 'Magnitude', che verranno combinate per formare
    un array di forma (n_samples, sequence_length, 4).
    """
    print("\nStarting CNN model training...")

    try:
        # Se x_train è un DataFrame, verifichiamo se sono presenti i quattro canali
        if isinstance(x_train, pd.DataFrame):
            required_cols = ['Real', 'Imaginary', 'Phase', 'Magnitude']
            if set(required_cols).issubset(x_train.columns):
                # Per ogni riga, creiamo un array di forma (sequence_length, 4)
                
                x_train_array = np.stack([
                    np.stack((row['Real'], row['Imaginary'], row['Phase'], row['Magnitude']), axis=-1)
                    for idx, row in x_train.iterrows()
                ])
            elif 'Values' in x_train.columns:
                # Se esiste la colonna 'Values', la usiamo
                x_train_array = np.stack(x_train['Values'].values)
            else:
                # Altrimenti, utilizziamo la prima colonna come fallback
                x_train_array = np.stack(x_train.iloc[:, 0].values)
        elif isinstance(x_train, pd.Series):
            x_train_array = np.stack(x_train.values)
        else:
            # Se x_train è già un array NumPy, lo usiamo direttamente
            x_train_array = x_train

        # Convertiamo in float
        x_train_array = x_train_array.astype(float)

        # Se non abbiamo già la dimensione canale (cioè, se l'array è 2D), la aggiungiamo.
        # Nel caso mode 'all' dovrebbe già avere shape (n_samples, sequence_length, 4)
        if x_train_array.ndim == 2:
            x_train_array = np.expand_dims(x_train_array, axis=-1)
        
        # Convertiamo le etichette in NumPy array
        y_train_array = np.array(y_train)

        # Creiamo il modello; l'input shape viene preso da x_train_array.shape[1:]
        model = make_model(N_LABELS, input_shape=x_train_array.shape[1:])
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

        y_train_pred = np.argmax(model.predict(x_train_array), axis=1)
        cnn_metrics(y_train_array, y_train_pred, f'{PLOTS_PATH}confusion_matrix_train.pdf')

    except Exception as e:
        raise e


def cnn_test(file: str, n_samples: int = None) -> None:
    """
    Testa il modello CNN sui dati in cui ogni esempio è una time series completa.
    Quando il mode è 'all', vengono usati tutti e quattro i canali:
    'Real', 'Imaginary', 'Phase' e 'Magnitude'.
    """
    try:
        print("\n")
        # Carica il CSV
        x = pd.read_csv(file)
        
        # Se il CSV contiene già una colonna 'Values', la usiamo (altrimenti siamo in mode diverso)
        if 'Values' in x.columns:
            x_series = x['Values']
            # Se necessario, converte le stringhe in liste (ad esempio con eval)
            # x_series = x_series.apply(eval)
            x_array = np.stack(x_series.values)
        else:
            # Raggruppa per 'File' per ottenere una time series per ogni file
            grouped = x.groupby('File').agg({
                'Real': list,
                'Imaginary': list,
                'Phase': list,
                'Magnitude': list,
                'Labels': 'first'  # Le etichette sono uguali per file
            }).reset_index()
            
            # Costruisci x_array combinando tutti e quattro i canali:
            x_array = np.stack([
                np.stack((row['Real'], row['Imaginary'], row['Phase'], row['Magnitude']), axis=-1)
                for index, row in grouped.iterrows()
            ])
            y_array = grouped['Labels'].to_numpy()
        
        # Se invece il CSV conteneva la colonna 'Values', raggruppiamo le etichette
        if 'Values' in x.columns:
            y_grouped = x.groupby('File')['Labels'].first().reset_index()
            y_array = y_grouped['Labels'].to_numpy()
        
        print("x shape before adding channel:", x_array.shape)
        # Nel caso mode 'all' l'array dovrebbe già avere shape (n_samples, sequence_length, 4)
        if x_array.ndim == 2:
            x_array = np.expand_dims(x_array, axis=-1)
        
        print("x shape:", x_array.shape)
        print("y shape:", y_array.shape)
        
        # Carica il modello
        model = keras.models.load_model(f'{CNN_MODEL_PATH}best_model.keras')
        
        test_loss, test_acc = model.evaluate(x_array, y_array)
        print("Test accuracy", test_acc)
        print("Test loss", test_loss)
        
        y_pred = model.predict(x_array)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        cnn_metrics(y_array, y_pred_classes, f'{PLOTS_PATH}confusion_matrix_test.pdf')
    
    except Exception as e:
        raise e


def cnn_test_dapp(x: np.ndarray, timeseries_size: int, model_path: str = CNN_MODEL_PATH) -> int:
    """
    Carica un modello CNN, esegue la predizione su un'unica time series `x` e restituisce 
    un'unica etichetta di output.
    
    Si assume che, in modalità 'all', l'input x debba avere forma (sequence_length, 4).
    Se necessario, vengono effettuati crop/padding e aggiunta della dimensione batch.
    
    Args:
        x (np.ndarray): Time series di ingresso.
        timeseries_size (int): Lunghezza attesa della time series (es. 400).
        model_path (str): Path del modello salvato (.keras, .h5, ecc.).
    
    Returns:
        int: Etichetta finale assegnata alla time series.
    """
    # Verifica input
    if x is None or len(x) == 0:
        raise ValueError("La time series di input è vuota o None.")
    
    # Se x non ha ancora la dimensione canale, controlliamo
    if x.ndim == 1:
        # Converti in 2D (sequence_length, 1)
        x = np.expand_dims(x, axis=-1)
    elif x.ndim == 2:
        # Se x ha 2 dimensioni, controlla se il numero di canali è quello atteso.
        # Per mode 'all', ci aspettiamo 4 canali.
        if x.shape[-1] != 4:
            raise ValueError(f"Per mode 'all', ci si aspetta 4 canali; ricevuti {x.shape[-1]}.")
    
    # Aggiungi la dimensione batch se necessario (da (sequence_length, 4) a (1, sequence_length, 4))
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    elif x.ndim == 3 and x.shape[0] != 1:
        # Se ci sono più di un esempio, prendi il primo
        x = x[0:1, :, :]
    
    # Carica il modello
    model = keras.models.load_model(model_path)
    
    predictions = model.predict(x, verbose=0)
    if predictions.ndim == 2:
        y_pred_classes = np.argmax(predictions, axis=1)
    else:
        predictions = predictions.reshape(-1, predictions.shape[-1])
        y_pred_classes = np.argmax(predictions, axis=1)
    
    # In questo caso, se il modello restituisce più predizioni (ad esempio per ogni timestep),
    # si applica un majority voting; qui, scegliamo la classe più frequente.
    final_label = Counter(y_pred_classes).most_common(1)[0][0]
    
    return STATIC_LABELS[final_label]