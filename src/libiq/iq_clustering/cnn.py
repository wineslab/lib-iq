from tensorflow import keras
import numpy as np
import pandas as pd
from libiq.utils.constants import PLOTS_PATH, PLOTS_MODE, N_LABELS, CNN_MODEL_PATH, LABELS, PLOT_LABELS, PLOT_CONFUSION_MATRIX
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

def cnn_train(x_train: pd.DataFrame, y_train: pd.DataFrame, epochs: int = 5, batch_size: int = 32) -> None:
    """
    Train a CNN model using the provided training data.

    Args:
        x_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training labels.
        epochs (int, optional): Number of training epochs. Default is 5.
        batch_size (int, optional): Batch size for training. Default is 32.

    Returns:
        None

    Raises:
        ValueError: If x_train or y_train is None.
        TypeError: If x_train or y_train are not in the expected format.
    """

    print("\nStarting CNN model training...")

    try:
        if x_train is None or y_train is None:
            raise ValueError("Input data is None")
        
        x_train = x_train.apply(pd.Series.explode)

        y_train = np.repeat(y_train, x_train.groupby(level=0).size())

        x_train = x_train.to_numpy()
        
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_train = x_train.astype(float)

        model = make_model(N_LABELS, input_shape=x_train.shape[1:])

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

        y_train_pred = np.argmax(model.predict(x_train), axis=1)
        cnn_metrics(y_train, y_train_pred, f'{PLOTS_PATH}confusion_matrix_train.pdf')
        
    except (ValueError, TypeError) as e:
        raise e

def cnn_test(file: str, n_samples: int = None) -> None:
    """
    Test a CNN model using data from a specified file.

    Args:
        file (str): Path to the CSV file containing test data.
        n_samples (int, optional): Number of samples per voting window. Default is None.

    Returns:
        None

    Raises:
        ValueError: If the file path is invalid or if n_samples is less than or equal to zero.
        TypeError: If the input types are incorrect.
    """

    try:
        print("\n")
        x = pd.read_csv(file)
        if n_samples is None:
            n_samples = int(len(x['File']) / len(np.unique(x['File'])))
        if x is None:
            raise ValueError("Input data is None")
        
        x = x.apply(pd.Series.explode)
        
        y = np.array(x['Labels'])
        x = x.drop(columns=['File', 'Labels'])
        
        x = x.to_numpy()
        x = x.reshape((x.shape[0], x.shape[1], 1))
        x = x.astype(float)

        model = keras.models.load_model(f'{CNN_MODEL_PATH}best_model.keras')

        test_loss, test_acc = model.evaluate(x, y)

        print("Test accuracy", test_acc)
        print("Test loss", test_loss)

        y_pred = model.predict(x)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_pred_majority = label_data(file, y_pred_classes, n_samples)

        y_true = y[::n_samples]
        y_pred_majority = y_pred_majority[::n_samples]

        labels_count = pd.Series(y_true).value_counts()
        pred_count = pd.Series(y_pred_majority).value_counts()

        report = pd.DataFrame({'Actual': labels_count, 'Predicted': pred_count}).fillna(0)
        report = report.astype(int)

        report.index = report.index.map({v: k for k, v in LABELS.items()})

        cnn_metrics(y, y_pred_classes, f'{PLOTS_PATH}confusion_matrix_test.pdf')
        print("\n")
        print(report)
    
    except (ValueError, TypeError) as e:
        raise e