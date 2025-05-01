import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Dict, List

plt.style.use(['science', 'no-latex'])
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.size'] = 14
rcParams['legend.fontsize'] = "medium"
rcParams['axes.grid'] = True
plt.tight_layout(pad=0.05)

def plot_accuracy_curve(history: dict[str, list[float]], path: str = '', interactive_plots: bool = False) -> None:
    """
    Plot the training and validation accuracy curves over epochs.

    This function takes the training history from a Keras model, plots the training and validation
    accuracy, and either shows the plot interactively or saves it to a file.

    Args:
        history (dict): Dictionary containing keys 'sparse_categorical_accuracy' and 'val_sparse_categorical_accuracy'.
        path (str): Path to save the output plot as a PDF. Required unless interactive_plots is set to 'interactive'.
        interactive_plots (bool): If True, the plot is shown on screen. Otherwise, it's saved to the given path.

    Raises:
        ValueError: If the history is missing required keys, or the path is empty when interactive_plots is not 'interactive'.

    Returns:
        None
    """
    try:
        if not history or 'sparse_categorical_accuracy' not in history or 'val_sparse_categorical_accuracy' not in history:
            raise ValueError("The history dictionary must contain the keys 'sparse_categorical_accuracy' and 'val_sparse_categorical_accuracy'.")

        epochs = range(1, len(history['sparse_categorical_accuracy']) + 1)

        plt.figure(figsize=(8, 6))

        plt.plot(epochs, history['sparse_categorical_accuracy'], 'bo-', label='Training Accuracy')

        plt.plot(epochs, history['val_sparse_categorical_accuracy'], 'ro-', label='Validation Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        if interactive_plots == True:
            plt.show()
        else:
            if path != '':
                plt.savefig(path, format='pdf')
                plt.close()
            else:
                raise ValueError("The path to save the plot is empty. Provide a valid path or set interactive_plots to True.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")
