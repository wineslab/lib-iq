import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use(['science', 'no-latex'])
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.size'] = 14
rcParams['legend.fontsize'] = "medium"
rcParams['axes.grid'] = True
plt.tight_layout(pad=0.05)

def plot_accuracy_curve(history: dict, path: str = '', plots_mode: str = ''):
    try:
        if not history or 'sparse_categorical_accuracy' not in history or 'val_sparse_categorical_accuracy' not in history:
            raise ValueError("The history dictionary must contain the keys 'sparse_categorical_accuracy' and 'val_sparse_categorical_accuracy'.")

        epochs = range(1, len(history['sparse_categorical_accuracy']) + 1)

        plt.figure(figsize=(8, 6))
        # Plot training sparse_categorical_accuracy
        plt.plot(epochs, history['sparse_categorical_accuracy'], 'bo-', label='Training Accuracy')
        # Plot val_sparse_categorical_accuracy
        plt.plot(epochs, history['val_sparse_categorical_accuracy'], 'ro-', label='Validation Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        if plots_mode == 'interactive':
            plt.show()
        else:
            if path != '':
                plt.savefig(path, format='pdf')
                plt.close()
            else:
                raise ValueError("The path to save the plot is empty. Provide a valid path or set PLOTS_MODE to 'interactive'.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")
