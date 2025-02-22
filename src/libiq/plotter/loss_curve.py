import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use(['science', 'no-latex'])
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.size'] = 14
rcParams['legend.fontsize'] = "medium"
rcParams['axes.grid'] = False
plt.tight_layout(pad=0.05)

def plot_loss_curve(history: dict, path: str = '', plots_mode: str = ''):
    try:
        # Check that the history dictionary contains both 'loss' and 'val_loss'
        if not history or 'loss' not in history or 'val_loss' not in history:
            raise ValueError("The history dictionary must contain the keys 'loss' and 'val_loss'.")

        epochs = range(1, len(history['loss']) + 1)

        plt.figure(figsize=(8, 6))
        # Plot training loss
        plt.plot(epochs, history['loss'], 'bo-', label='Training Loss')
        # Plot validation loss
        plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # If plots_mode is interactive, display the plot; otherwise, save it
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
