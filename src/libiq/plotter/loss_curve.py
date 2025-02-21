import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use(['science','no-latex'])
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.size'] = 14
rcParams['legend.fontsize'] = "medium"
rcParams['axes.grid'] = False
plt.tight_layout(pad=0.05)

def plot_loss_curve(history: dict, path: str = '', plots_mode: str = ''):
        try:
            if not history or 'loss' not in history or 'val_loss' not in history:
                raise ValueError("Il dizionario history deve contenere le chiavi 'loss' e 'val_loss'.")

            epochs = range(1, len(history['loss']) + 1)

            plt.figure(figsize=(8, 6))
            plt.plot(epochs, history['loss'], 'bo-', label='Training Loss')
            plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()

            if plots_mode == 'interactive':
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