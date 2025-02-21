import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import List
import numpy as np
import seaborn as sns

plt.style.use(['science','no-latex'])
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.size'] = 14
rcParams['legend.fontsize'] = "medium"
rcParams['axes.grid'] = False
plt.tight_layout(pad=0.05)

def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], path: str = '', plot_mode:str = ''):
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

        if plot_mode == 'interactive':
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