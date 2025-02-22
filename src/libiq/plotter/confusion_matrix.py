import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import List
import numpy as np
import seaborn as sns

plt.style.use(['science', 'no-latex'])
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.size'] = 14
rcParams['legend.fontsize'] = "medium"
rcParams['axes.grid'] = False
plt.tight_layout(pad=0.05)

def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], path: str = '', plot_mode: str = ''):
    try:
        # Convert the confusion matrix to a NumPy array
        cm = np.array(cm)
        if cm.size == 0:
            raise ValueError("The confusion matrix is empty. Provide a valid matrix.")

        # Compute the sum of each row and normalize the confusion matrix
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = cm / row_sums
        cm_normalized = np.nan_to_num(cm_normalized)

        # Create a figure for the heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Real')

        # Show or save the plot depending on the plot_mode and path provided
        if plot_mode == 'interactive':
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
