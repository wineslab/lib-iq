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

def plot_confusion_matrix(cm: list[list[int]], class_names: list[str], path: str = '', interactive_plots: bool = False) -> None:
    """
    Plot a normalized confusion matrix as a heatmap and either display or save it.

    The confusion matrix is normalized by row (i.e., by actual class), and NaNs are replaced with zeros.
    It is displayed as a heatmap with class labels on both axes.

    Args:
        cm (list[list[int]]): Confusion matrix (2D list of integers).
        class_names (list[str]): List of class labels for axis ticks.
        path (str): File path to save the heatmap as a PDF (used if interactive_plots is not 'interactive').
        interactive_plots (bool): If True, shows the plot using plt.show(); otherwise saves it to file.

    Raises:
        ValueError: If the confusion matrix is empty or path is empty when saving is required.

    Returns:
        None
    """
    try:
        cm = np.array(cm)
        if cm.size == 0:
            raise ValueError("The confusion matrix is empty. Provide a valid matrix.")

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = cm / row_sums
        cm_normalized = np.nan_to_num(cm_normalized)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Real')

        if interactive_plots == True:
            plt.show()
        else:
            if path != '':
                plt.savefig(path, format='pdf')
                plt.close()
            else:
                raise ValueError("The path to save the plot is empty. Provide a valid path or set interactive_plots to 'interactive'.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")
