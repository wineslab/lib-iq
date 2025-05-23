import matplotlib

from libiq.utils.logger import logger

try:
    import tkinter
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.close()
except Exception as e:
    logger.warning(f"TkAgg not available or usable: {e}. Falling back to Agg.")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
from matplotlib import rcParams

try:
    import scienceplots
    plt.style.use(["science", "no-latex"])
except (ImportError, OSError) as e:
    logger.warning(f"Matplotlib style 'science' not found. Using default style. ({e})")
    plt.style.use("default")
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
rcParams["font.size"] = 14
rcParams["legend.fontsize"] = "medium"
rcParams["axes.grid"] = False


def plot_loss_curve(
    history: dict[str, list[float]], path: str = "", interactive_plots: bool = False
) -> None:
    """
    Plot the training and validation loss curves over epochs.

    This function takes the training history from a Keras model and plots the loss progression
    for both the training and validation sets. It either shows the plot or saves it to a file.

    Args:
        history (dict[str, list[float]]): Dictionary containing 'loss' and 'val_loss' values over epochs.
        path (str): File path to save the loss plot (PDF format). Required unless interactive_plots is 'interactive'.
        interactive_plots (str): If set to 'interactive', the plot will be displayed instead of saved.

    Raises:
        ValueError: If required keys are missing in the history or the path is empty when saving is expected.

    Returns:
        None
    """
    try:
        if not history or "loss" not in history or "val_loss" not in history:
            raise ValueError(
                "The history dictionary must contain the keys 'loss' and 'val_loss'."
            )

        epochs = range(1, len(history["loss"]) + 1)

        plt.figure(figsize=(8, 6))

        plt.plot(epochs, history["loss"], "bo-", label="Training Loss")

        plt.plot(epochs, history["val_loss"], "ro-", label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        if interactive_plots:
            plt.show()
        else:
            if path != "":
                plt.savefig(path, format="pdf")
                plt.close()
            else:
                raise ValueError(
                    "The path to save the plot is empty. Provide a valid path or set INTERACTIVE_PLOTS to 'interactive'."
                )
    except ValueError as ve:
        logger.error(f"Error: {ve}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
