from pathlib import Path

LIBRARY_PATH = Path("../").resolve()

CSV_FILE_PATH = f"{LIBRARY_PATH}/libiq/examples/test_results/iq_samples/"
REPORT_PATH = f"{LIBRARY_PATH}/libiq/examples/test_results/reports/"
CNN_MODEL_PATH = f"{LIBRARY_PATH}/libiq/examples/test_results/model/"
PLOTS_PATH = f"{LIBRARY_PATH}/libiq/examples/test_results/plots/"


LABELS = {
    "Undefined": -1,
    "No RFI": 0,
    "Jammer": 1,
    "Radar": 2,
    "Triangular": 3,
    "Square": 4,
    "LTE": 5,
}

PLOT_LABELS = ["No RFI", "Jammer", "Radar", "Triangular", "Square", "LTE"]

STATIC_LABELS = {
    0: "No RFI",
    1: "Jammer",
    2: "Radar",
    3: "Triangular",
    4: "Square",
    5: "LTE",
}

RANDOM_STATE = 11
