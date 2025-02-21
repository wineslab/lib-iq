import numpy as np

PLATFORM = 'local'

if PLATFORM == 'Colosseum':
    CAPTURES_PATH = '/iq_samples/'
    LIBRARY_PATH = '/root/'
elif PLATFORM == 'local':
    #CAPTURES_PATH = '/home/wines/spear-dApp/logs/'
    #LIBRARY_PATH = '/home/wines/'
    CAPTURES_PATH = '/home/user/Desktop/catture/'
    LIBRARY_PATH = '/home/user/Desktop/'
elif PLATFORM == 'Docker':
    CAPTURES_PATH = '/home/user/iq_samples/'
    LIBRARY_PATH = '/home/user/'
else:
    CAPTURES_PATH = '/home/user/iq_samples/'
    LIBRARY_PATH = '/home/user/'

COMBINED_CSV_FILE_PATH = f'{LIBRARY_PATH}libiq/examples/test_results/iq_samples/csv/'
ORIGINAL_COMBINED_CSV_FILE_PATH = f'{LIBRARY_PATH}libiq/examples/test_results/iq_samples/csv/'
OUTPUT_PATH = f'{LIBRARY_PATH}libiq/examples/test_results/iq_samples/csv/'
REPORT_PATH = f'{LIBRARY_PATH}libiq/examples/test_results/iq_samples/reports/'

CNN_MODEL_PATH = f'{LIBRARY_PATH}libiq/examples/test_results/models/' #saved_models/
CLUSTER_MODEL_PATH = f'{LIBRARY_PATH}libiq/examples/test_results/models/'
OUTPUT_PATH_TO_LABEL = f'{LIBRARY_PATH}libiq/examples/test_results/iq_samples/to_label/csv/'
COMBINED_CSV_FILE_PATH_TO_LABEL = f'{LIBRARY_PATH}libiq/examples/test_results/iq_samples/to_label/csv/'
ORIGINAL_COMBINED_CSV_FILE_PATH_TO_LABEL = f'{LIBRARY_PATH}libiq/examples/test_results/iq_samples/to_label/csv/'
COMBINED_CSV_FILE_PATH_LABELED = f'{LIBRARY_PATH}libiq/examples/test_results/iq_samples/to_label/csv/'

PLOTS_PATH = f'{LIBRARY_PATH}libiq/examples/test_results/plots/'


LABELS = {
    'Undefined': -1,
    'No RFI': 0,
    'Jammer': 1,
    'Radar': 2,
    'Triangular': 3,
    'Square': 4,
    'LTE': 5
}

PLOT_LABELS = [
    'No RFI',
    'Jammer',
    'Radar',
    'Triangular',
    'Square',
    'LTE'
]

STATIC_LABELS = {
    0: 'No RFI',
    1: 'Jammer',
    2: 'Radar',
    3: 'Triangular',
    4: 'Square',
    5: 'LTE'
}

PLOTS = True
PLOTS_MODE = ''
PLOT_CONFUSION_MATRIX = True
REPORTS = False

DTYPE = np.int16    #for spear dapp
#DTYPE = np.complex64

TEST_SIZE = 0.2
RANDOM_STATE = 11#8
N_EPOCHS = 10