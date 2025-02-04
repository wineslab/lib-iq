import multiprocessing
import numpy as np

PLATFORM = 'local'

if PLATFORM == 'Colosseum':
    CAPTURES_PATH = '/iq_samples/'
    LIBRARY_PATH = '/root/'
elif PLATFORM == 'local':
    #CAPTURES_PATH = '/home/wines/spear-dApp/logs/'
    #LIBRARY_PATH = '/home/wines/'
    #CAPTURES_PATH = '/home/user/Desktop/iq_samples/'
    #LIBRARY_PATH = '/home/user/Desktop/'
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
#PLOTS_PATH = '/home/user/Desktop/Docker/libiq_and_iqclustering_container/libiq/examples/test_results/plots/'


LABELS = {
    #'Undefined': -1,
    #'5G': 0,
    #'WIFI': 1,
    #'Triangular': 2,
    #'Noise': 3
    'Undefined': -1,
    'No RFI': 0,
    'Jammer': 1,
    'Radar': 2
}

PLOT_LABELS = [
    #'5G',
    #'WIFI',
    #'Triangular',
    #'Noise'
    'Undefined',
    'No RFI',
    'Jammer',
    'Radar'
]

STATIC_LABELS = {
    #0: '5G',
    #1: 'WIFI',
    #2: 'Triangular',
    #3: 'Noise'
    0: 'No RFI',
    1: 'Jammer',
    2: 'Radar'
}

COLUMNS_LIST = {
    'complex': ['Complex', 'Labels'],
    'real-imag': ['Real', 'Imaginary', 'Labels'],
    'phase-magnitude': ['Phase', 'Magnitude', 'Labels'],
    'all': ['Real', 'Imaginary','Phase', 'Magnitude', 'Labels']
}

N_CLUSTERS = 4
N_LABELS = 4

DATA_FORMAT_OPTIONS = ['real-imag', 'phase-magnitude', 'all']
NORMALIZATION_TYPES = ['RobustScaler', 'MinMax', 'TimeSeriesScalerMeanVariance', None]  #da sistemare tutte perchè non vanno
NORMALIZATION_TYPE = NORMALIZATION_TYPES[3]

MODES = [None, 'Magnitude', 'all']

N_INSTANTS = float(10)
NUM_FILES = 2
MODE = MODES[2]
PLOTS = True
PLOTS_MODE = ''
PLOT_CONFUSION_MATRIX = True
REPORTS = False
GRID_SEARCH = False
SCATTERPLOT_MAGNITUDE = False
SCATTERPLOT_TIME = False

if MODE == MODES[0]:
    DATA_FORMAT = DATA_FORMAT_OPTIONS[0]
    COLUMNS = ['Real', 'Imaginary']
elif MODE == MODES[1]:
    DATA_FORMAT = DATA_FORMAT_OPTIONS[1]
    COLUMNS = ['Magnitude']
else:
    DATA_FORMAT = DATA_FORMAT_OPTIONS[2]
    COLUMNS = ['Real', 'Imaginary', 'Magnitude']

DTYPE = np.int16    #for spear dapp
#DTYPE = np.complex64
CHUNK_SIZE = 30000000
TEST_SIZE = 0.2
RANDOM_STATE = 5
N_JOBS = multiprocessing.cpu_count()

PLOT_FONT_FAMILY = 'DejaVu Sans'
PLOT_FIGURE_SIZE = (10, 7)

PLOT_TRAFFIC_FIGURE_SIZE = (12, 8)
PLOT_TRAFFIC_CMAP = 'tab10'

CLUSTERS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
METRICS = ['euclidean', 'softdtw', 'dtw']
GRIDSEARCH_INIT = ['k-means++']

FILES = {
    #f"{CAPTURES_PATH}Triangular/triangular_0.bin": LABELS['Triangular'],
    #f"{CAPTURES_PATH}Triangular/triangular_1.bin": LABELS['Triangular'],

    #f"{CAPTURES_PATH}5G/5G_0.bin": LABELS['5G'],
    #f"{CAPTURES_PATH}5G/5G_1.bin": LABELS['5G'],

    #f"{CAPTURES_PATH}Noise/noise_0.bin": LABELS['Noise'],
    #f"{CAPTURES_PATH}Noise/noise_1.bin": LABELS['Noise'],

    #f"{CAPTURES_PATH}WIFI/wifi_0.bin": LABELS['WIFI'],
    #f"{CAPTURES_PATH}WIFI/wifi_1.bin": LABELS['WIFI'],
}

FILES_TO_LABEL = {
    #f"{CAPTURES_PATH}Triangular/triangular_2.bin": LABELS['Triangular'],
    
    #f"{CAPTURES_PATH}5G/5G_2.bin": LABELS['5G'],
    
    #f"{CAPTURES_PATH}Noise/noise_2.bin": LABELS['Noise'],
    
    #f"{CAPTURES_PATH}WIFI/wifi_2.bin": LABELS['WIFI'],   
}