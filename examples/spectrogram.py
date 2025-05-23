from pathlib import Path

import libiq
import libiq.plotter.spectrogram as sp
from libiq.utils.constants import (
    CNN_MODEL_PATH,
    CSV_FILE_PATH,
    LABELS,
    PLOTS_PATH,
    REPORT_PATH,
)
from libiq.utils.logger import logger

LIBRARY_PATH = f"{Path('../').resolve()}/libiq"

input_file_path = f"{LIBRARY_PATH}/examples/test_results/iq_samples/combined_output.csv"

analyzer = libiq.Analyzer()

data_type = libiq.IQDataType.INT16.value

onverlap = 300
window_size = 1536
sample_rate = 1000000
center_frequency = 1000000000

diff = 1000000  # max value = 2147483647
start = 0
end = start + diff
logger.debug(window_size)

iq = analyzer.getIQSamples(input_file_path, start, end, data_type)
fft = analyzer.generateIQSpectrogram(iq, onverlap, window_size, sample_rate)
sp.spectrogram(fft, sample_rate, center_frequency, False, f"{PLOTS_PATH}spectrogram.pdf")
