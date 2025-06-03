import libiq
import libiq.plotter.spectrogram as sp
from libiq.utils.constants import PLOTS_PATH
from libiq.utils.logger import logger

input_file_path = "docs/sample_data/sample.bin"

analyzer = libiq.Analyzer()

data_type = libiq.IQDataType.INT16.value

onverlap = 100
window_size = 10240
sample_rate = 1000000
center_frequency = 1000000000

diff = 10000000  # max value = 2147483647
start = 0
end = start + diff
logger.debug(window_size)

iq = analyzer.getIQSamples(input_file_path, start, end, data_type)
fft = analyzer.generateIQSpectrogram(iq, onverlap, window_size, sample_rate)
sp.spectrogram(fft, sample_rate, center_frequency, False, f"{PLOTS_PATH}spectrogram.pdf")
