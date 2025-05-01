import libiq
import libiq.plotter.scatterplot as scplt
import libiq.plotter.spectrogram as sp
import libiq.plotter.waterfall as wf
from pathlib import Path

LIBRARY_PATH = f"{Path('../').resolve()}/libiq"

input_file_path = f'{LIBRARY_PATH}/examples/test_results/iq_samples/combined_output.csv'

analyzer = libiq.Analyzer()

data_type = libiq.IQDataType.INT16.value

onverlap = 0
window_size = 1536
sample_rate = 1000000
center_frequency = 1000000000

diff = 1000000 #max value = 2147483647
start = 0
end = start + diff
print(window_size)

iq = analyzer.get_iq_samples(input_file_path, data_type)
fft = analyzer.generate_IQ_Spectrogram(iq, onverlap, window_size, sample_rate)
sp.spectrogram(fft, sample_rate, center_frequency)