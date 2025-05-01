import libiq
import libiq.plotter.scatterplot as scplt
from pathlib import Path

LIBRARY_PATH = f"{Path('../').resolve()}/libiq"

input_file_path = f'{LIBRARY_PATH}/examples/test_results/iq_samples/combined_output.csv'

analyzer = libiq.Analyzer()

data_type = libiq.IQDataType.INT16.value

diff = 10000000  #max value = 2147483647
start = 0
end = start + diff

grid = False
data_formats = ['real-imag', 'magnitude-phase']
data_format = data_formats[0]

iq = analyzer.get_iq_samples(input_file_path, start, end, data_type)

scplt.scatterplot(iq, data_format, grids=grid)