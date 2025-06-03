import libiq
import libiq.plotter.scatterplot as scplt
from libiq.utils.constants import PLOTS_PATH

input_file_path = "docs/sample_data/sample.bin"

analyzer = libiq.Analyzer()

data_type = libiq.IQDataType.INT16.value

diff = 100000  # max value = 2147483647
start = 0
end = start + diff

grid = False
data_formats = ["real-imag", "magnitude-phase"]
data_format = data_formats[0]

iq = analyzer.getIQSamples(input_file_path, start, end, data_type)

scplt.scatterplot(iq, data_format, grid, False, f"{PLOTS_PATH}scatterplot.pdf")
