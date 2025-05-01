import libiq
import libiq.plotter.scatterplot as scplt
import libiq.plotter.spectrogram as sp
import libiq.plotter.waterfall as wf

from pathlib import Path

LIBRARY_PATH = f"{Path('../').resolve()}/libiq"

file_path1 = f'{LIBRARY_PATH}/examples/test_results/iq_samples/combined_output.csv'
file_path2 = f'{LIBRARY_PATH}/examples/test_results/iq_samples/acombined_output.mat'
file_path3 = f'{LIBRARY_PATH}/examples/test_results/iq_samples/acombined_output.sigmf-meta'

converter = libiq.Converter()

converter.freq_lower_edge = 213456
converter.freq_upper_edge = 3456768
converter.sample_rate = 23456
converter.frequency = 567890
converter.global_index = 9999
converter.sample_start = 1
converter.hw = "superpc"
converter.version = "1.0.0"

converter.from_csv_or_bin_to_mat(file_path1, file_path2)
converter.from_mat_to_sigmf(file_path2, file_path3)