from pathlib import Path
from libiq.converter.mat import MATConverter
from libiq.converter.sigmf import SigMFConverter

LIBRARY_PATH = f"{Path('../').resolve()}/libiq"

file_path1 = f"{LIBRARY_PATH}/examples/test_results/iq_samples/combined_output.csv"
file_path2 = f"{LIBRARY_PATH}/examples/test_results/iq_samples/acombined_output.mat"
file_path3 = f"{LIBRARY_PATH}/examples/test_results/iq_samples/acombined_output.sigmf-meta"

mat_converter = MATConverter(
    freq_lower_edge=213456,
    freq_upper_edge=3456768,
    sample_rate=23456,
    frequency=567890,
    global_index=9999,
    sample_start=1,
    hw="superpc",
    version="1.0.0"
)

sigmf_converter = SigMFConverter(
    freq_lower_edge=213456,
    freq_upper_edge=3456768,
    sample_rate=23456,
    frequency=567890,
    global_index=9999,
    sample_start=1,
    hw="superpc",
    version="1.0.0"
)

mat_converter.convert_to_mat(str(file_path1), str(file_path2))
sigmf_converter.convert_to_sigmf(str(file_path2), str(file_path3))
