import libiq
import libiq.plotter.scatterplot as scplt
import libiq.plotter.spectrogram as sp
import libiq.plotter.waterfall as wf
import numpy as np
from pathlib import Path

LIBRARY_PATH = f"{Path('../').resolve()}/libiq"

input_file_path = f'{LIBRARY_PATH}/examples/test_results/iq_samples/combined_output.csv'

analyzer = libiq.Analyzer()

data_type = libiq.IQDataType.INT16.value

sample_rate = 1000000
center_frequency = 3619200000
diff = 10000 #max value = 2147483647
start = 0
end = start + diff

window_size = 1536
overlap = 128

fft = analyzer.fast_fourier_transform(input_file_path, data_type)
print(f"FFT shape calculated with overload 1: {np.shape(fft)}")

iq = analyzer.get_iq_samples(input_file_path, start, start+diff, data_type)
fft = analyzer.fast_fourier_transform(iq)
print(f"FFT shape calculated with overload 2: {np.shape(fft)}")

fft = analyzer.fast_fourier_transform(input_file_path, start, start + diff, data_type) 
print(f"FFT shape calculated with overload 3: {np.shape(fft)}\n")

psd = analyzer.calculate_PSD(input_file_path, sample_rate, data_type)
print(f"PSD shape calculated with overload 1: {np.shape(psd)}")

iq = analyzer.get_iq_samples(input_file_path, start, start + diff, data_type)
psd = analyzer.calculate_PSD(iq, sample_rate)
print(f"PSD shape calculated with overload 2: {np.shape(psd)}")

psd = analyzer.calculate_PSD(input_file_path, start, start + diff, data_type)
print(f"PSD shape calculated with overload 3: {np.shape(psd)}\n")

iq_samples = analyzer.get_iq_samples(input_file_path, data_type)
print(f"iq_samples shape extracted with overload 1: {np.shape(iq_samples)}")

iq_samples = analyzer.get_iq_samples(input_file_path, start, end, data_type)
print(f"iq_samples shape extracted with overload 2: {np.shape(iq_samples)}")

iq_samples = analyzer.get_iq_samples(input_file_path, data_type, ["Real", "Imaginary"])
print(f"iq_samples shape extracted with overload 3: {np.shape(iq_samples)}\n")

spectrogram = analyzer.generate_IQ_Spectrogram(input_file_path, overlap, window_size, sample_rate, data_type)
print(f"Spectrogram shape with overload 1: {np.shape(spectrogram)}")

iq = analyzer.get_iq_samples(input_file_path, start, start+diff, data_type)
spectrogram_mem = analyzer.generate_IQ_Spectrogram(iq, overlap, window_size, sample_rate)
print(f"Spectrogram shape with overload 2: {np.shape(spectrogram_mem)}\n")

real_part = analyzer.real_part_iq_samples(input_file_path, data_type)
print(f"Real part shape with overload 1: {np.shape(real_part)}")

real_mem = analyzer.real_part_iq_samples(iq, 0, diff)
print(f"Real part shape with overload 2: {np.shape(real_mem)}\n")

imag_part = analyzer.imaginary_part_iq_samples(input_file_path, data_type)
print(f"Imaginary part shape with overload 1: {np.shape(imag_part)}")

imag_mem = analyzer.imaginary_part_iq_samples(iq, 0, diff)
print(f"Imaginary part shape with overload 2: {np.shape(imag_mem)}")
