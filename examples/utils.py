from pathlib import Path

import numpy as np

import libiq
from libiq.utils.logger import logger

LIBRARY_PATH = f"{Path('../').resolve()}/libiq"

input_file_path = f"{LIBRARY_PATH}/examples/test_results/iq_samples/combined_output.csv"

analyzer = libiq.Analyzer()

data_type = libiq.IQDataType.INT16.value

sample_rate = 1000000
center_frequency = 3619200000
diff = 10000  # max value = 2147483647
start = 0
end = start + diff

window_size = 1536
overlap = 128

fft = analyzer.fastFourierTransform(input_file_path, data_type)
logger.debug(f"FFT shape calculated with overload 1: {np.shape(fft)}")

iq = analyzer.getIQSamples(input_file_path, start, start + diff, data_type)
fft = analyzer.fastFourierTransform(iq)
logger.debug(f"FFT shape calculated with overload 2: {np.shape(fft)}")

fft = analyzer.fastFourierTransform(input_file_path, start, start + diff, data_type)
logger.debug(f"FFT shape calculated with overload 3: {np.shape(fft)}\n")

psd = analyzer.calculatePSD(input_file_path, sample_rate, data_type)
logger.debug(f"PSD shape calculated with overload 1: {np.shape(psd)}")

iq = analyzer.getIQSamples(input_file_path, start, start + diff, data_type)
psd = analyzer.calculatePSD(iq, sample_rate)
logger.debug(f"PSD shape calculated with overload 2: {np.shape(psd)}")

psd = analyzer.calculatePSD(input_file_path, start, start + diff, data_type)
logger.debug(f"PSD shape calculated with overload 3: {np.shape(psd)}\n")

iq_samples = analyzer.getIQSamples(input_file_path, data_type)
logger.debug(f"iq_samples shape extracted with overload 1: {np.shape(iq_samples)}")

iq_samples = analyzer.getIQSamples(input_file_path, start, end, data_type)
logger.debug(f"iq_samples shape extracted with overload 2: {np.shape(iq_samples)}")

iq_samples = analyzer.getIQSamples(input_file_path, data_type, ["Real", "Imaginary"])
logger.debug(f"iq_samples shape extracted with overload 3: {np.shape(iq_samples)}\n")

spectrogram = analyzer.generateIQSpectrogram(
    input_file_path, overlap, window_size, sample_rate, data_type
)
logger.debug(f"Spectrogram shape with overload 1: {np.shape(spectrogram)}")

iq = analyzer.getIQSamples(input_file_path, start, start + diff, data_type)
spectrogram_mem = analyzer.generateIQSpectrogram(
    iq, overlap, window_size, sample_rate
)
logger.debug(f"Spectrogram shape with overload 2: {np.shape(spectrogram_mem)}\n")

real_part = analyzer.realPartIQSamples(input_file_path, data_type)
logger.debug(f"Real part shape with overload 1: {np.shape(real_part)}")

real_mem = analyzer.realPartIQSamples(iq, 0, diff)
logger.debug(f"Real part shape with overload 2: {np.shape(real_mem)}\n")

imag_part = analyzer.imaginaryPartIQSamples(input_file_path, data_type)
logger.debug(f"Imaginary part shape with overload 1: {np.shape(imag_part)}")

imag_mem = analyzer.imaginaryPartIQSamples(iq, 0, diff)
logger.debug(f"Imaginary part shape with overload 2: {np.shape(imag_mem)}")
