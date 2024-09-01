import sys

platform = 'local'
path = ''
if platform == 'Docker':
    path = '/home/user/libiq'
elif platform == 'Colosseum':
    path = '/root/libiq'
elif platform == 'local':
    path = '/home/user/Desktop/project/libiq'

sys.path.append(path)

import libiq
import src_python.scatterplot as scplt
import src_python.spectrogram as sp
import time

start_time = time.time()

if platform == 'Docker':
    capture_path = '/home/user/iq_samples'
elif platform == 'Colosseum':
    capture_path = '/iq_samples'
elif platform == 'local':
    capture_path = '/home/user/Desktop/project/iq_samples'

input_file_path = f'{capture_path}/WIFI/wifi_0.bin'
#input_file_path = f'{capture_path}/5G/5G_0.bin'
#input_file_path = f'{capture_path}/WIFI/wifi_0.bin'
#input_file_path = f'{capture_path}/Triangular/triangular_0.bin'

analyzer = libiq.Analyzer() 
data_type = libiq.IQDataType_FLOAT32

onverlap = 0
window_size = 256
sample_rate = 20000000
center_frequency = 1000000000

diff = 1000#10000000 #max value = 2147483647
start = 0
end = start + diff
print(window_size)
window_size_scatterplot = 100
interval_update_scatterplot = 10
data_formats = ['real-imag', 'magnitude-phase']
data_format = data_formats[1]
grid = False


'''
input_file_path1 = str(f'{capture_path}/WIFI/wifi_0.bin')
output_file_path1 = str(f'{capture_path}/WIFI/wifi_0.mat')
input_file_path2 = str(f'{capture_path}/WIFI/wifi_0.mat')
output_file_path2 = str(f'{capture_path}/WIFI/wifi_0.sigmf-meta')

converter = libiq.Converter()

converter.freq_lower_edge = 213456
converter.freq_upper_edge = 3456768
converter.sample_rate = 23456
converter.frequency = 567890
converter.global_index = 9999
converter.sample_start = 1
converter.hw = "superpc"
converter.version = "1.0.0"

converter.from_bin_to_mat(input_file_path1, output_file_path1)
converter.from_mat_to_sigmf(input_file_path2, output_file_path2)
'''


'''
fft = analyzer.fast_fourier_transform(input_file_path, data_type) 
print(len(fft))
'''


'''
iq = analyzer.get_iq_samples(input_file_path, data_type)
psd = analyzer.calculate_PSD(iq, sample_rate)
print(len(psd))
'''


'''
iq_sample = analyzer.get_iq_samples(input_file_path, data_type)
print(len(iq_sample))
'''

'''
iq = analyzer.get_iq_samples(input_file_path, start, end, data_type)
fft = analyzer.generate_IQ_Spectrogram(iq, onverlap, window_size, sample_rate)
middle_time = time.time()
print(f"The code took {middle_time - start_time} seconds to read iq sample and to calculate fftw.")
sp.spectrogram(fft, sample_rate, center_frequency)
'''

'''
iq = analyzer.get_iq_samples(input_file_path, start, end, data_type)
scplt.scatterplot(iq, data_format, grids=grid)
#scplt.animated_scatterplot(iq, data_format, interval=interval_update_scatterplot, window=window_size_scatterplot, grids=grid)
'''

end_time = time.time()
elapsed_time = end_time - start_time
print(f"The code took {elapsed_time} seconds to execute")