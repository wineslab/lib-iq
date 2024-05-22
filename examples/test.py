import sys
sys.path.append('/home/user/Desktop/project/libiq')
import libiq
import src_python.scatterplot as scplt
import src_python.spectrogram as sp
import time

start_time = time.time()

input_file_path = '/home/user/Desktop/project/libiq/examples/iq_samples/iq_sample_captured2.bin'
analyzer = libiq.Analyzer() 
data_type = libiq.IQDataType_FLOAT32

onverlap = 0
window_size = 2**12
sample_rate = 1000000
center_frequency = 1000000000


'''
input_file_path1 = str('/root/libiq-101/examples/iq_samples/uav1_6ft_burst1_001.bin')
output_file_path1 = str('/root/libiq-101/examples/iq_samples_mat/uav1_6ft_burst1_001.mat')
input_file_path2 = str('/root/libiq-101/examples/iq_samples_mat/uav1_6ft_burst1_001.mat')
output_file_path2 = str('/root/libiq-101/examples/iq_samples_sigmf/uav1_6ft_burst1_001.sigmf-meta')

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
iq = analyzer.get_iq_samples(input_file_path, data_type)
fft = analyzer.generate_IQ_Spectrogram(iq, onverlap, window_size, sample_rate)
middle_time = time.time()
print(f"The code took {middle_time - start_time} seconds to read iq sample and to calculate fftw.")
sp.spectrogram(fft, sample_rate, center_frequency)
'''



diff = 30000
start = 0
end = start + diff
window = 100
interval = 1
iq = analyzer.get_iq_samples(input_file_path, start, end, data_type)
#scplt.scatterplot(iq, grids=True)
scplt.animated_scatterplot(iq, interval=interval, window=window, grids=True)



end_time = time.time()
elapsed_time = end_time - start_time
print(f"The code took {elapsed_time} seconds to execute")