import sys
sys.path.append('/root/libiq-101')
import libiq

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
import time
# Start the stopwatch
start_time = time.time()

input_file_path = str('/root/libiq-101/examples/iq_samples/iq_sample_captured.bin')

analyzer = libiq.Analyzer() 

fft = analyzer.fast_fourier_transform(input_file_path) 
c = 0
for i in fft:
    if c <= 1000:
        print(i)
        c += 1
    else:
        break

# Stop the stopwatch
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"The code took {elapsed_time} seconds to execute")
'''

'''
import time
# Start the stopwatch
start_time = time.time()

input_file_path = str('/root/libiq-101/examples/iq_samples/iq_sample_captured.bin')

analyzer = libiq.Analyzer() 

sample_rate = 1000.0
psd = analyzer.calculate_PSD(input_file_path, sample_rate)

c = 0
for i in psd:
    if c <= 1000:
        print(i)
        c += 1
    else:
        break

# Stop the stopwatch
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"The code took {elapsed_time} seconds to execute")
'''


'''
import time
# Start the stopwatch
start_time = time.time()

input_file_path = str('/root/libiq-101/examples/iq_samples/iq_sample_captured.bin')

analyzer = libiq.Analyzer() 
iq_sample = analyzer.get_iq_sample(input_file_path)
print(iq_sample)

# Stop the stopwatch
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"The code took {elapsed_time} seconds to execute")
'''


import src_python.spectrogram as sp
import time
# Start the stopwatch
start_time = time.time()

input_file_path = str('/root/demo/iq_samples/iq_sample_captured.bin')

analyzer = libiq.Analyzer() 
onverlap = 0
window_size = 256
sample_rate = 1000000
center_frequency = 1000000000
fft = analyzer.generate_IQ_Spectrogram(input_file_path, onverlap, window_size, sample_rate)
middle_time = time.time()
print(f"The code took {middle_time - start_time} seconds to read iq sample and to calculate fftw.")
sp.spectrogram(fft, sample_rate, center_frequency)

# Stop the stopwatch
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"The code took {elapsed_time} seconds to execute")


'''
import src_python.scatterplot as scplt
import time
# Start the stopwatch
start_time = time.time()

input_file_path = str('/root/libiq-101/examples/iq_samples/iq_sample_captured.bin')

analyzer = libiq.Analyzer() 
iq = analyzer.get_iq_sample(input_file_path)
scplt.scatterplot(iq)

# Stop the stopwatch
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"The code took {elapsed_time} seconds to execute")
'''