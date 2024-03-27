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
# Inizia il cronometro
start_time = time.time()

input_file_path = str('/root/libiq-101/examples/iq_samples/uav1_6ft_burst1_001.bin')

analyzer = libiq.Analyzer() 

fft = analyzer.fast_fourier_transform(input_file_path) 

for i in range(10):
    print(fft[i])

# Ferma il cronometro
end_time = time.time()

# Calcola il tempo trascorso
elapsed_time = end_time - start_time

# Stampa il tempo trascorso
print(f"Il codice ha impiegato {elapsed_time} secondi per essere eseguito.")
'''

'''
import time
# Inizia il cronometro
start_time = time.time()

input_file_path = str('/root/libiq-101/examples/iq_samples/uav1_6ft_burst1_001.bin')

analyzer = libiq.Analyzer() 

sample_rate = 1000.0
psd = analyzer.calculatePSD(input_file_path, sample_rate) 

for i in range(10):
    print(psd[i])

# Ferma il cronometro
end_time = time.time()

# Calcola il tempo trascorso
elapsed_time = end_time - start_time

# Stampa il tempo trascorso
print(f"Il codice ha impiegato {elapsed_time} secondi per essere eseguito.")
'''