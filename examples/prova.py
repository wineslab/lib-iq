import libiq
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