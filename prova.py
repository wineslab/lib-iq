import libiq
import os
import scipy as sc

path = '/root/libiq-101/iq_samples/uav1_6ft_burst1_001.bin'

result = libiq.Converter.from_bin_to_mat(path)
print(result)

#mat = sc.io.loadmat('/root/libiq-101/iq_samples_mat/uav1_6ft_burst1_001.mat')