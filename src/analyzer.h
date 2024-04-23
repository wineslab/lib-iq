#ifndef ANALYZER_H
#define ANALYZER_H

#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <cmath>
#include <fftw3.h>
#include <complex>
#include <Python.h>

class Analyzer {
public:
    Analyzer(){}
    std::vector<std::vector<double>> fast_fourier_transform(const std::string& input_file_path);
    std::vector<double> calculate_PSD(const std::string& input_file_path, double sampleRate);
    std::vector<std::vector<double>> generate_IQ_Spectrogram(const std::string& input_file_path, int overlap, int windowsize, double sample_rate);
    std::vector<std::vector<double>> generate_IQ_Spectrogram_live(std::vector<std::vector<double>> iq_samples_input, int overlap, int window_size, double sample_rate);
    std::vector<double> real_part_iq_sample(const std::string& input_file_path);
    std::vector<double> complex_part_iq_sample(const std::string& input_file_path);
    std::vector<std::vector<double>> get_iq_sample(const std::string& input_file_path);
};

#endif // ANALYZER_H