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

class Analyzer {
public:
    Analyzer(){}

    /**
     * @brief Performs a Fast Fourier Transform (FFT) on IQ data from a specified file.
     *
     * This function reads IQ data from the file at the given path, performs an FFT on the data,
     * and returns the transformed data. The FFT is performed using the FFTW library.
     *
     * The function first checks if the file exists and if it has the correct extension (.iq or .bin).
     * If the file is valid, it reads the IQ data, performs the FFT, and returns the result.
     * If the file is not valid or an error occurs during processing, the function returns an empty vector.
     *
     * @param input_file_path The path to the file containing the IQ data.
     * @return A 2D vector containing the real and imaginary parts of the FFT output. Each inner vector
     *         represents a complex number, with the first element being the real part and the second element
     *         being the imaginary part. If an error occurs, an empty vector is returned.
     */
    std::vector<std::vector<double>> fast_fourier_transform(const std::string& input_file_path);

    /**
     * @brief Calculates the Power Spectral Density (PSD) of IQ data from a specified file.
     *
     * This function reads IQ data from the file at the given path, performs a Fast Fourier Transform (FFT) on the data,
     * calculates the PSD, and returns the PSD data. The PSD is calculated as the square of the magnitude of the FFT output,
     * normalized by the size of the FFT and the sample rate.
     *
     * The function first checks if the file exists and if it has the correct extension (.iq or .bin).
     * If the file is valid, it reads the IQ data, performs the FFT, calculates the PSD, and returns the result.
     * If the file is not valid or an error occurs during processing, the function returns an empty vector.
     *
     * @param input_file_path The path to the file containing the IQ data.
     * @param sampleRate The sample rate of the IQ data.
     * @return A vector containing the PSD data. Each element represents the power spectral density at a specific frequency.
     *         If an error occurs, an empty vector is returned.
     */
    std::vector<double> calculate_PSD(const std::string& input_file_path, double sampleRate);

    /**
    * @brief Generates an IQ spectrogram from a file.
    *
    * This method reads a set of IQ samples from a specified file and generates an IQ spectrogram.
    * It uses a specified window size and a certain degree of overlap between windows.
    * It computes the Fast Fourier Transform (FFT) for each window and returns the spectrogram as a two-dimensional vector.
    *
    * @param input_file_path The path of the file from which to read the IQ samples.
    * @param overlap The number of samples that overlap between two consecutive windows.
    * @param window_size The window size for the FFT.
    * @param sample_rate The sampling rate of the IQ samples.
    *
    * @return A two-dimensional vector representing the spectrogram. Each element of the vector represents a time window, and contains a vector of powers in dB for each frequency sample.
    */
    std::vector<std::vector<double>> generate_IQ_Spectrogram_from_file(const std::string& input_file_path, int overlap, int windowsize, double sample_rate);

    /**
    * @brief Generates a real-time IQ spectrogram.
    *
    * This method generates a real-time IQ spectrogram from a set of input IQ samples.
    * It uses a specified window size and a certain degree of overlap between windows.
    * It computes the Fast Fourier Transform (FFT) for each window and returns the spectrogram as a two-dimensional vector.
    *
    * @param iq_samples_input A two-dimensional vector of input IQ samples. Each IQ sample is a vector of two elements, representing the real and imaginary parts.
    * @param overlap The number of samples that overlap between two consecutive windows.
    * @param window_size The window size for the FFT.
    * @param sample_rate The sampling rate of the IQ samples.
    *
    * @return A two-dimensional vector representing the spectrogram. Each element of the vector represents a time window, and contains a vector of powers in dB for each frequency sample.
    */
    std::vector<std::vector<double>> generate_IQ_Spectrogram(std::vector<std::vector<double>> iq_samples_input, int overlap, int window_size, double sample_rate);

    /**
    * @brief Extracts the real part of IQ samples from a file.
    *
    * This method reads a set of IQ samples from a specified file and returns a vector containing only the real part of each sample.
    * It uses the `read_iq_sample` method to read the IQ samples from the file.
    *
    * @param input_file_path The path of the file from which to read the IQ samples.
    *
    * @return A vector of doubles containing the real part of each IQ sample.
    */
    std::vector<double> real_part_iq_sample(const std::string& input_file_path);

    /**
    * @brief Extracts the imaginary part of IQ samples from a file.
    *
    * This method reads a set of IQ samples from a specified file and returns a vector containing only the imaginary part of each sample.
    * It uses the `read_iq_sample` method to read the IQ samples from the file.
    *
    * @param input_file_path The path of the file from which to read the IQ samples.
    *
    * @return A vector of doubles containing the imaginary part of each IQ sample.
    */
    std::vector<double> complex_part_iq_sample(const std::string& input_file_path);

    /**
    * @brief Extracts the real and imaginary parts of IQ samples from a file.
    *
    * This method reads a set of IQ samples from a specified file and returns a two-dimensional vector containing the real and imaginary parts of each sample.
    * It uses the `read_iq_sample` method to read the IQ samples from the file.
    *
    * @param input_file_path The path of the file from which to read the IQ samples.
    *
    * @return A two-dimensional vector of doubles. Each inner vector contains two elements: the real part and the imaginary part of an IQ sample.
    */
    std::vector<std::vector<double>> get_iq_sample(const std::string& input_file_path);
};

#endif // ANALYZER_H