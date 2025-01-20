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

/**
 * @brief Enum to specify the data type of IQ samples.
 */
enum class IQDataType {
    FLOAT32,
    FLOAT64
};

class Analyzer {
public:
    Analyzer() {}

    /**
     * @brief Performs a Fast Fourier Transform (FFT) on IQ data from a specified file.
     *
     * @param input_file_path The path to the file containing the IQ data.
     * @param data_type The data type of the IQ samples (FLOAT32 or FLOAT64).
     * @return A 2D vector containing the real and imaginary parts of the FFT output.
     */
    std::vector<std::vector<double>> fast_fourier_transform(const std::string& input_file_path, IQDataType data_type);

    /**
     * @brief Performs a Fast Fourier Transform (FFT) on provided IQ data.
     *
     * @param iq_samples A 2D vector containing the IQ samples [real, imag].
     * @return A 2D vector containing the real and imaginary parts of the FFT output.
     */
    std::vector<std::vector<double>> fast_fourier_transform(const std::vector<std::vector<double>>& iq_samples);

    /**
    * @brief Performs a Fast Fourier Transform (FFT) on IQ data from a specified file within a range.
    *
    * @param input_file_path The path to the file containing the IQ data.
    * @param start_sample The starting sample index.
    * @param end_sample The ending sample index.
    * @param data_type The data type of the IQ samples (FLOAT32 or FLOAT64).
    * @return A 2D vector containing the real and imaginary parts of the FFT output.
    */
    std::vector<std::vector<double>> fast_fourier_transform(const std::string& input_file_path, int start_sample, int end_sample, IQDataType data_type);

    /**
    * @brief Calculates the Power Spectral Density (PSD) of IQ data from a specified file within a range.
    *
    * @param input_file_path The path to the file containing the IQ data.
    * @param start_sample The starting sample index.
    * @param end_sample The ending sample index.
    * @param data_type The data type of the IQ samples (FLOAT32 or FLOAT64).
    * @return A vector containing the PSD data.
    */
    std::vector<double> calculate_PSD(const std::string& input_file_path, int start_sample, int end_sample, IQDataType data_type);


    /**
     * @brief Calculates the Power Spectral Density (PSD) of IQ data from a specified file.
     *
     * @param input_file_path The path to the file containing the IQ data.
     * @param sampleRate The sample rate of the IQ data.
     * @param data_type The data type of the IQ samples (FLOAT32 or FLOAT64).
     * @return A vector containing the PSD data.
     */
    std::vector<double> calculate_PSD(const std::string& input_file_path, double sampleRate, IQDataType data_type);

    /**
     * @brief Calculates the Power Spectral Density (PSD) of provided IQ data.
     *
     * @param iq_samples A 2D vector containing the IQ samples [real, imag].
     * @param sampleRate The sample rate of the IQ data.
     * @return A vector containing the PSD data.
     */
    std::vector<double> calculate_PSD(const std::vector<std::vector<double>>& iq_samples, double sampleRate);

    /**
     * @brief Generates an IQ spectrogram from a file.
     *
     * @param input_file_path The path of the file from which to read the IQ samples.
     * @param overlap The number of samples that overlap between two consecutive windows.
     * @param window_size The window size for the FFT.
     * @param sample_rate The sampling rate of the IQ samples.
     * @param data_type The data type of the IQ samples (FLOAT32 or FLOAT64).
     * @return A two-dimensional vector representing the spectrogram in dB.
     */
    std::vector<std::vector<double>> generate_IQ_Spectrogram(const std::string& input_file_path, int overlap, int window_size, double sample_rate, IQDataType data_type);

    /**
     * @brief Generates a real-time IQ spectrogram from provided IQ data.
     *
     * @param iq_samples_input A 2D vector of input IQ samples [real, imag].
     * @param overlap The number of samples that overlap between two consecutive windows.
     * @param window_size The window size for the FFT.
     * @param sample_rate The sampling rate of the IQ samples.
     * @return A two-dimensional vector representing the spectrogram in dB.
     */
    std::vector<std::vector<double>> generate_IQ_Spectrogram(const std::vector<std::vector<double>>& iq_samples_input, int overlap, int window_size, double sample_rate);

    /**
     * @brief Extracts the real part of IQ samples from a file.
     *
     * @param input_file_path The path of the file from which to read the IQ samples.
     * @param data_type The data type of the IQ samples (FLOAT32 or FLOAT64).
     * @return A vector of doubles containing the real part.
     */
    std::vector<double> real_part_iq_sample(const std::string& input_file_path, IQDataType data_type);

    /**
     * @brief Extracts the real part of provided IQ samples within a specified range.
     *
     * @param iq_samples A 2D vector of IQ samples [real, imag].
     * @param start_sample The starting sample index.
     * @param end_sample The ending sample index.
     * @return A vector of doubles containing the real part in the specified range.
     */
    std::vector<double> real_part_iq_sample(const std::vector<std::vector<double>>& iq_samples, int start_sample, int end_sample);

    /**
     * @brief Extracts the imaginary part of IQ samples from a file.
     *
     * @param input_file_path The path of the file from which to read the IQ samples.
     * @param data_type The data type of the IQ samples (FLOAT32 or FLOAT64).
     * @return A vector of doubles containing the imaginary part.
     */
    std::vector<double> complex_part_iq_sample(const std::string& input_file_path, IQDataType data_type);

    /**
     * @brief Extracts the imaginary part of provided IQ samples within a specified range.
     *
     * @param iq_samples A 2D vector of IQ samples [real, imag].
     * @param start_sample The starting sample index.
     * @param end_sample The ending sample index.
     * @return A vector of doubles containing the imaginary part in the specified range.
     */
    std::vector<double> complex_part_iq_sample(const std::vector<std::vector<double>>& iq_samples, int start_sample, int end_sample);

    /**
     * @brief Extracts the real and imaginary parts of IQ samples from a file.
     *
     * @param input_file_path The path of the file from which to read the IQ samples.
     * @param data_type The data type of the IQ samples (FLOAT32 or FLOAT64).
     * @return A 2D vector of [real, imag].
     */
    std::vector<std::vector<double>> get_iq_samples(const std::string& input_file_path, IQDataType data_type);

    /**
     * @brief Extracts IQ samples from a file within a specified range.
     *
     * @param input_file_path The path to the file containing the IQ data.
     * @param start_sample The starting sample index.
     * @param end_sample The ending sample index.
     * @param data_type The data type of the IQ samples (FLOAT32 or FLOAT64).
     * @return A 2D vector of [real, imag].
     */
    std::vector<std::vector<double>> get_iq_samples(const std::string& input_file_path, int start_sample, int end_sample, IQDataType data_type);

    /**
     * @brief Extracts IQ samples from provided data within a specified range.
     *
     * @param iq_samples A 2D vector of IQ samples [real, imag].
     * @param start_sample The starting sample index.
     * @param end_sample The ending sample index.
     * @return A 2D vector di [real, imag].
     */
    std::vector<std::vector<double>> get_iq_samples(const std::vector<std::vector<double>>& iq_samples, int start_sample, int end_sample);

private:
    /**
     * @brief Reads IQ samples from a file based on the specified data type.
     *
     * @param input_file_path The path to the file containing the IQ data.
     * @param data_type The data type of the IQ samples (FLOAT32 or FLOAT64).
     * @return A vector of complex numbers representing the IQ samples.
     */
    std::vector<std::complex<double>> read_iq_samples(const std::string& input_file_path, IQDataType data_type);
};

#endif // ANALYZER_H
