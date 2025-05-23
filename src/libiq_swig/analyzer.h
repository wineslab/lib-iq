#ifndef ANALYZER_H
#define ANALYZER_H

#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <stdexcept>
#include <iomanip>
#include <algorithm>

// ============================================================================
// Enum to specify the data type of IQ samples
// ============================================================================
enum class IQDataType {
    FLOAT32,
    FLOAT64,
    INT16
};

class Analyzer {
public:
    Analyzer() {}

    /**
     * @brief Performs a Fast Fourier Transform (FFT) on IQ data read from a file.
     *
     * @param input_file_path The path to the file containing IQ data.
     * @param data_type The data type of the IQ samples.
     * @return A 2D vector containing the real and imaginary parts of the FFT output.
     */
    std::vector<std::vector<double>> fastFourierTransform(const std::string& input_file_path, IQDataType data_type);

    /**
     * @brief Performs an FFT on provided IQ data.
     *
     * @param iq_samples A 2D vector containing IQ samples [real, imaginary].
     * @return A 2D vector containing the real and imaginary parts of the FFT output.
     */
    std::vector<std::vector<double>> fastFourierTransform(const std::vector<std::vector<double>>& iq_samples);

    /**
     * @brief Performs an FFT on IQ data read from a file within a specified sample range.
     *
     * @param input_file_path The path to the file containing IQ data.
     * @param start_sample The starting sample index.
     * @param end_sample The ending sample index (exclusive).
     * @param data_type The data type of the IQ samples.
     * @return A 2D vector containing the real and imaginary parts of the FFT output.
     */
    std::vector<std::vector<double>> fastFourierTransform(const std::string& input_file_path, int start_sample, int end_sample, IQDataType data_type);

    /**
     * @brief Calculates the Power Spectral Density (PSD) of IQ data read from a file within a specified range.
     *
     * @param input_file_path The path to the file containing IQ data.
     * @param start_sample The starting sample index.
     * @param end_sample The ending sample index (exclusive).
     * @param data_type The data type of the IQ samples.
     * @return A vector containing the PSD data.
     */
    std::vector<double> calculatePSD(const std::string& input_file_path, int start_sample, int end_sample, IQDataType data_type);

    /**
     * @brief Calculates the PSD of IQ data read from a file.
     *
     * @param input_file_path The path to the file containing IQ data.
     * @param sampleRate The sampling rate of the IQ data.
     * @param data_type The data type of the IQ samples.
     * @return A vector containing the PSD data.
     */
    std::vector<double> calculatePSD(const std::string& input_file_path, double sampleRate, IQDataType data_type);

    /**
     * @brief Calculates the PSD of provided IQ data.
     *
     * @param iq_samples A 2D vector containing IQ samples [real, imaginary].
     * @param sampleRate The sampling rate of the IQ data.
     * @return A vector containing the PSD data.
     */
    std::vector<double> calculatePSD(const std::vector<std::vector<double>>& iq_samples, double sampleRate);

    /**
     * @brief Generates an IQ spectrogram from a file.
     *
     * @param input_file_path The path to the file containing IQ data.
     * @param overlap The number of overlapping samples between consecutive windows.
     * @param window_size The window size for the FFT.
     * @param sample_rate The sampling rate of the IQ data.
     * @param data_type The data type of the IQ samples.
     * @return A 2D vector representing the spectrogram in dB.
     */
    std::vector<std::vector<double>> generateIQSpectrogram(const std::string& input_file_path, int overlap, int window_size, double sample_rate, IQDataType data_type);

    /**
     * @brief Generates a real-time IQ spectrogram from provided IQ data.
     *
     * @param iq_samples_input A 2D vector of IQ samples [real, imaginary].
     * @param overlap The number of overlapping samples between consecutive windows.
     * @param window_size The window size for the FFT.
     * @param sample_rate The sampling rate of the IQ data.
     * @return A 2D vector representing the spectrogram in dB.
     */
    std::vector<std::vector<double>> generateIQSpectrogram(const std::vector<std::vector<double>>& iq_samples_input, int overlap, int window_size, double sample_rate);

    /**
     * @brief Extracts the real part of IQ samples from a file.
     *
     * @param input_file_path The path to the file containing IQ data.
     * @param data_type The data type of the IQ samples.
     * @return A vector of doubles containing the real part.
     */
    std::vector<double> realPartIQSamples(const std::string& input_file_path, IQDataType data_type);

    /**
     * @brief Extracts the real part from provided IQ data within a specified range.
     *
     * @param iq_samples A 2D vector containing IQ samples [real, imaginary].
     * @param start_sample The starting sample index.
     * @param end_sample The ending sample index (exclusive).
     * @return A vector of doubles containing the real part.
     */
    std::vector<double> realPartIQSamples(const std::vector<std::vector<double>>& iq_samples, int start_sample, int end_sample);

    /**
     * @brief Extracts the imaginary part of IQ samples from a file.
     *
     * @param input_file_path The path to the file containing IQ data.
     * @param data_type The data type of the IQ samples.
     * @return A vector of doubles containing the imaginary part.
     */
    std::vector<double> imaginaryPartIQSamples(const std::string& input_file_path, IQDataType data_type);

    /**
     * @brief Extracts the imaginary part from provided IQ data within a specified range.
     *
     * @param iq_samples A 2D vector containing IQ samples [real, imaginary].
     * @param start_sample The starting sample index.
     * @param end_sample The ending sample index (exclusive).
     * @return A vector of doubles containing the imaginary part.
     */
    std::vector<double> imaginaryPartIQSamples(const std::vector<std::vector<double>>& iq_samples, int start_sample, int end_sample);

    /**
     * @brief Extracts IQ samples (real and imaginary parts) from a file.
     *
     * @param input_file_path The path to the file containing IQ data.
     * @param data_type The data type of the IQ samples.
     * @return A 2D vector with [real, imaginary].
     */
    std::vector<std::vector<double>> getIQSamples(const std::string& input_file_path, IQDataType data_type);

    /**
     * @brief Extracts IQ samples from a file within a specified range.
     *
     * @param input_file_path The path to the file containing IQ data.
     * @param start_sample The starting sample index.
     * @param end_sample The ending sample index (exclusive).
     * @param data_type The data type of the IQ samples.
     * @return A 2D vector with [real, imaginary].
     */
    std::vector<std::vector<double>> getIQSamples(const std::string& input_file_path, int start_sample, int end_sample, IQDataType data_type);

    /**
     * @brief Overload: Extracts IQ samples from a file (supports both .bin and .csv).
     *        For CSV files, the column names to extract are specified (default: "Real" and "Imaginary").
     *
     * @param input_file_path The path to the file containing IQ data.
     * @param data_type The data type of the IQ samples.
     * @param csv_columns A vector of strings with the column names to use (default: {"Real", "Imaginary"}).
     * @return A 2D vector with [real, imaginary].
     */
    std::vector<std::vector<double>> getIQSamples(const std::string& input_file_path, IQDataType data_type, const std::vector<std::string>& csv_columns);

private:
    /**
     * @brief Reads IQ samples from a file based on the specified data type.
     *        If the file is CSV, the columns specified in csv_columns are used.
     *        If csv_columns is not provided, the default is {"Real", "Imaginary"}.
     *
     * @param input_file_path The path to the file containing IQ data.
     * @param data_type The data type of the IQ samples.
     * @return A vector of std::complex<double> containing the IQ samples.
     */
    std::vector<std::complex<double>> readIQSamples(const std::string& input_file_path, IQDataType data_type);
    /**
     * @brief Reads IQ samples from a file based on the specified data type.
     *        If the file is CSV, the columns specified in csv_columns are used.
     *        If csv_columns is not provided, the default is {"Real", "Imaginary"}.
     *
     * @param input_file_path The path to the file containing IQ data.
     * @param data_type The data type of the IQ samples.
     * @param csv_columns A vector of strings with the column names (default: {"Real", "Imaginary"}).
     * @return A vector of std::complex<double> containing the IQ samples.
     */
    std::vector<std::complex<double>> readIQSamples(const std::string& input_file_path, IQDataType data_type, const std::vector<std::string>& csv_columns);

};

#endif // ANALYZER_H