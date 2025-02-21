#include "analyzer.h"

// ============================================================================
// Template function to read IQ sample blocks from binary files (.iq or .bin)
// ============================================================================
template <typename T>
std::vector<std::complex<double>> read_iq_sample_block(const std::string& input_file_path) {
    std::filesystem::path input_filepath = input_file_path;
    std::vector<std::complex<double>> iq_samples;

    if (!std::filesystem::exists(input_filepath)) {
        std::cerr << "Error: File does not exist: " << input_filepath << std::endl;
        return iq_samples;
    }

    if (input_filepath.extension() != ".iq" && input_filepath.extension() != ".bin") {
        std::cerr << "Error: Invalid file extension. Required: .iq or .bin" << std::endl;
        return iq_samples;
    }

    std::uintmax_t file_size = std::filesystem::file_size(input_filepath);
    if (file_size % (2 * sizeof(T)) != 0) {
        std::cerr << "Error: File size is not aligned with the expected data type size." << std::endl;
        return iq_samples;
    }

    std::size_t num_complex_samples = file_size / (2 * sizeof(T));
    iq_samples.resize(num_complex_samples);

    std::ifstream file(input_file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: File cannot be opened: " << input_filepath << std::endl;
        iq_samples.clear();
        return iq_samples;
    }

    std::vector<T> buffer(num_complex_samples * 2);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(T));

    if (static_cast<std::size_t>(file.gcount()) != buffer.size() * sizeof(T)) {
        std::cerr << "Error: Unexpected file read size. Possibly corrupted file or read error." << std::endl;
        iq_samples.clear();
        return iq_samples;
    }

    for (std::size_t i = 0; i < num_complex_samples; ++i) {
        iq_samples[i] = std::complex<double>(
            static_cast<double>(buffer[2 * i]),
            static_cast<double>(buffer[2 * i + 1])
        );
    }
    return iq_samples;
}

// ============================================================================
// Helper function to convert a 2D vector of [real, imaginary] into a vector of std::complex<double>
// ============================================================================
static std::vector<std::complex<double>> convert_to_complex(const std::vector<std::vector<double>>& iq_samples_input) {
    std::vector<std::complex<double>> iq_sample;
    iq_sample.reserve(iq_samples_input.size());
    for (const auto& pair : iq_samples_input) {
        if (pair.size() == 2) {
            iq_sample.emplace_back(pair[0], pair[1]);
        }
    }
    return iq_sample;
}

// ============================================================================
// Executes an FFT using FFTW on a vector of std::complex<double> and returns a 2D vector [real, imaginary]
// ============================================================================
static std::vector<std::vector<double>> execute_fft_ctoc(std::vector<std::complex<double>>& iq_sample) {
    int signalSize = static_cast<int>(iq_sample.size());
    if (signalSize == 0) {
        return {};
    }
    fftw_complex* in = reinterpret_cast<fftw_complex*>(iq_sample.data());
    std::vector<fftw_complex> out(signalSize);
    fftw_plan p = fftw_plan_dft_1d(signalSize, in, out.data(), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    std::vector<std::vector<double>> vec(signalSize, std::vector<double>(2));
    for (int i = 0; i < signalSize; ++i) {
        vec[i][0] = out[i][0];
        vec[i][1] = out[i][1];
    }
    return vec;
}

// ============================================================================
// Implementation of read_iq_samples for CSV and binary files (three-parameter version)
// Default CSV columns: {"Real", "Imaginary"}
// ============================================================================
std::vector<std::complex<double>> Analyzer::read_iq_samples(const std::string& input_file_path, IQDataType data_type, const std::vector<std::string>& csv_columns) {
    std::filesystem::path input_filepath(input_file_path);
    std::string ext = input_filepath.extension().string();

    // CSV case
    if (ext == ".csv" || ext == ".CSV") {
        if (csv_columns.size() < 2) {
            std::cerr << "Error: CSV columns must include at least two entries for IQ data (Real and Imaginary)." << std::endl;
            throw std::invalid_argument("Insufficient CSV column names provided.");
        }
        std::ifstream csv_file(input_file_path);
        if (!csv_file.is_open()) {
            std::cerr << "Error: Cannot open CSV file: " << input_file_path << std::endl;
            throw std::runtime_error("Cannot open CSV file.");
        }
        std::string header_line;
        if (!std::getline(csv_file, header_line)) {
            std::cerr << "Error: Cannot read header from CSV file: " << input_file_path << std::endl;
            throw std::runtime_error("CSV file header is empty.");
        }
        std::vector<std::string> headers;
        std::istringstream header_stream(header_line);
        std::string col;
        while (std::getline(header_stream, col, ',')) {
            // Trim whitespace from beginning and end
            col.erase(0, col.find_first_not_of(" \t\r\n"));
            col.erase(col.find_last_not_of(" \t\r\n") + 1);
            headers.push_back(col);
        }
        int real_index = -1, imag_index = -1;
        for (size_t i = 0; i < headers.size(); ++i) {
            if (headers[i] == csv_columns[0])
                real_index = static_cast<int>(i);
            if (headers[i] == csv_columns[1])
                imag_index = static_cast<int>(i);
        }
        if (real_index < 0 || imag_index < 0) {
            std::cerr << "Error: Specified column names not found in CSV header." << std::endl;
            throw std::invalid_argument("CSV column names not found.");
        }
        std::vector<std::complex<double>> iq_samples;
        std::string line;
        while (std::getline(csv_file, line)) {
            if (line.empty())
                continue;
            std::istringstream line_stream(line);
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(line_stream, token, ',')) {
                token.erase(0, token.find_first_not_of(" \t\r\n"));
                token.erase(token.find_last_not_of(" \t\r\n") + 1);
                tokens.push_back(token);
            }
            if (tokens.size() <= static_cast<size_t>(std::max(real_index, imag_index))) {
                std::cerr << "Warning: Incomplete line encountered, skipping: " << line << std::endl;
                continue;
            }
            try {
                double real_val = std::stod(tokens[real_index]);
                double imag_val = std::stod(tokens[imag_index]);
                iq_samples.push_back(std::complex<double>(real_val, imag_val));
            } catch (const std::exception& e) {
                std::cerr << "Error: Failed to parse line: " << line << ". Exception: " << e.what() << std::endl;
                throw std::runtime_error("CSV parsing error.");
            }
        }
        csv_file.close();
        return iq_samples;
    }
    // Binary file case (.iq or .bin)
    else if (ext == ".iq" || ext == ".bin") {
        if (data_type == IQDataType::FLOAT32) {
            return read_iq_sample_block<float>(input_file_path);
        } else if (data_type == IQDataType::FLOAT64) {
            return read_iq_sample_block<double>(input_file_path);
        } else if (data_type == IQDataType::INT16) {
            return read_iq_sample_block<std::int16_t>(input_file_path);
        }
        std::cerr << "Error: Invalid data type specified." << std::endl;
        throw std::invalid_argument("Invalid data type specified.");
    }
    else {
        std::cerr << "Error: Unsupported file extension: " << ext << std::endl;
        throw std::invalid_argument("Unsupported file extension.");
    }
}

// ============================================================================
// Two-parameter overload for read_iq_samples (calls the three-parameter version with default CSV columns)
// ============================================================================
std::vector<std::complex<double>> Analyzer::read_iq_samples(const std::string& input_file_path, IQDataType data_type) {
    std::vector<std::string> default_csv_columns = {"Real", "Imaginary"};
    return read_iq_samples(input_file_path, data_type, default_csv_columns);
}

// ============================================================================
// Public get_iq_samples functions using read_iq_samples
// ============================================================================
std::vector<std::vector<double>> Analyzer::get_iq_samples(const std::string& input_file_path, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<std::vector<double>> result;
    result.reserve(iq_sample.size());
    for (const auto &c : iq_sample) {
        result.push_back({ c.real(), c.imag() });
    }
    return result;
}

std::vector<std::vector<double>> Analyzer::get_iq_samples(const std::string& input_file_path, int start_sample, int end_sample, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<std::vector<double>> result;
    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty or could not be read." << std::endl;
        return result;
    }
    if (start_sample < 0 || end_sample <= start_sample || start_sample >= static_cast<int>(iq_sample.size())) {
        std::cerr << "Error: Invalid sample range." << std::endl;
        return result;
    }
    end_sample = std::min(end_sample, static_cast<int>(iq_sample.size()));
    result.reserve(end_sample - start_sample);
    for (int i = start_sample; i < end_sample; ++i) {
        result.push_back({ iq_sample[i].real(), iq_sample[i].imag() });
    }
    return result;
}

std::vector<std::vector<double>> Analyzer::get_iq_samples(const std::string& input_file_path, IQDataType data_type, const std::vector<std::string>& csv_columns) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type, csv_columns);
    std::vector<std::vector<double>> result;
    result.reserve(iq_sample.size());
    for (const auto &c : iq_sample) {
        result.push_back({ c.real(), c.imag() });
    }
    return result;
}

// ============================================================================
// FFT functions implementations
// ============================================================================
std::vector<std::vector<double>> Analyzer::fast_fourier_transform(const std::string& input_file_path, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty or not valid." << std::endl;
        return {};
    }
    return execute_fft_ctoc(iq_sample);
}

std::vector<std::vector<double>> Analyzer::fast_fourier_transform(const std::vector<std::vector<double>>& iq_samples) {
    std::vector<std::complex<double>> iq_sample = convert_to_complex(iq_samples);
    if (iq_sample.empty()) {
        std::cerr << "Error: Provided IQ samples are empty or invalid." << std::endl;
        return {};
    }
    return execute_fft_ctoc(iq_sample);
}

std::vector<std::vector<double>> Analyzer::fast_fourier_transform(const std::string& input_file_path, int start_sample, int end_sample, IQDataType data_type) {
    std::vector<std::vector<double>> iq_samples = get_iq_samples(input_file_path, start_sample, end_sample, data_type);
    if (iq_samples.empty()) {
        std::cerr << "Error: Could not extract IQ samples from the specified range." << std::endl;
        return {};
    }
    std::vector<std::complex<double>> iq_sample = convert_to_complex(iq_samples);
    return execute_fft_ctoc(iq_sample);
}

// ============================================================================
// PSD functions implementations
// ============================================================================
std::vector<double> Analyzer::calculate_PSD(const std::string& input_file_path, int start_sample, int end_sample, IQDataType data_type) {
    std::vector<std::vector<double>> iq_samples = get_iq_samples(input_file_path, start_sample, end_sample, data_type);
    if (iq_samples.empty()) {
        std::cerr << "Error: Could not extract IQ samples from the specified range." << std::endl;
        return {};
    }
    std::vector<std::complex<double>> iq_sample = convert_to_complex(iq_samples);
    if (iq_sample.empty()) {
        std::cerr << "Error: No valid IQ samples available for PSD calculation." << std::endl;
        return {};
    }
    std::vector<std::vector<double>> fft = execute_fft_ctoc(iq_sample);
    int fft_size = static_cast<int>(fft.size());
    std::vector<double> psd;
    psd.reserve(fft_size);
    for (int i = 0; i < fft_size; ++i) {
        double re = fft[i][0];
        double im = fft[i][1];
        double magnitude2 = re * re + im * im;
        psd.push_back(magnitude2 / fft_size);
    }
    return psd;
}

std::vector<double> Analyzer::calculate_PSD(const std::string& input_file_path, double sampleRate, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty or could not be read." << std::endl;
        return {};
    }
    std::vector<std::vector<double>> fft = execute_fft_ctoc(iq_sample);
    int fft_size = static_cast<int>(fft.size());
    std::vector<double> result;
    result.reserve(fft_size);
    for (int i = 0; i < fft_size; ++i) {
        double re = fft[i][0];
        double im = fft[i][1];
        double magnitude2 = re * re + im * im;
        result.push_back(magnitude2 / (fft_size * sampleRate));
    }
    return result;
}

std::vector<double> Analyzer::calculate_PSD(const std::vector<std::vector<double>>& iq_samples, double sampleRate) {
    std::vector<std::complex<double>> iq_sample = convert_to_complex(iq_samples);
    if (iq_sample.empty()) {
        std::cerr << "Error: Provided IQ samples are empty or invalid." << std::endl;
        return {};
    }
    std::vector<std::vector<double>> fft = execute_fft_ctoc(iq_sample);
    int fft_size = static_cast<int>(fft.size());
    std::vector<double> result;
    result.reserve(fft_size);
    for (int i = 0; i < fft_size; ++i) {
        double re = fft[i][0];
        double im = fft[i][1];
        double magnitude2 = re * re + im * im;
        result.push_back(magnitude2 / (fft_size * sampleRate));
    }
    return result;
}

// ============================================================================
// Spectrogram functions implementations
// ============================================================================
std::vector<std::vector<double>> Analyzer::generate_IQ_Spectrogram(const std::string& input_file_path, int overlap, int window_size, double sample_rate, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<std::vector<double>> result;
    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty or could not be read." << std::endl;
        return result;
    }
    int iq_sample_size = static_cast<int>(iq_sample.size());
    if (window_size <= 0 || window_size > iq_sample_size) {
        std::cerr << "Error: window_size is invalid or larger than the total samples." << std::endl;
        return result;
    }
    if (overlap < 0 || overlap >= window_size) {
        std::cerr << "Error: overlap must be >= 0 and < window_size." << std::endl;
        return result;
    }
    int hop_size = window_size - overlap;
    int num_windows = 1 + (iq_sample_size - window_size) / hop_size;
    if (num_windows <= 0) {
        std::cerr << "Error: Not enough samples to form even one window." << std::endl;
        return result;
    }
    result.resize(num_windows, std::vector<double>(window_size));
    for (int i = 0; i < num_windows; ++i) {
        int start_index = i * hop_size;
        int end_index = start_index + window_size;
        std::vector<std::complex<double>> iq_sample_window(iq_sample.begin() + start_index, iq_sample.begin() + end_index);
        std::vector<std::vector<double>> fft_result = execute_fft_ctoc(iq_sample_window);
        for (int j = 0; j < window_size; ++j) {
            double re = fft_result[j][0];
            double im = fft_result[j][1];
            double magnitude2 = re * re + im * im;
            double power = magnitude2 / window_size;
            double power_db = 10.0 * std::log10(power + 1e-20);
            result[i][j] = power_db;
        }
    }
    return result;
}

std::vector<std::vector<double>> Analyzer::generate_IQ_Spectrogram(const std::vector<std::vector<double>>& iq_samples_input, int overlap, int window_size, double sample_rate) {
    std::vector<std::complex<double>> iq_sample = convert_to_complex(iq_samples_input);
    std::vector<std::vector<double>> result;
    if (iq_sample.empty()) {
        std::cerr << "Error: Provided IQ samples are empty or invalid." << std::endl;
        return result;
    }
    int iq_sample_size = static_cast<int>(iq_sample.size());
    if (window_size <= 0 || window_size > iq_sample_size) {
        std::cerr << "Error: window_size is invalid or larger than the total samples." << std::endl;
        return result;
    }
    if (overlap < 0 || overlap >= window_size) {
        std::cerr << "Error: overlap must be >= 0 and < window_size." << std::endl;
        return result;
    }
    int hop_size = window_size - overlap;
    int num_windows = 1 + (iq_sample_size - window_size) / hop_size;
    if (num_windows <= 0) {
        std::cerr << "Error: Not enough samples to form even one window." << std::endl;
        return result;
    }
    result.resize(num_windows, std::vector<double>(window_size));
    for (int i = 0; i < num_windows; ++i) {
        int start_index = i * hop_size;
        int end_index = start_index + window_size;
        std::vector<std::complex<double>> iq_sample_window(iq_sample.begin() + start_index, iq_sample.begin() + end_index);
        std::vector<std::vector<double>> fft_result = execute_fft_ctoc(iq_sample_window);
        for (int j = 0; j < window_size; ++j) {
            double re = fft_result[j][0];
            double im = fft_result[j][1];
            double magnitude = std::sqrt(re * re + im * im);
            double power = (magnitude * magnitude) / iq_sample_size;
            double power_db_per_rad_sample = 10.0 * std::log10(power) - 10.0 * std::log10(2.0 * M_PI / sample_rate);
            result[i][j] = power_db_per_rad_sample;
        }
    }
    return result;
}

// ============================================================================
// Real part extraction functions
// ============================================================================
std::vector<double> Analyzer::real_part_iq_samples(const std::string& input_file_path, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<double> result;
    result.reserve(iq_sample.size());
    for (const auto& c : iq_sample) {
        result.push_back(c.real());
    }
    return result;
}

std::vector<double> Analyzer::real_part_iq_samples(const std::vector<std::vector<double>>& iq_samples, int start_sample, int end_sample) {
    std::vector<double> result;
    if (iq_samples.empty() || start_sample < 0 || end_sample <= start_sample || start_sample >= static_cast<int>(iq_samples.size())) {
        std::cerr << "Error: Invalid sample range or empty input data." << std::endl;
        return result;
    }
    int sample_size = std::min(end_sample, static_cast<int>(iq_samples.size()));
    result.reserve(sample_size - start_sample);
    for (int i = start_sample; i < sample_size; ++i) {
        if (iq_samples[i].size() == 2) {
            result.push_back(iq_samples[i][0]);
        }
    }
    return result;
}

// ============================================================================
// Imaginary part extraction functions
// ============================================================================
std::vector<double> Analyzer::imaginary_part_iq_samples(const std::string& input_file_path, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<double> result;
    result.reserve(iq_sample.size());
    for (const auto& c : iq_sample) {
        result.push_back(c.imag());
    }
    return result;
}

std::vector<double> Analyzer::imaginary_part_iq_samples(const std::vector<std::vector<double>>& iq_samples, int start_sample, int end_sample) {
    std::vector<double> result;
    if (iq_samples.empty() || start_sample < 0 || end_sample <= start_sample || start_sample >= static_cast<int>(iq_samples.size())) {
        std::cerr << "Error: Invalid sample range or empty input data." << std::endl;
        return result;
    }
    int sample_size = std::min(end_sample, static_cast<int>(iq_samples.size()));
    result.reserve(sample_size - start_sample);
    for (int i = start_sample; i < sample_size; ++i) {
        if (iq_samples[i].size() == 2) {
            result.push_back(iq_samples[i][1]);
        }
    }
    return result;
}
