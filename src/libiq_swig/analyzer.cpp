#include "analyzer.h"

/**
 * @brief Template function to read IQ samples from file in a single block to improve performance.
 *
 * @tparam T float or double
 * @param input_file_path Path to the IQ file
 * @return std::vector<std::complex<double>> containing the IQ data
 */
template <typename T>
std::vector<std::complex<double>> read_iq_sample_block(const std::string& input_file_path) {
    std::filesystem::path input_filepath = input_file_path;
    std::vector<std::complex<double>> iq_samples;

    // Check if the file exists
    if (!std::filesystem::exists(input_filepath)) {
        std::cerr << "Error: File does not exist: " << input_filepath << std::endl;
        return iq_samples;
    }

    // Check file extension
    if (input_filepath.extension() != ".iq" && input_filepath.extension() != ".bin") {
        std::cerr << "Error: Invalid file extension. Required: .iq or .bin" << std::endl;
        return iq_samples;
    }

    // Determine file size
    std::uintmax_t file_size = std::filesystem::file_size(input_filepath);
    if (file_size < sizeof(T) * 2) {
        std::cerr << "Error: File is too small to contain IQ data." << std::endl;
        return iq_samples;
    }

    // Number of complex samples (each sample has a real part and an imaginary part)
    std::size_t num_complex_samples = file_size / (2 * sizeof(T));
    iq_samples.resize(num_complex_samples);

    // Open file in binary mode
    std::ifstream file(input_filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error: File cannot be opened: " << input_filepath << std::endl;
        iq_samples.clear();
        return iq_samples;
    }

    // Read the entire file in a single block into a buffer of type T
    std::vector<T> buffer(num_complex_samples * 2);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(T));

    // If we didn't read the expected amount of data, return an empty vector
    if (static_cast<std::size_t>(file.gcount()) != buffer.size() * sizeof(T)) {
        std::cerr << "Error: Unexpected file read size. Possibly corrupted file or read error." << std::endl;
        iq_samples.clear();
        return iq_samples;
    }

    // Convert the interleaved [real, imag, real, imag, ...] to std::complex<double>
    for (std::size_t i = 0; i < num_complex_samples; ++i) {
        iq_samples[i] = std::complex<double>(
            static_cast<double>(buffer[2 * i]),
            static_cast<double>(buffer[2 * i + 1])
        );
    }

    return iq_samples;
}

/**
 * @brief Private method of Analyzer to read IQ samples, dispatching to the specialized function.
 */
std::vector<std::complex<double>> Analyzer::read_iq_samples(const std::string& input_file_path, IQDataType data_type) {
    if (data_type == IQDataType::FLOAT32) {
        return read_iq_sample_block<float>(input_file_path);
    } else if (data_type == IQDataType::FLOAT64) {
        return read_iq_sample_block<double>(input_file_path);
    }
    std::cerr << "Error: Invalid data type specified." << std::endl;
    return {};
}

/**
 * @brief Helper function to convert a 2D vector of [real, imag] into std::vector<std::complex<double>>.
 */
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

/**
 * @brief Executes an FFT in-place on a vector of complex<double> and returns real+imag in a 2D vector.
 */
static std::vector<std::vector<double>> execute_fft_ctoc(std::vector<std::complex<double>>& iq_sample) {
    int signalSize = static_cast<int>(iq_sample.size());

    if (signalSize == 0) {
        return {};
    }

    // Use the memory of iq_sample as input by reinterpreting it as fftw_complex
    fftw_complex* in = reinterpret_cast<fftw_complex*>(iq_sample.data());

    // Allocate output
    std::vector<fftw_complex> out(signalSize);

    // Create plan
    fftw_plan p = fftw_plan_dft_1d(signalSize, in, out.data(), FFTW_FORWARD, FFTW_ESTIMATE_PATIENT);
    fftw_execute(p);
    fftw_destroy_plan(p);

    // Normalize and store in a 2D vector
    std::vector<std::vector<double>> vec(signalSize, std::vector<double>(2));
    double norm_factor = 1.0 / signalSize;
    for (int i = 0; i < signalSize; ++i) {
        vec[i][0] = out[i][0] * norm_factor; // real part
        vec[i][1] = out[i][1] * norm_factor; // imag part
    }
    return vec;
}

std::vector<std::vector<double>> Analyzer::fast_fourier_transform(const std::string& input_file_path, int start_sample, int end_sample, IQDataType data_type) {
    // Validate input range
    if (start_sample < 0 || end_sample <= start_sample) {
        std::cerr << "Error: Invalid sample range specified." << std::endl;
        return {};
    }

    // Extract IQ samples in the specified range
    std::vector<std::vector<double>> iq_samples = get_iq_samples(input_file_path, start_sample, end_sample, data_type);
    if (iq_samples.empty()) {
        std::cerr << "Error: Could not extract IQ samples from the specified range." << std::endl;
        return {};
    }

    // Convert IQ samples to complex format
    std::vector<std::complex<double>> iq_sample = convert_to_complex(iq_samples);

    // Perform FFT on the extracted samples
    return execute_fft_ctoc(iq_sample);
}

/**
 * @brief Performs an FFT on IQ data from a specified file.
 */
std::vector<std::vector<double>> Analyzer::fast_fourier_transform(const std::string& input_file_path, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<std::vector<double>> result;

    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty or not valid." << std::endl;
        return result;
    }

    result = execute_fft_ctoc(iq_sample);
    return result;
}

/**
 * @brief Performs an FFT on provided IQ data in [real, imag] format.
 */
std::vector<std::vector<double>> Analyzer::fast_fourier_transform(const std::vector<std::vector<double>>& iq_samples) {
    std::vector<std::complex<double>> iq_sample = convert_to_complex(iq_samples);

    if (iq_sample.empty()) {
        std::cerr << "Error: Provided IQ samples are empty or invalid." << std::endl;
        return {};
    }

    return execute_fft_ctoc(iq_sample);
}

std::vector<double> Analyzer::calculate_PSD(const std::string& input_file_path, int start_sample, int end_sample, IQDataType data_type) {
    // Validate input range
    if (start_sample < 0 || end_sample <= start_sample) {
        std::cerr << "Error: Invalid sample range specified." << std::endl;
        return {};
    }

    // Extract IQ samples in the specified range
    std::vector<std::vector<double>> iq_samples = get_iq_samples(input_file_path, start_sample, end_sample, data_type);
    if (iq_samples.empty()) {
        std::cerr << "Error: Could not extract IQ samples from the specified range." << std::endl;
        return {};
    }

    // Convert IQ samples to complex format
    std::vector<std::complex<double>> iq_sample = convert_to_complex(iq_samples);

    if (iq_sample.empty()) {
        std::cerr << "Error: No valid IQ samples available for PSD calculation." << std::endl;
        return {};
    }

    // Perform FFT on the extracted samples
    std::vector<std::vector<double>> fft = execute_fft_ctoc(iq_sample);

    // Calculate the PSD
    int fft_size = static_cast<int>(fft.size());
    std::vector<double> psd;
    psd.reserve(fft_size);

    for (int i = 0; i < fft_size; ++i) {
        double re = fft[i][0];
        double im = fft[i][1];
        double magnitude2 = re * re + im * im; // |FFT|^2
        psd.push_back(magnitude2 / fft_size); // Normalize PSD
    }

    return psd;
}


/**
 * @brief Calculates the Power Spectral Density (PSD) of IQ data from a file.
 */
std::vector<double> Analyzer::calculate_PSD(const std::string& input_file_path, double sampleRate, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<double> result;

    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty or could not be read." << std::endl;
        return result;
    }

    std::vector<std::vector<double>> fft = execute_fft_ctoc(iq_sample);
    int fft_size = static_cast<int>(fft.size());
    result.reserve(fft_size);

    for (int i = 0; i < fft_size; i++) {
        double re = fft[i][0];
        double im = fft[i][1];
        double magnitude2 = re * re + im * im; // |FFT|^2
        // PSD = |FFT|^2 / (N * fs)
        result.push_back(magnitude2 / (fft_size * sampleRate));
    }

    return result;
}

/**
 * @brief Calculates the Power Spectral Density (PSD) of provided IQ data in [real, imag].
 */
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

    for (int i = 0; i < fft_size; i++) {
        double re = fft[i][0];
        double im = fft[i][1];
        double magnitude2 = re * re + im * im;
        result.push_back(magnitude2 / (fft_size * sampleRate));
    }
    return result;
}

/**
 * @brief Generates an IQ spectrogram from a file.
 */
std::vector<std::vector<double>> Analyzer::generate_IQ_Spectrogram(const std::string& input_file_path, int overlap, int window_size, double sample_rate, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<std::vector<double>> result;

    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty or could not be read." << std::endl;
        return result;
    }

    int iq_sample_size = static_cast<int>(iq_sample.size());
    if (window_size <= 0 || window_size > iq_sample_size) {
        std::cerr << "Error: window_size is invalid or bigger than the total samples." << std::endl;
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

        // Slice of the IQ array
        std::vector<std::complex<double>> iq_sample_window(
            iq_sample.begin() + start_index,
            iq_sample.begin() + end_index
        );

        // FFT result
        std::vector<std::vector<double>> fft_result = execute_fft_ctoc(iq_sample_window);

        // Compute dB power for each frequency bin
        for (int j = 0; j < window_size; ++j) {
            double re = fft_result[j][0];
            double im = fft_result[j][1];
            double magnitude = std::sqrt(re * re + im * im);
            double power = (magnitude * magnitude) / iq_sample_size;
            double power_db_per_rad_sample = 10.0 * std::log10(power)
                                             - 10.0 * std::log10(2.0 * M_PI / sample_rate);
            result[i][j] = power_db_per_rad_sample;
        }
    }

    return result;
}

/**
 * @brief Generates a real-time IQ spectrogram from provided IQ data [real, imag].
 */
std::vector<std::vector<double>> Analyzer::generate_IQ_Spectrogram(const std::vector<std::vector<double>>& iq_samples_input, int overlap, int window_size, double sample_rate) {
    std::vector<std::complex<double>> iq_sample = convert_to_complex(iq_samples_input);
    std::vector<std::vector<double>> result;

    if (iq_sample.empty()) {
        std::cerr << "Error: Provided IQ samples are empty or invalid." << std::endl;
        return result;
    }

    int iq_sample_size = static_cast<int>(iq_sample.size());
    if (window_size <= 0 || window_size > iq_sample_size) {
        std::cerr << "Error: window_size is invalid or bigger than the total samples." << std::endl;
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

        std::vector<std::complex<double>> iq_sample_window(
            iq_sample.begin() + start_index,
            iq_sample.begin() + end_index
        );

        std::vector<std::vector<double>> fft_result = execute_fft_ctoc(iq_sample_window);

        for (int j = 0; j < window_size; ++j) {
            double re = fft_result[j][0];
            double im = fft_result[j][1];
            double magnitude = std::sqrt(re * re + im * im);
            double power = (magnitude * magnitude) / iq_sample_size;
            double power_db_per_rad_sample = 10.0 * std::log10(power)
                                             - 10.0 * std::log10(2.0 * M_PI / sample_rate);
            result[i][j] = power_db_per_rad_sample;
        }
    }

    return result;
}

/**
 * @brief Extracts the real part of IQ samples from a file.
 */
std::vector<double> Analyzer::real_part_iq_sample(const std::string& input_file_path, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<double> result;
    result.reserve(iq_sample.size());

    for (const auto& c : iq_sample) {
        result.push_back(c.real());
    }

    return result;
}

/**
 * @brief Extracts the real part of provided IQ samples [real, imag] in the specified range.
 */
std::vector<double> Analyzer::real_part_iq_sample(const std::vector<std::vector<double>>& iq_samples, int start_sample, int end_sample) {
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

/**
 * @brief Extracts the imaginary part of IQ samples from a file.
 */
std::vector<double> Analyzer::complex_part_iq_sample(const std::string& input_file_path, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<double> result;
    result.reserve(iq_sample.size());

    for (const auto& c : iq_sample) {
        result.push_back(c.imag());
    }

    return result;
}

/**
 * @brief Extracts the imaginary part of provided IQ samples [real, imag] in the specified range.
 */
std::vector<double> Analyzer::complex_part_iq_sample(const std::vector<std::vector<double>>& iq_samples, int start_sample, int end_sample) {
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

/**
 * @brief Extracts the real and imaginary parts of IQ samples from a file.
 */
std::vector<std::vector<double>> Analyzer::get_iq_samples(const std::string& input_file_path, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<std::vector<double>> result;
    result.reserve(iq_sample.size());

    for (auto &c : iq_sample) {
        result.push_back({c.real(), c.imag()});
    }
    return result;
}

/**
 * @brief Extracts IQ samples from a file within a specified range.
 */
std::vector<std::vector<double>> Analyzer::get_iq_samples(const std::string& input_file_path, int start_sample, int end_sample, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<std::vector<double>> result;

    // Validate IQ sample data
    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty or could not be read." << std::endl;
        return result;
    }

    // Validate sample range
    if (start_sample < 0 || end_sample <= start_sample || start_sample >= static_cast<int>(iq_sample.size())) {
        std::cerr << "Error: Invalid sample range." << std::endl;
        return result;
    }

    end_sample = std::min(end_sample, static_cast<int>(iq_sample.size()));
    result.reserve(end_sample - start_sample);

    for (int i = start_sample; i < end_sample; ++i) {
        result.push_back({iq_sample[i].real(), iq_sample[i].imag()});
    }

    return result;
}

/**
 * @brief Extracts IQ samples from provided data within a specified range.
 */
std::vector<std::vector<double>> Analyzer::get_iq_samples(const std::vector<std::vector<double>>& iq_samples, int start_sample, int end_sample) {
    std::vector<std::vector<double>> result;

    if (iq_samples.empty() || start_sample < 0 || end_sample <= start_sample || start_sample >= static_cast<int>(iq_samples.size())) {
        std::cerr << "Error: Invalid sample range or empty input data." << std::endl;
        return result;
    }

    end_sample = std::min(end_sample, static_cast<int>(iq_samples.size()));
    result.reserve(end_sample - start_sample);

    for (int i = start_sample; i < end_sample; ++i) {
        result.push_back(iq_samples[i]);
    }
    return result;
}
