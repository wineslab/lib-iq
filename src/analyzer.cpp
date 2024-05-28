#include "analyzer.h"

template <typename T>
std::vector<std::complex<double>> read_iq_sample(const std::string& input_file_path) {
    std::filesystem::path input_filepath = input_file_path;
    std::cout << "Processing file: " << input_filepath << std::endl;
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

    // Open the file
    std::ifstream file(input_filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error: File cannot be opened: " << input_filepath << std::endl;
        return iq_samples;
    }

    // Read IQ samples from the file
    T real, imag;
    while (file.read(reinterpret_cast<char*>(&real), sizeof(real)) && file.read(reinterpret_cast<char*>(&imag), sizeof(imag))) {
        iq_samples.emplace_back(static_cast<double>(real), static_cast<double>(imag));
    }

    if (file.bad()) {
        std::cerr << "Error: Problem occurred while reading the file: " << input_filepath << std::endl;
        return {};
    }

    return iq_samples;
}

std::vector<std::complex<double>> Analyzer::read_iq_samples(const std::string& input_file_path, IQDataType data_type) {
    if (data_type == IQDataType::FLOAT32) {
        return read_iq_sample<float>(input_file_path);
    } else if (data_type == IQDataType::FLOAT64) {
        return read_iq_sample<double>(input_file_path);
    }
    std::cerr << "Error: Invalid data type specified." << std::endl;
    return {};
}

std::vector<std::complex<double>> convert_to_complex(const std::vector<std::vector<double>>& iq_samples_input) {
    std::vector<std::complex<double>> iq_sample;
    for (const auto& pair : iq_samples_input) {
        if (pair.size() == 2) {
            iq_sample.emplace_back(pair[0], pair[1]);
        }
    }
    return iq_sample;
}

std::vector<std::vector<double>> execute_fft_ctoc(std::vector<std::complex<double>> iq_sample) {
    int signalSize = iq_sample.size();
    fftw_complex *in = reinterpret_cast<fftw_complex*>(iq_sample.data());
    std::vector<fftw_complex> out(signalSize);

    fftw_plan p = fftw_plan_dft_1d(signalSize, in, out.data(), FFTW_FORWARD, FFTW_ESTIMATE_PATIENT);
    fftw_execute(p);
    fftw_destroy_plan(p);

    std::vector<std::vector<double>> vec(signalSize, std::vector<double>(2));
    for (int i = 0; i < signalSize; ++i) {
        vec[i][0] = out[i][0] / signalSize;
        vec[i][1] = out[i][1] / signalSize;
    }

    return vec;
}

std::vector<std::vector<double>> Analyzer::fast_fourier_transform(const std::string& input_file_path, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<std::vector<double>> result;
    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty." << std::endl;
        return result;
    }
    result = execute_fft_ctoc(iq_sample);
    return result;
}

std::vector<std::vector<double>> Analyzer::fast_fourier_transform(const std::vector<std::vector<double>>& iq_samples) {
    std::vector<std::complex<double>> iq_sample = convert_to_complex(iq_samples);
    if (iq_sample.empty()) {
        std::cerr << "Error: Provided IQ samples are empty or invalid." << std::endl;
        return {};
    }
    return execute_fft_ctoc(iq_sample);
}

std::vector<double> Analyzer::calculate_PSD(const std::string& input_file_path, double sampleRate, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<double> result;
    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty." << std::endl;
        return result;
    }

    std::vector<std::vector<double>> fft = execute_fft_ctoc(iq_sample);
    int fft_size = fft.size();
    for (int i = 0; i < fft_size; i++) {
        result.push_back((fft[i][0] * fft[i][0] + fft[i][1] * fft[i][1]) / (fft_size * sampleRate));
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
    std::vector<double> result;
    int fft_size = fft.size();
    for (int i = 0; i < fft_size; i++) {
        result.push_back((fft[i][0] * fft[i][0] + fft[i][1] * fft[i][1]) / (fft_size * sampleRate));
    }
    return result;
}

std::vector<std::vector<double>> Analyzer::generate_IQ_Spectrogram(const std::string& input_file_path, int overlap, int window_size, double sample_rate, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<std::vector<double>> result;

    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty." << std::endl;
        return result;
    }

    int iq_sample_size = iq_sample.size();
    int hop_size = window_size - overlap;
    int num_windows = 1 + (iq_sample.size() - window_size) / hop_size;

    result.resize(num_windows, std::vector<double>(window_size));
    for (int i = 0; i < num_windows; ++i) {
        int start_index = i * hop_size;
        int end_index = start_index + window_size;
        if (end_index > iq_sample_size) {
            end_index = iq_sample_size;
        }
        std::vector<std::complex<double>> iq_sample_window(iq_sample.begin() + start_index, iq_sample.begin() + end_index);
        std::vector<std::vector<double>> fft_result = execute_fft_ctoc(iq_sample_window);

        for (int j = 0; j < window_size; ++j) {
            double magnitude = std::sqrt(fft_result[j][0] * fft_result[j][0] + fft_result[j][1] * fft_result[j][1]);
            double power = magnitude * magnitude / iq_sample_size;
            double power_db_per_rad_sample = 10 * std::log10(power) - 10 * std::log10(2 * M_PI / sample_rate);
            result[i][j] = power_db_per_rad_sample;
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

    int iq_sample_size = iq_sample.size();
    int hop_size = window_size - overlap;
    int num_windows = 1 + (iq_sample.size() - window_size) / hop_size;

    result.resize(num_windows, std::vector<double>(window_size));
    for (int i = 0; i < num_windows; ++i) {
        int start_index = i * hop_size;
        int end_index = start_index + window_size;
        if (end_index > iq_sample_size) {
            end_index = iq_sample_size;
        }
        std::vector<std::complex<double>> iq_sample_window(iq_sample.begin() + start_index, iq_sample.begin() + end_index);
        std::vector<std::vector<double>> fft_result = execute_fft_ctoc(iq_sample_window);

        for (int j = 0; j < window_size; ++j) {
            double magnitude = std::sqrt(fft_result[j][0] * fft_result[j][0] + fft_result[j][1] * fft_result[j][1]);
            double power = magnitude * magnitude / iq_sample_size;
            double power_db_per_rad_sample = 10 * std::log10(power) - 10 * std::log10(2 * M_PI / sample_rate);
            result[i][j] = power_db_per_rad_sample;
        }
    }

    return result;
}

std::vector<double> Analyzer::real_part_iq_sample(const std::string& input_file_path, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<double> result;
    for (const auto& complex_num : iq_sample) {
        result.push_back(complex_num.real());
    }
    return result;
}

std::vector<double> Analyzer::real_part_iq_sample(const std::vector<std::vector<double>>& iq_samples, int start_sample, int end_sample) {
    std::vector<double> result;

    if (iq_samples.empty() || start_sample < 0 || end_sample < 0 || start_sample >= end_sample || start_sample >= static_cast<int>(iq_samples.size())) {
        std::cerr << "Error: Invalid sample range or empty input data." << std::endl;
        return result;
    }

    int sample_size = std::min(end_sample, static_cast<int>(iq_samples.size()));
    for (int i = start_sample; i < sample_size; ++i) {
        if (iq_samples[i].size() == 2) {
            result.push_back(iq_samples[i][0]);
        }
    }
    return result;
}

std::vector<double> Analyzer::complex_part_iq_sample(const std::string& input_file_path, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<double> result;
    for (const auto& complex_num : iq_sample) {
        result.push_back(complex_num.imag());
    }
    return result;
}

std::vector<double> Analyzer::complex_part_iq_sample(const std::vector<std::vector<double>>& iq_samples, int start_sample, int end_sample) {
    std::vector<double> result;

    if (iq_samples.empty() || start_sample < 0 || end_sample < 0 || start_sample >= end_sample || start_sample >= static_cast<int>(iq_samples.size())) {
        std::cerr << "Error: Invalid sample range or empty input data." << std::endl;
        return result;
    }

    int sample_size = std::min(end_sample, static_cast<int>(iq_samples.size()));
    for (int i = start_sample; i < sample_size; ++i) {
        if (iq_samples[i].size() == 2) {
            result.push_back(iq_samples[i][1]);
        }
    }
    return result;
}

std::vector<std::vector<double>> Analyzer::get_iq_samples(const std::string& input_file_path, IQDataType data_type) {
    std::vector<std::complex<double>> iq_sample = read_iq_samples(input_file_path, data_type);
    std::vector<std::vector<double>> result;
    for (const auto& complex_num : iq_sample) {
        std::vector<double> real_imag;
        real_imag.push_back(complex_num.real());
        real_imag.push_back(complex_num.imag());
        result.push_back(real_imag);
    }
    return result;
}

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

    // Adjust end_sample if it exceeds the size of the samples
    end_sample = std::min(end_sample, static_cast<int>(iq_sample.size()));

    for (int i = start_sample; i < end_sample; ++i) {
        std::vector<double> real_imag;
        real_imag.push_back(iq_sample[i].real());
        real_imag.push_back(iq_sample[i].imag());
        result.push_back(real_imag);
    }
    return result;
}

std::vector<std::vector<double>> Analyzer::get_iq_samples(const std::vector<std::vector<double>>& iq_samples, int start_sample, int end_sample) {
    std::vector<std::vector<double>> result;

    // Validate IQ sample data
    if (iq_samples.empty() || start_sample < 0 || end_sample <= start_sample || start_sample >= static_cast<int>(iq_samples.size())) {
        std::cerr << "Error: Invalid sample range or empty input data." << std::endl;
        return result;
    }

    // Adjust end_sample if it exceeds the size of the samples
    end_sample = std::min(end_sample, static_cast<int>(iq_samples.size()));

    for (int i = start_sample; i < end_sample; ++i) {
        result.push_back(iq_samples[i]);
    }
    return result;
}
