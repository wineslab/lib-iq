#include "analyzer.h"

std::vector<std::complex<double>> read_iq_sample(const std::string& input_file_path){
    std::filesystem::path input_filepath = input_file_path;
    std::cout << "Processing file: " << input_filepath << std::endl;
    std::vector<std::complex<double>> ris;

    if (!std::filesystem::exists(input_filepath)) {
        std::cerr << "Error: File does not exist in: " << input_filepath << std::endl;
        return ris;
    }

    if (input_filepath.extension() != ".iq" && input_filepath.extension() != ".bin"){
        std::cerr << "Error: File extension not valid, .iq or .bin is required " << input_filepath << std::endl;
        return ris;
    }

    std::ifstream file(input_filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error: File does not exist or cannot be opened" << std::endl;
        return ris;
    }

    std::vector<std::complex<double>> iq_samples;
    float real, imag;
    while (file.read(reinterpret_cast<char*>(&real), sizeof(real)) && file.read(reinterpret_cast<char*>(&imag), sizeof(imag))) {
        iq_samples.emplace_back(static_cast<double>(real), static_cast<double>(imag));
    }
    return iq_samples;
}

std::vector<std::vector<double>> execute_fft_ctoc(std::vector<std::complex<double>> iq_sample){
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

std::vector<double> Analyzer::real_part_iq_sample(const std::string& input_file_path){
    std::vector<std::complex<double>> iq_sample = read_iq_sample(input_file_path);
    std::vector<double> ris;

    for (const auto& complex_num : iq_sample) {
        ris.push_back(complex_num.real());
    }
    return ris;
}

std::vector<double> Analyzer::complex_part_iq_sample(const std::string& input_file_path){
    std::vector<std::complex<double>> iq_sample = read_iq_sample(input_file_path);
    std::vector<double> ris;

    for (const auto& complex_num : iq_sample) {
        ris.push_back(complex_num.imag());
    }
    return ris;
}

std::vector<std::vector<double>> Analyzer::get_iq_sample(const std::string& input_file_path){
    std::vector<std::complex<double>> iq_sample = read_iq_sample(input_file_path);
    std::vector<std::vector<double>> ris;

    for (const auto& complex_num : iq_sample) {
        std::vector<double> real_imag;
        real_imag.push_back(complex_num.real());
        real_imag.push_back(complex_num.imag());
        ris.push_back(real_imag);
    }
    return ris;
}

std::vector<std::vector<double>> Analyzer::fast_fourier_transform(const std::string& input_file_path){
    std::vector<std::complex<double>> iq_sample = read_iq_sample(input_file_path);
    std::vector<std::vector<double>> ris;
    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty." << std::endl;
        return ris;
    }
    std::vector<std::vector<double>> fft = execute_fft_ctoc(iq_sample);
    int fft_size = fft.size();
    for (int i = 0; i < fft_size; ++i) {
        double real = fft[i][0];
        double imag = fft[i][1];
        fft[i][0] = real;
        fft[i][1] = imag;
    }

    return fft;
}

std::vector<double> Analyzer::calculate_PSD(const std::string& input_file_path, double sampleRate) {
    std::vector<std::complex<double>> iq_sample = read_iq_sample(input_file_path);
    std::vector<double> ris;
    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty." << std::endl;
        return ris;
    }

    std::vector<std::vector<double>> fft = execute_fft_ctoc(iq_sample);
    std::vector<double> psd;
    int fft_size = fft.size();

    for (int i = 0; i < fft_size; i++) {
        psd.push_back((fft[i][0] * fft[i][0] + fft[i][1] * fft[i][1]) / (fft_size * sampleRate));
    }

    return psd;
}

std::vector<std::vector<double>> Analyzer::generate_IQ_Spectrogram_from_file(const std::string& input_file_path, int overlap, int window_size, double sample_rate) {
    std::vector<std::complex<double>> iq_sample = read_iq_sample(input_file_path);
    std::vector<std::vector<double>> res;

    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty." << std::endl;
        return res;
    }
    int iq_sample_size = iq_sample.size();

    int hop_size = window_size - overlap;
    int num_windows = 1 + (iq_sample.size() - window_size) / hop_size;

    std::vector<std::vector<double>> spectrogram(num_windows, std::vector<double>(window_size));

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
            spectrogram[i][j] = power_db_per_rad_sample;

        }
    }
    
    return spectrogram;
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

std::vector<std::vector<double>> Analyzer::generate_IQ_Spectrogram(std::vector<std::vector<double>> iq_samples_input, int overlap, int window_size, double sample_rate) {
    std::vector<std::complex<double>> iq_sample = convert_to_complex(iq_samples_input);
    std::vector<std::vector<double>> res;

    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty." << std::endl;
        return res;
    }
    int iq_sample_size = iq_sample.size();

    int hop_size = window_size - overlap;
    int num_windows = 1 + (iq_sample.size() - window_size) / hop_size;

    std::vector<std::vector<double>> spectrogram(num_windows, std::vector<double>(window_size));

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
            spectrogram[i][j] = power_db_per_rad_sample;

        }
    }
    
    return spectrogram;
}