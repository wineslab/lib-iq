#include "analyzer.h"

std::vector<std::vector<double>> Analyzer::fast_fourier_transform(const std::string& input_file_path){
    std::filesystem::path input_filepath = input_file_path;
    std::cout << "Processing file: " << input_filepath << std::endl;

    std::vector<std::vector<double>> ris;

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
        std::cerr << "Errore nell'apertura del file." << std::endl;
        return ris;
    }

    std::vector<int16_t> iq_sample;
    int16_t value;
    while (file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
        iq_sample.push_back(value);
    }

    int signalSize = iq_sample.size(); 
    std::cout << "signal size: " << signalSize << std::endl;
    std::vector<double> in(signalSize);

    for (size_t i = 0; i < iq_sample.size(); ++i) {
        in[i] = static_cast<double>(iq_sample[i]);
    }

    file.close();

    fftw_complex *out_fft;

    out_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (signalSize / 2 + 1));

    fftw_plan plan = fftw_plan_dft_r2c_1d(signalSize, in.data(), out_fft, FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    std::vector<double> double_array;

    std::vector<std::vector<double>> vec(signalSize, std::vector<double>(2));

    for (int i = 0; i < signalSize; ++i) {
        vec[i][0] = out_fft[i][0];  // Parte reale
        vec[i][1] = out_fft[i][1];  // Parte immaginaria
    }

    fftw_free(out_fft);

    return vec;
}

std::vector<double> Analyzer::calculatePSD(const std::string& input_file_path, double sampleRate) {
    std::filesystem::path input_filepath = input_file_path;
    std::cout << "Processing file: " << input_filepath << std::endl;
    int16_t value;
    std::vector<double> in;

    // Lettura dei campioni I/Q dal file binario
    std::ifstream file(input_filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Impossibile aprire il file " << input_filepath << std::endl;
        return {};
    }

    while (file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
        in.push_back(static_cast<double>(value));
    }
    file.close();

    int signalSize = in.size();
    std::cout << "signal size: " << signalSize << std::endl;

    fftw_complex* fftResult;
    fftResult = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (signalSize / 2 + 1));

    fftw_plan plan = fftw_plan_dft_r2c_1d(signalSize, in.data(), fftResult, FFTW_ESTIMATE);

    fftw_execute(plan);

    std::vector<double> psd(signalSize / 2 + 1);
    for (int i = 0; i <= signalSize / 2; ++i) {
        psd[i] = (fftResult[i][0] * fftResult[i][0] + fftResult[i][1] * fftResult[i][1]) / (signalSize * sampleRate);
    }

    fftw_destroy_plan(plan);
    fftw_free(fftResult);

    return psd;
}