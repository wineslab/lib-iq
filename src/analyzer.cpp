#include "analyzer.h"


int Analyzer::fast_fourier_transform(const std::string& input_file_path, const std::string& output_file_path){

    std::filesystem::path input_filepath = input_file_path;

    std::cout << "Processing file: " << input_filepath << std::endl;
    std::cout << "from_bin_to_mat" << std::endl;
    
    if (!std::filesystem::exists(input_filepath)) {
        std::cerr << "Error: File does not exist in: " << input_filepath << std::endl;
        return -1;
    }

    if (input_filepath.extension() != ".iq" && input_filepath.extension() != ".bin"){
        std::cerr << "Error: File extension not valid, .iq or .bin is required " << input_filepath << std::endl;
        return -1;
    }

    std::ifstream file(input_filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Errore nell'apertura del file." << std::endl;
        return -1;
    }

    std::filesystem::path output_dir = std::filesystem::path(output_file_path).parent_path();
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    std::vector<int16_t> iq_sample;
    int16_t value;
    while (file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
        iq_sample.push_back(value);
    }

    int N = iq_sample.size(); 
    std::cout << "N: " << N << std::endl;
    std::vector<double> in(N);

    for (size_t i = 0; i < iq_sample.size(); ++i) {
        in[i] = static_cast<double>(iq_sample[i]);
    }

    file.close();

    fftw_complex *out_fft;

    out_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));


    fftw_plan plan = fftw_plan_dft_r2c_1d(N, in.data(), out_fft, FFTW_ESTIMATE);

    fftw_execute(plan);

    std::ofstream outfile(output_file_path);
    std::cout << "Saving fourier in: " << output_file_path << std::endl;
    for (int i = 0; i < 10; ++i) {
        outfile << "Componente " << i << ": " << out_fft[i][0] << " + " << out_fft[i][1] << "i" << std::endl;
    }

    outfile.close();

    fftw_destroy_plan(plan);
    fftw_free(out_fft);

    return 0;
}

