#include "analyzer.h"
/*
std::vector<std::complex<double>> generateIQSample(int N, double frequency) {
    std::vector<std::complex<double>> iq_sample(N);
    for (int i = 0; i < N; ++i) {
        double theta = 2.0 * M_PI * frequency * i / N;
        iq_sample[i] = std::complex<double>(cos(theta), sin(theta));
    }
    return iq_sample;
}

std::vector<std::complex<double>> generateChirp(int N, double startFrequency, double endFrequency) {
    std::vector<std::complex<double>> iq_sample(N);
    for (int i = 0; i < N; ++i) {
        double frequency = startFrequency + i * (endFrequency - startFrequency) / N;
        double theta = 2.0 * M_PI * frequency * i / N;
        iq_sample[i] = std::complex<double>(cos(theta), sin(theta));
    }
    return iq_sample;
}

    //int N = 1000;
    //double frequency1 = 0.2;
    //double frequency2 = 0.4;
    //std::vector<std::complex<double>> iq_samples = generateTwoToneSignal(N, frequency1, frequency2);

std::vector<std::complex<double>> generateMultiToneSignal(int N, std::vector<double> frequencies) {
    std::vector<std::complex<double>> iq_sample(N);
    for (int i = 0; i < N; ++i) {
        std::complex<double> sample(0, 0);
        for (double frequency : frequencies) {
            double theta = 2.0 * M_PI * frequency * i / N;
            sample += std::complex<double>(cos(theta), sin(theta));
        }
        iq_sample[i] = sample;
    }
    return iq_sample;
}

std::vector<std::complex<double>> generateSineWave(int N, double frequency) {
    std::vector<std::complex<double>> iq_sample(N);
    for (int i = 0; i < N; ++i) {
        double theta = 2.0 * M_PI * frequency * i / N;
        iq_sample[i] = std::complex<double>(cos(theta), sin(theta));
    }
    return iq_sample;
}
    //int N = 1000; // Numero di campioni
    //double frequency = 5.0; // Frequenza del segnale sinusoidale
    //std::vector<std::complex<double>> iq_samples = generateIQSample(N, frequency);

    //int N = 10000;
    //double startFrequency = 0.1;
    //double endFrequency = 0.5;
    //std::vector<std::complex<double>> iq_samples = generateChirp(N, startFrequency, endFrequency);

    //int N = 10000;
    //double frequency = 0.2;
    //std::vector<std::complex<double>> iq_samples = generateSineWave(N, frequency);
std::vector<std::complex<double>> generateTwoToneSignal(int N, double frequency1, double frequency2) {
    std::vector<std::complex<double>> iq_sample(N);
    for (int i = 0; i < N; ++i) {
        double theta1 = 2.0 * M_PI * frequency1 * i / N;
        double theta2 = 2.0 * M_PI * frequency2 * i / N;
        iq_sample[i] = std::complex<double>(cos(theta1), sin(theta1)) + std::complex<double>(cos(theta2), sin(theta2));
    }
    return iq_sample;
}
*/
//////////////////////////////////////////////////////

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
        std::cerr << "Errore nell'apertura del file." << std::endl;
        return ris;
    }

    std::vector<std::complex<double>> iq_samples;
    uint16_t real, imag;
    while (file.read(reinterpret_cast<char*>(&real), sizeof(real)) && file.read(reinterpret_cast<char*>(&imag), sizeof(imag))) {
        iq_samples.emplace_back(static_cast<double>(real), static_cast<double>(imag));
    }
    return iq_samples;
}

std::vector<std::vector<double>> execute_fft_ctoc(std::vector<std::complex<double>> iq_sample){
    // Esegui la FFT sui campioni IQ
    int signalSize = iq_sample.size();
    fftw_complex *in = reinterpret_cast<fftw_complex*>(iq_sample.data());
    fftw_complex *out;
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * signalSize);

    fftw_plan p = fftw_plan_dft_1d(signalSize, in, out, FFTW_FORWARD, FFTW_ESTIMATE_PATIENT);
    fftw_execute(p);

    fftw_destroy_plan(p);

    std::vector<std::vector<double>> vec(signalSize, std::vector<double>(2));

    for (int i = 0; i < signalSize; ++i) {
        vec[i][0] = out[i][0];
        vec[i][1] = out[i][1];
    }

    fftw_free(out);
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
    for (int i = 0; i < fft_size; i++) {
        double real = fft[i][0];
        double imag = fft[i][1];
        fft[i][0] = sqrt(real * real + imag * imag);  // Ampiezza
        fft[i][1] = atan2(imag, real);  // Fase
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

void Analyzer::generate_IQ_Scatterplot(const std::string& input_file_path) {
    std::vector<std::complex<double>> iq_sample = read_iq_sample(input_file_path);

    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty." << std::endl;
        return;
    }

    double minReal = std::numeric_limits<double>::max();
    double maxReal = std::numeric_limits<double>::min();
    double minImag = std::numeric_limits<double>::max();
    double maxImag = std::numeric_limits<double>::min();
    for (const auto& sample : iq_sample) {
        minReal = std::min(minReal, sample.real());
        maxReal = std::max(maxReal, sample.real());
        minImag = std::min(minImag, sample.imag());
        maxImag = std::max(maxImag, sample.imag());
    }

    // Creazione del plot
    cv::Mat scatterplot(700, 700, CV_8UC3, cv::Scalar(255, 255, 255)); // Creazione di un'immagine bianca più grande

    // Disegno degli assi
    cv::line(scatterplot, cv::Point(70, 650), cv::Point(650, 650), cv::Scalar(255, 0, 0)); // Asse x
    cv::line(scatterplot, cv::Point(70, 25), cv::Point(70, 650), cv::Scalar(255, 0, 0)); // Asse y

    double midReal1 = minReal + (maxReal - minReal) / 3;
    double midReal2 = minReal + 2 * (maxReal - minReal) / 3;
    double midImag1 = minImag + (maxImag - minImag) / 3;
    double midImag2 = minImag + 2 * (maxImag - minImag) / 3;

    // Aggiunta delle etichette sugli assi
    cv::putText(scatterplot, std::to_string(static_cast<int>(minReal)), cv::Point(70, 670), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
    cv::putText(scatterplot, std::to_string(static_cast<int>(maxReal)), cv::Point(70 + static_cast<int>((maxReal - minReal) * 550.0 / (maxReal - minReal)), 670), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
    cv::putText(scatterplot, std::to_string(static_cast<int>(minImag)), cv::Point(50, 650), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
    cv::putText(scatterplot, std::to_string(static_cast<int>(maxImag)), cv::Point(15, 650 - static_cast<int>((maxImag - minImag) * 550.0 / (maxImag - minImag))), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));

    cv::putText(scatterplot, std::to_string(static_cast<int>(midReal1)), cv::Point(70 + static_cast<int>((midReal1 - minReal) * 550.0 / (maxReal - minReal)), 670), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
    cv::putText(scatterplot, std::to_string(static_cast<int>(midReal2)), cv::Point(70 + static_cast<int>((midReal2 - minReal) * 550.0 / (maxReal - minReal)), 670), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
    cv::putText(scatterplot, std::to_string(static_cast<int>(midImag1)), cv::Point(15, 650 - static_cast<int>((midImag1 - minImag) * 550.0 / (maxImag - minImag))), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
    cv::putText(scatterplot, std::to_string(static_cast<int>(midImag2)), cv::Point(15, 650 - static_cast<int>((midImag2 - minImag) * 550.0 / (maxImag - minImag))), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));

    // Disegno dei punti sullo scatterplot
    for (const auto& sample : iq_sample) {
        // Mappatura delle coordinate del sample nel range 50-600 per il plot
        int x = 70 + static_cast<int>((sample.real() - minReal) * 550.0 / (maxReal - minReal)); // Scaling tra minReal e maxReal a 50 e 600
        int y = 650 - static_cast<int>((sample.imag() - minImag) * 550.0 / (maxImag - minImag)); // Scaling tra minImag e maxImag a 50 e 600 e inversione dell'asse y
        cv::circle(scatterplot, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), cv::FILLED); // Disegno del punto sullo scatterplot
    }

    // Salvataggio del plot
    cv::imwrite("/root/libiq-101/IQ_Scatterplot.png", scatterplot);


}


void Analyzer::generate_IQ_Spectrogram(const std::string& input_file_path, int overlap, int window_size) {
    std::vector<std::complex<double>> iq_sample = read_iq_sample(input_file_path);

    if (iq_sample.empty()) {
        std::cerr << "Error: File is empty." << std::endl;
        return;
    }

    int num_windows = iq_sample.size() / window_size;
    std::cout << iq_sample.size() << " " << num_windows << std::endl;

    cv::Mat spectrogram(window_size, num_windows, CV_32F);

    for (int i = 0; i < num_windows; ++i) {
        std::vector<std::complex<double>> iq_sample_window(window_size);
        for(int j = 0; j < window_size; j++){
            iq_sample_window[j] = iq_sample[i*window_size + j];
        }

        std::vector<std::vector<double>> fft_result = execute_fft_ctoc(iq_sample_window);

        for (int j = 0; j < window_size; ++j) {
            float magnitude = std::sqrt(fft_result[j][0] * fft_result[j][0] + fft_result[j][1] * fft_result[j][1]);
            spectrogram.at<float>(i, j) = 10 * std::log10(magnitude);
        }
    }
    
    cv::Mat img;
    cv::normalize(spectrogram, img, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(img, img, cv::COLORMAP_JET);
    cv::convertScaleAbs(img, img, 1.0, 2.0); // Aumenta il contrasto
    cv::imwrite("/root/libiq-101/IQ_Spectogram.png", img);

}


