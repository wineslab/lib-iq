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

class Analyzer {
public:
    Analyzer(){}
    std::vector<std::vector<double>> fast_fourier_transform(const std::string& input_file_path);
    std::vector<double> calculatePSD(const std::string& input_file_path, double sampleRate);
};

#endif // ANALYZER_H