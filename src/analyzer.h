#ifndef ANALYZER_H
#define ANALYZER_H

#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <fftw3.h>
#include <string>

class Analyzer {
public:
    Analyzer(){}
    int fast_fourier_transform(const std::string& input_file_path, const std::string& output_file_path);
    
};

#endif // ANALYZER_H