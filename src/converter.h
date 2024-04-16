#ifndef CONVERTER_H
#define CONVERTER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <cassert>
#include <iostream>
#include <map>
#include <matio.h>
#include <nlohmann/json.hpp>
#include <sigmf.h>

using json = nlohmann::json;

class Converter {
public:
    Converter(){}
    int from_bin_to_mat(const std::string& input_file_path, const std::string& output_file_path);
    int from_mat_to_sigmf(const std::string& input_file_path, const std::string& output_file_path);

    double freq_lower_edge;
    double freq_upper_edge;
    double sample_rate;
    double frequency;
    uint64_t global_index;
    uint64_t sample_start;
    std::string hw;
    std::string version;
};

#endif // CONVERTER_H