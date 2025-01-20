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

    /**
    * @brief Converts a binary file to a .mat file.
    *
    * This method reads a binary file containing IQ samples and converts it into a .mat file.
    * The binary file is expected to contain IQ samples as 16-bit integers.
    * The method checks if the input file exists and if it has the correct extension (.iq or .bin).
    * It then reads the IQ samples from the binary file and writes them into a .mat file.
    * The .mat file also contains additional metadata such as frequency, sample rate, hardware version, etc.
    *
    * @param input_file_path The path of the binary file from which to read the IQ samples.
    * @param output_file_path The path of the .mat file to which to write the IQ samples and metadata.
    *
    * @return 0 if the conversion is successful, -1 if there is an error or if the .mat file cannot be opened.
    */
    int from_bin_to_mat(const std::string& input_file_path, const std::string& output_file_path);

    /**
    * @brief Converts a .mat file to a SigMF file.
    *
    * This method reads a .mat file containing IQ samples and metadata and converts it into a SigMF file.
    * The .mat file is expected to contain IQ samples as well as metadata such as frequency, sample rate, hardware version, etc.
    * The method checks if the input file exists and if it has the correct extension (.mat).
    * It then reads the IQ samples and metadata from the .mat file and writes them into a SigMF file using the `create_sigmf_meta` method.
    *
    * @param input_file_path The path of the .mat file from which to read the IQ samples and metadata.
    * @param output_file_path The path of the SigMF file to which to write the IQ samples and metadata.
    *
    * @return 0 if the conversion is successful, -1 if there is an error or if the .mat file cannot be opened.
    */
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