#ifndef FROMBINTOSIGMF_H
#define FROMBINTOSIGMF_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

namespace fd = std::filesystem;

class Converter {
public:
    static int from_bin_to_sigmf(const fd::path& filepath);
};

#endif // CONVERTER_H
