#ifndef FROMBINTOSIGMF_H
#define FROMBINTOSIGMF_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>  // python interpreter
#include <pybind11/stl.h>  // type conversion

namespace py = pybind11;
namespace fd = std::filesystem;


class Converter {
public:
    static int from_bin_to_sigmf(const fd::path& filepath);
};

#endif // CONVERTER_H
