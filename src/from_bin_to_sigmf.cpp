#include "from_bin_to_sigmf.h"
#include <Python.h>

int Converter::from_bin_to_sigmf(const fd::path& filepath) {
    std::cout << "Processing file: " << filepath << std::endl;
    
    if (filepath.extension() != ".iq" && filepath.extension() != ".bin"){
        std::cerr << "Error: File extension not valid " << filepath << std::endl;
        return -1;
    }

    //py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    py::module my_module = py::module::import("RFDataFactory.SigMF.sigmf_converter");
    py::function convert_bin_to_mat = my_module.attr("convert_bin_to_mat");
    convert_bin_to_mat(std::string(filepath), std::string("/root/iq_samples_mat"));

    std::cout << "Arriva2" << std::endl;

    py::function convert_mat_to_sigmf = my_module.attr("convert_mat_to_sigmf");
    convert_mat_to_sigmf(std::string("/root/iq_samples_sigmf/"), std::string("/root/iq_samples_mat/"));

    std::cout << "Arriva3" << std::endl;

    return 0;
}
