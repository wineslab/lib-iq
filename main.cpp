#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include "./src/from_bin_to_sigmf.h"
#include <iostream>

namespace py = pybind11;
/*
PYBIND11_MODULE(libiq, m) {
    m.doc() = R"pbdoc(
        Converter
    )pbdoc";

    m.def("from_bin_to_sigmf", &Converter::from_bin_to_sigmf, R"pbdoc(
        Convert a binary file to SIGMF format
    )pbdoc");

    m.attr("__version__") = "dev";
}
*/
PYBIND11_MODULE(libiq, m) {
    py::class_<Converter>(m, "Converter").def_static("from_bin_to_sigmf", &Converter::from_bin_to_sigmf);
}