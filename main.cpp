#include <pybind11/pybind11.h>
#include "./src/calculations.h"
#include <iostream>

using namespace std;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

string printName(string name, string lastname) {
    return "This function works " + name + " " + lastname + "!";
}

double divide(double num1, double num2) {
    Calculator calculator;
    double result;
    result = calculator.divide(num1, num2);
    return result;
}

namespace py = pybind11;

PYBIND11_MODULE(libiq, m) {
    m.doc() = R"pbdoc(
        Calculations are sum, multiply and divide
    )pbdoc";

    m.def("printName", &printName, R"pbdoc(
        Print first and last name
    )pbdoc");

    m.def("multiply", [](int i, int j) { return i * j; }, R"pbdoc(
        Multiply two numbers
    )pbdoc");

    m.def("divide", &divide, R"pbdoc(
        Divide two numbers
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}