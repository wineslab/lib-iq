%module libiqwrapped

%{
#include "analyzer.h"
#include "converter.h"
#include <vector>
#include <array>
#include <complex>
%}

%include "std_string.i"
%include "std_vector.i"
%include "stdint.i"
%include "std_array.i"
%include "std_complex.i"

%template(DoubleVector) std::vector<double>;
%template(VectorOfDoubleVector) std::vector<std::vector<double>>;
%template(DoubleArray) std::array<double, 2>;
%template(ComplexVector) std::vector<std::array<double, 2>>;

%include "converter.h"
%include "analyzer.h"
