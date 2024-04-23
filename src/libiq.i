%module libiq
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

namespace std {
    %template(DoubleVector) vector<double>;
    %template(VectorOfDoubleVector) vector<vector<double>>;
    %template(DoubleArray) array<double, 2>;
    %template(ComplexVector) vector<array<double, 2>>;
}

%include "converter.i"
%include "analyzer.i"