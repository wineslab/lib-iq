%module libiq
%{
#include "analyzer.h"
#include "converter.h"
%}

%include "std_string.i"
%include "std_vector.i"
%include "stdint.i"

%include "converter.i"
%include "analyzer.i"