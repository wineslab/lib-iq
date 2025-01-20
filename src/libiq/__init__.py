from .libiqwrapped import Analyzer, IQDataType_FLOAT32, IQDataType_FLOAT64, Converter

from enum import Enum

class IQDataType(Enum):
    FLOAT32 = IQDataType_FLOAT32
    FLOAT64 = IQDataType_FLOAT64
