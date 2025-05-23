from enum import Enum

from .libiqwrapped import (
    Analyzer,
    IQDataType_FLOAT32,
    IQDataType_FLOAT64,
    IQDataType_INT16
)


class IQDataType(Enum):
    FLOAT32 = IQDataType_FLOAT32
    FLOAT64 = IQDataType_FLOAT64
    INT16 = IQDataType_INT16
