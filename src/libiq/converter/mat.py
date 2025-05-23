import csv
import os

import numpy as np
import scipy.io as sio

from libiq.utils.logger import logger


class MATConverter:
    def __init__(
        self,
        freq_lower_edge,
        freq_upper_edge,
        sample_rate,
        frequency,
        global_index,
        sample_start,
        hw,
        version,
    ):
        self.freq_lower_edge = freq_lower_edge
        self.freq_upper_edge = freq_upper_edge
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.global_index = global_index
        self.sample_start = sample_start
        self.hw = hw
        self.version = version

    def convert_to_mat(self, input_file_path, output_file_path):
        _, ext = os.path.splitext(input_file_path.lower())
        if ext not in [".iq", ".bin", ".csv"]:
            raise ValueError("Unsupported file type. Use .iq, .bin, or .csv")

        real, imag = [], []

        if ext == ".csv":
            with open(input_file_path, "r") as csv_file:
                reader = csv.DictReader(csv_file)
                if (
                    "Real" not in reader.fieldnames
                    or "Imaginary" not in reader.fieldnames
                ):
                    raise ValueError("CSV must have 'Real' and 'Imaginary' columns")

                for row in reader:
                    try:
                        real.append(float(row["Real"].strip()))
                        imag.append(float(row["Imaginary"].strip()))
                    except ValueError:
                        continue
        else:
            iq_data = np.fromfile(input_file_path, dtype=np.int16)
            if len(iq_data) % 2 != 0:
                raise ValueError("Binary file contains an odd number of samples.")

            real = iq_data[::2].astype(np.float64)
            imag = iq_data[1::2].astype(np.float64)

        mat_dict = {
            "real": np.array(real),
            "imag": np.array(imag),
            "freq_lower_edge": self.freq_lower_edge,
            "freq_upper_edge": self.freq_upper_edge,
            "sample_rate": self.sample_rate,
            "frequency": self.frequency,
            "global_index": self.global_index,
            "sample_start": self.sample_start,
            "hw": self.hw,
            "version": self.version,
        }

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        sio.savemat(output_file_path, mat_dict)
        logger.info(f"Successfully wrote IQ data to: {output_file_path}")