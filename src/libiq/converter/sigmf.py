import json
import os

import scipy.io as sio

from libiq.utils.logger import logger

class SigMFConverter:
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

    def convert_to_sigmf(self, input_file_path, output_file_path):
        if not input_file_path.lower().endswith(".mat"):
            raise ValueError("Input file must have a .mat extension")

        mat_data = sio.loadmat(input_file_path, squeeze_me=True)

        sigmf_meta = {
            "global": {
                "sample_rate": float(mat_data["sample_rate"]),
                "hw": str(mat_data["hw"]),
                "version": str(mat_data["version"]),
            },
            "captures": [
                {
                    "frequency": float(mat_data["frequency"]),
                    "global_index": int(mat_data["global_index"]),
                    "sample_start": int(mat_data["sample_start"]),
                }
            ],
            "annotations": [
                {
                    "freq_lower_edge": float(mat_data["freq_lower_edge"]),
                    "freq_upper_edge": float(mat_data["freq_upper_edge"]),
                    "sample_start": int(mat_data["sample_start"]),
                }
            ],
        }

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w") as f:
            json.dump(sigmf_meta, f, indent=4)

        logger.info(f"SigMF metadata written to: {output_file_path}")
