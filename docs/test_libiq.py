import json
import os
import shutil

import numpy as np
import pytest

import libiq
import libiq.plotter.scatterplot as scplt
import libiq.plotter.spectrogram as sp
import libiq.plotter.waterfall as wf
from libiq.classifier.cnn import Classifier
from libiq.classifier.preprocessing import preprocess_data
from libiq.converter.mat import MATConverter
from libiq.converter.sigmf import SigMFConverter
from libiq.utils.logger import logger

report_path = "sample_data/test_results/reports/"
plots_path = "sample_data/test_results/plots/"

def create_directories(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            logger.debug(f"Directory created successfully: {path}")

def test_classifier():
    if os.path.exists("sample_data/test_results"):
        shutil.rmtree("sample_data/test_results")
    directories = [report_path, plots_path]
    create_directories(directories)

    classification_model = Classifier(
        time_window=1,
        input_vector=1536,
        moving_avg_window=30,
        extraction_window=600,
        epochs=7,
        batch_size=32,
        plots=True,
        interactive_plots=False,
    )

    classification_model.load_model("sample_data/test_model.keras")

    test_percentage = 0.8

    x_train, x_test, y_train, y_test = preprocess_data(
        "sample_data/combined_output.csv",
        test_percentage,
        report=False,
        report_path=f"{report_path}test.html",
    )

    if classification_model.plots:
        wf.plot_waterfall(
            "sample_data/combined_output.csv",
            interactive_plots=classification_model.interactive_plots,
            fft_size=classification_model.extraction_window,
            path=f"{plots_path}waterfall_test.pdf",
        )

    acc, precision, recall, f1, label = classification_model.cnn_test(
        x_test, y_test, plots_path
    )

    assert label == 'Radar'
    assert os.path.isfile(f"{plots_path}confusion_matrix_test.pdf")
    assert os.path.isfile(f"{plots_path}waterfall_test.pdf")


def test_classifier_with_report():

    pytest.importorskip("ydata_profiling", reason="ydata_profiling not installed")

    if os.path.exists("sample_data/test_results"):
        shutil.rmtree("sample_data/test_results")
    directories = [report_path, plots_path]
    create_directories(directories)

    classification_model = Classifier(
        time_window=1,
        input_vector=1536,
        moving_avg_window=30,
        extraction_window=600,
        epochs=7,
        batch_size=32,
        plots=True,
        interactive_plots=False,
    )

    classification_model.load_model("sample_data/test_model.keras")

    x_train, x_test, y_train, y_test = preprocess_data(
        "sample_data/combined_output.csv",
        0.8,
        report=True,
        report_path=f"{report_path}test.html",
    )

    if classification_model.plots:
        wf.plot_waterfall(
            "sample_data/combined_output.csv",
            interactive_plots=classification_model.interactive_plots,
            fft_size=classification_model.extraction_window,
            path=f"{plots_path}waterfall_test.pdf",
        )

    acc, precision, recall, f1, label = classification_model.cnn_test(
        x_test, y_test, plots_path
    )

    assert label == 'Radar'
    assert os.path.isfile(f"{plots_path}confusion_matrix_test.pdf")
    assert os.path.isfile(f"{plots_path}waterfall_test.pdf")
    assert os.path.isfile(f"{report_path}test.html")


def test_utils():
    input_file_path = "sample_data/combined_output.csv"

    analyzer = libiq.Analyzer()

    data_type = libiq.IQDataType.INT16.value

    sample_rate = 1000000
    diff = 1000  # max value = 2147483647
    start = 0
    end = start + diff

    window_size = 32
    overlap = 4

    fft = analyzer.fastFourierTransform(input_file_path, data_type)
    logger.debug(f"FFT shape calculated with overload 1: {np.shape(fft)}")
    assert np.shape(fft) == (3600, 2)

    iq = analyzer.getIQSamples(input_file_path, start, start + diff, data_type)
    fft = analyzer.fastFourierTransform(iq)
    logger.debug(f"FFT shape calculated with overload 2: {np.shape(fft)}")
    assert np.shape(fft) == (1000, 2)

    fft = analyzer.fastFourierTransform(
        input_file_path, start, start + diff, data_type
    )
    logger.debug(f"FFT shape calculated with overload 3: {np.shape(fft)}\n")
    assert np.shape(fft) == (1000, 2)

    psd = analyzer.calculatePSD(input_file_path, sample_rate, data_type)
    logger.debug(f"PSD shape calculated with overload 1: {np.shape(psd)}")
    assert np.shape(psd) == (3600,)

    iq = analyzer.getIQSamples(input_file_path, start, start + diff, data_type)
    psd = analyzer.calculatePSD(iq, sample_rate)
    logger.debug(f"PSD shape calculated with overload 2: {np.shape(psd)}")
    assert np.shape(psd) == (1000,)

    psd = analyzer.calculatePSD(input_file_path, start, start + diff, data_type)
    logger.debug(f"PSD shape calculated with overload 3: {np.shape(psd)}\n")
    assert np.shape(psd) == (1000,)

    iq_samples = analyzer.getIQSamples(input_file_path, data_type)
    logger.debug(f"iq_samples shape extracted with overload 1: {np.shape(iq_samples)}")
    assert np.shape(iq_samples) == (3600, 2)

    iq_samples = analyzer.getIQSamples(input_file_path, start, end, data_type)
    logger.debug(f"iq_samples shape extracted with overload 2: {np.shape(iq_samples)}")
    assert np.shape(iq_samples) == (1000, 2)

    iq_samples = analyzer.getIQSamples(
        input_file_path, data_type, ["Real", "Imaginary"]
    )
    logger.debug(
        f"iq_samples shape extracted with overload 3: {np.shape(iq_samples)}\n"
    )
    assert np.shape(iq_samples) == (3600, 2)

    spectrogram = analyzer.generateIQSpectrogram(
        input_file_path, overlap, window_size, sample_rate, data_type
    )
    logger.debug(f"Spectrogram shape with overload 1: {np.shape(spectrogram)}")
    assert np.shape(spectrogram) == (128, 32)

    iq = analyzer.getIQSamples(input_file_path, start, start + diff, data_type)
    spectrogram_mem = analyzer.generateIQSpectrogram(
        iq, overlap, window_size, sample_rate
    )
    logger.debug(f"Spectrogram shape with overload 2: {np.shape(spectrogram_mem)}\n")
    assert np.shape(spectrogram_mem) == (35, 32)

    real_part = analyzer.realPartIQSamples(input_file_path, data_type)
    logger.debug(f"Real part shape with overload 1: {np.shape(real_part)}")
    assert np.shape(real_part) == (3600,)

    real_mem = analyzer.realPartIQSamples(iq, 0, diff)
    logger.debug(f"Real part shape with overload 2: {np.shape(real_mem)}\n")
    assert np.shape(real_mem) == (1000,)

    imag_part = analyzer.imaginaryPartIQSamples(input_file_path, data_type)
    logger.debug(f"Imaginary part shape with overload 1: {np.shape(imag_part)}")
    assert np.shape(imag_part) == (3600,)

    imag_mem = analyzer.imaginaryPartIQSamples(iq, 0, diff)
    logger.debug(f"Imaginary part shape with overload 2: {np.shape(imag_mem)}")
    assert np.shape(imag_mem) == (1000,)


def test_converter():
    file_path1 = "sample_data/combined_output.csv"
    file_path2 = "sample_data/test_results/combined_output.mat"
    file_path3 = "sample_data/test_results/combined_output.sigmf-meta"

    mat_converter = MATConverter(
        freq_lower_edge=213456,
        freq_upper_edge=3456768,
        sample_rate=23456,
        frequency=567890,
        global_index=9999,
        sample_start=1,
        hw="superpc",
        version="1.0.0",
    )

    sigmf_converter = SigMFConverter(
        freq_lower_edge=213456,
        freq_upper_edge=3456768,
        sample_rate=23456,
        frequency=567890,
        global_index=9999,
        sample_start=1,
        hw="superpc",
        version="1.0.0",
    )

    mat_converter.convert_to_mat(str(file_path1), str(file_path2))
    sigmf_converter.convert_to_sigmf(str(file_path2), str(file_path3))

    assert os.path.isfile(file_path2)
    assert os.path.isfile(file_path3)

    with open(file_path3, "r") as f:
        data = json.load(f)

    # Controlli sui campi globali
    assert "global" in data
    assert data["global"].get("sample_rate") == 23456.0
    assert data["global"].get("hw") == "superpc"
    assert data["global"].get("version") == "1.0.0"

    assert "captures" in data and len(data["captures"]) > 0
    capture = data["captures"][0]
    assert capture.get("frequency") == 567890.0
    assert capture.get("global_index") == 9999
    assert capture.get("sample_start") == 1

    assert "annotations" in data and len(data["annotations"]) > 0
    annotation = data["annotations"][0]
    assert annotation.get("freq_lower_edge") == 213456.0
    assert annotation.get("freq_upper_edge") == 3456768.0
    assert annotation.get("sample_start") == 1


def test_scatterplot():
    input_file_path = "sample_data/combined_output.csv"

    analyzer = libiq.Analyzer()

    data_type = libiq.IQDataType.INT16.value

    diff = 10000000  # max value = 2147483647
    start = 0
    end = start + diff

    grid = False
    data_formats = ["real-imag", "magnitude-phase"]
    data_format = data_formats[0]

    iq = analyzer.getIQSamples(input_file_path, start, end, data_type)

    scplt.scatterplot(iq, data_format, grid, False, f"{plots_path}scatterplot.pdf")

    assert os.path.isfile(f"{plots_path}scatterplot.pdf")


def test_spectrogram():
    input_file_path = "sample_data/combined_output.csv"

    analyzer = libiq.Analyzer()

    data_type = libiq.IQDataType.INT16.value

    onverlap = 300
    window_size = 1536
    sample_rate = 1000000
    center_frequency = 1000000000

    diff = 1000000  # max value = 2147483647
    start = 0
    end = start + diff
    logger.debug(window_size)

    iq = analyzer.getIQSamples(input_file_path, start, end, data_type)
    fft = analyzer.generateIQSpectrogram(iq, onverlap, window_size, sample_rate)
    sp.spectrogram(
        fft, sample_rate, center_frequency, False, f"{plots_path}spectrogram.pdf"
    )

    assert os.path.isfile(f"{plots_path}spectrogram.pdf")
