import matplotlib.pyplot as plt
import numpy as np

def calculate_window_duration_ms(num_samples, num_windows, sample_rate):
    total_duration_s = num_samples / sample_rate
    window_duration_s = total_duration_s / num_windows
    window_duration_ms = window_duration_s * 1000
    return window_duration_ms

def get_frequency_scale(freqs):
    freq = freqs[0]
    if freq >= 1e9:
        return 1e9, "GHz"
    elif freq >= 1e6:
        return 1e6, "MHz"
    elif freq >= 1e3:
        return 1e3, "kHz"
    else:
        return 1, "Hz"

def spectrogram(spectrogram, sample_rate, center_frequency):
    spectrogram = np.array(spectrogram).T
    print(spectrogram.shape)
    plt.figure()

    max_power_db = np.max(spectrogram)
    min_power_db = np.min(spectrogram)
    img = plt.imshow(spectrogram, aspect='auto', cmap='jet', origin='lower', vmin=min_power_db, vmax=max_power_db)

    num_windows = spectrogram.shape[1]

    x_ticks = np.linspace(0, num_windows-1, min(num_windows, 7))
    x_labels = np.linspace(0, spectrogram.shape[1], min(num_windows, 7))
    plt.xticks(x_ticks, labels=np.round(x_labels*0.001).astype(int))

    num_freqs = spectrogram.shape[0]
    freqs = np.linspace(-sample_rate / 2, sample_rate / 2, num_freqs) + center_frequency

    y_ticks = np.linspace(0, num_freqs-1, min(num_freqs, 11))

    freq_scale, freq_unit = get_frequency_scale(freqs)

    y_labels = [f"{freq / freq_scale:.4f}" for freq in freqs[y_ticks.astype(int)]]

    plt.yticks(y_ticks, labels=y_labels)
    window_ms = calculate_window_duration_ms(spectrogram.shape[0]*spectrogram.shape[1], spectrogram.shape[1], sample_rate)
    plt.xlabel(f'Window number (x10³) --- Each window is {window_ms} ms')
    plt.ylabel(f'Frequency ({freq_unit})')

    plt.colorbar(img, label='Power (db)')

    plt.subplots_adjust(left=0.2)

    plt.gca()

    plt.show()