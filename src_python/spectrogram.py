import matplotlib.pyplot as plt
import numpy as np

def spectrogram(spectrogram, sample_rate):
    spectrogram = np.array(spectrogram).T

    plt.figure()

    img = plt.imshow(spectrogram, aspect='auto', cmap='jet', origin='lower')

    num_windows = spectrogram.shape[1]

    x_ticks = np.linspace(0, num_windows-1, min(num_windows, 7))

    plt.xticks(x_ticks, labels=np.round(x_ticks).astype(int))

    num_freqs = spectrogram.shape[0]
    freqs = np.arange(num_freqs) * sample_rate / num_freqs

    freqs = np.round(freqs, 2)

    y_ticks = np.linspace(0, num_freqs-1, min(num_freqs, 10))

    plt.yticks(y_ticks, labels=np.round(freqs[y_ticks.astype(int)], 2))

    plt.xlabel('Window')
    plt.ylabel('Frequency (Hz)')

    plt.colorbar(img, label='Power')

    plt.savefig("spectrogram.png")