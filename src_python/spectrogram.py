import matplotlib.pyplot as plt
import numpy as np

def spectrogram(spectrogram, sample_rate):
    spectrogram = np.array(spectrogram).T
    print(spectrogram.shape)
    plt.figure()

    # Set the color scale to have a minimum value of -150 and a maximum value of 100
    max_power_db = np.max(spectrogram)
    min_power_db = np.min(spectrogram)
    img = plt.imshow(spectrogram, aspect='auto', cmap='jet', origin='lower', vmin=min_power_db, vmax=max_power_db)

    num_windows = spectrogram.shape[1]

    # Set the x-axis labels to go from 0 to spectrogram.shape[0]*spectrogram.shape[1]
    x_ticks = np.linspace(0, num_windows-1, min(num_windows, 7))
    x_labels = np.linspace(0, spectrogram.shape[1], min(num_windows, 7))
    plt.xticks(x_ticks, labels=np.round(x_labels*0.001).astype(int))

    num_freqs = spectrogram.shape[0]
    freqs = np.arange(num_freqs) * (sample_rate / num_freqs) * 0.000001

    freqs = np.round(freqs, 2)

    y_ticks = np.linspace(0, num_freqs-1, min(num_freqs, 10))

    plt.yticks(y_ticks, labels=np.round(freqs[y_ticks.astype(int)], 2))

    plt.xlabel('Window number (x10³)')
    plt.ylabel('Frequency (MHz)')

    plt.colorbar(img, label='Power (db)')

    #plt.savefig("spectrogram.png")
    plt.show()
