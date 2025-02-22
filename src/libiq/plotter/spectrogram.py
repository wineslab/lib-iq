import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time

def calculate_window_duration_ms(num_samples, num_windows, sample_rate):
    total_duration_s = num_samples / sample_rate
    window_duration_s = total_duration_s / num_windows
    window_duration_ms = window_duration_s * 1000
    return window_duration_ms

def get_frequency_scale(freqs):
    freq = freqs[0]
    if freq >= 1e9:
        return 1e9, "(GHz)"
    elif freq >= 1e6:
        return 1e6, "(MHz)"
    elif freq >= 1e3:
        return 1e3, "(kHz)"
    else:
        return 1, "(Hz)"

def get_window_size_scale(times):
    t = times[-1] if len(times) > 0 else 0
    if t >= 1e9:
        return 1e9, "(10⁹)"
    elif t >= 1e6:
        return 1e6, "(10⁶)"
    elif t >= 1e3:
        return 1e3, "(10³)"
    else:
        return 1, ""

def update_y_labels(ax, num_freqs, sample_rate, center_frequency):
    # Get the current y-axis limits
    y_lims = ax.get_ylim()
    # Create a frequency array from -fs/2 to +fs/2, then shift by center_frequency
    freqs = np.linspace(-sample_rate / 2, sample_rate / 2, num_freqs) + center_frequency
    # Clip the y-axis limits to valid indices
    y_lims = np.clip(y_lims, 0, num_freqs - 1)
    # Determine the visible frequencies based on the current y-axis limits
    visible_freqs = freqs[int(y_lims[0]):int(y_lims[1])]
    if len(visible_freqs) == 0:
        return
    freq_scale, freq_unit = get_frequency_scale(visible_freqs)
    num_ticks = 11
    if (y_lims[1] - y_lims[0]) > 0:
        y_ticks = np.linspace(int(y_lims[0]), int(y_lims[1]), num_ticks)
    else:
        y_ticks = np.linspace(0, num_freqs - 1, min(num_freqs, num_ticks))
    valid_ticks = [int(y) for y in y_ticks if 0 <= int(y) < num_freqs]
    y_labels = [f"{freqs[idx] / freq_scale:.2f}" for idx in valid_ticks]
    ax.set_yticks(valid_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel(f"Frequency {freq_unit}")
    plt.draw()

def update_x_labels(ax, num_windows, window_duration_ms):
    # Get the current x-axis limits
    x_lims = ax.get_xlim()
    # Clip the x-axis limits to valid window indices
    x_lims = np.clip(x_lims, 0, num_windows - 1)
    num_ticks = 7
    if (x_lims[1] - x_lims[0]) > 0:
        x_ticks = np.linspace(int(x_lims[0]), int(x_lims[1]), num_ticks)
    else:
        x_ticks = np.linspace(0, num_windows - 1, min(num_windows, num_ticks))
    x_labels = np.round(x_ticks).astype(int)

    time_scale, time_unit = get_window_size_scale(x_labels)
    x_labels = x_labels / time_scale
    x_labels = [f"{x:.2f}" for x in x_labels]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(f"Window number {time_unit}")
    plt.draw()

def on_zoom(event, ax, num_freqs, sample_rate, center_frequency, num_windows, window_duration_ms):
    # Update both y and x labels when zooming or panning
    update_y_labels(ax, num_freqs, sample_rate, center_frequency)
    update_x_labels(ax, num_windows, window_duration_ms)

def spectrogram(spectrogram_data, sample_rate, center_frequency):
    """
    Displays the spectrogram (2D dB matrix) with the X-axis representing windows (time)
    and the Y-axis representing frequency, centered on center_frequency.
    """
    # Transpose the spectrogram data so that its shape is [frequencies, time]
    spectrogram_data = np.array(spectrogram_data).T
    print(f"There are {spectrogram_data.shape[1]} windows of size {spectrogram_data.shape[0]}")

    fig, ax = plt.subplots(dpi=300)

    max_power_db = np.max(spectrogram_data)
    min_power_db = np.min(spectrogram_data)
    img = ax.imshow(spectrogram_data, aspect='auto', cmap='jet', origin='lower',
                    vmin=min_power_db, vmax=max_power_db)

    num_windows = spectrogram_data.shape[1]
    num_freqs = spectrogram_data.shape[0]

    # Frequencies: from -fs/2 to +fs/2 shifted by center_frequency
    freqs = np.linspace(-sample_rate / 2, sample_rate / 2, num_freqs) + center_frequency
    y_ticks = np.linspace(0, num_freqs - 1, min(num_freqs, 11))
    freq_scale, freq_unit = get_frequency_scale(freqs)
    y_labels = [f"{freqs[int(y)] / freq_scale:.2f}" for y in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel(f"Frequency {freq_unit}")

    # X-axis represents the number of windows
    window_duration_ms = calculate_window_duration_ms(num_freqs * num_windows, num_windows, sample_rate)
    x_ticks = np.linspace(0, num_windows - 1, min(num_windows, 7))
    x_labels = np.round(x_ticks).astype(int)
    time_scale, time_unit = get_window_size_scale(x_labels)
    x_labels = x_labels / time_scale
    x_labels = [f"{x:.2f}" for x in x_labels]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(f"Window number {time_unit}")

    fig.colorbar(img, label='Power (dB)')
    fig.subplots_adjust(left=0.2)

    plt.text(1, 1.05,
             f'Each window is {window_duration_ms:.2f} ms',
             transform=ax.transAxes, ha='right')

    # Connect callbacks to update labels when the view changes (zoom or pan)
    ax.callbacks.connect('xlim_changed', lambda evt: on_zoom(evt, ax, num_freqs, sample_rate, center_frequency, num_windows, window_duration_ms))
    ax.callbacks.connect('ylim_changed', lambda evt: on_zoom(evt, ax, num_freqs, sample_rate, center_frequency, num_windows, window_duration_ms))

    plt.show()
