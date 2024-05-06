import sys
sys.path.append('/root/libiq-101')
import libiq
import multiprocessing
from multiprocessing import Value
import zmq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np

spectrogram = None
img = None
total_samples = 0
#spectrogram = Value('d', float('-inf'))  # 'd' indica un double
#img = Value('d', float('inf'))
max_value = Value('d', float('-inf'))  # 'd' indica un double
min_value = Value('d', float('inf'))

def calculate_window_duration_ms(num_samples, num_windows, sample_rate):
    total_duration_s = num_samples / sample_rate
    window_duration_s = total_duration_s / num_windows
    window_duration_ms = window_duration_s * 1000
    return window_duration_ms

def get_frequency_scale(center_frequency):
    if center_frequency >= 1e9:
        return 1e9, "GHz"
    elif center_frequency >= 1e6:
        return 1e6, "MHz"
    elif center_frequency >= 1e3:
        return 1e3, "kHz"
    else:
        return 1, "Hz"

def calculate_fft(message, overlap, window_size, sample_rate):
    analyzer = libiq.Analyzer() 
    fft = analyzer.generate_IQ_Spectrogram_live(message, overlap, window_size, sample_rate)
    return fft

def process_data(data, overlap, window_size, sample_rate):
    processed_data = calculate_fft(data, overlap, window_size, sample_rate)
    max_v = max(processed_data[0])
    min_v = min(processed_data[0])
    with max_value.get_lock():
        if max_value.value < max_v:
            max_value.value = max_v
    with min_value.get_lock():
        if min_value.value > min_v and min_v > -40:
            min_value.value = min_v  # Qui dovrebbe essere min_value, non max_value
    return processed_data

def update_graph(num, data_queue, sample_rate, center_frequency, window_size, spectrogram_size):
    global spectrogram
    global img
    global total_samples

    if spectrogram is None:
        time.sleep(5)

    if not data_queue.empty():
        data = data_queue.get()
        total_samples += len(data)
        if spectrogram is None:
            spectrogram = np.array(data).T
            img = plt.imshow(spectrogram, aspect='auto', cmap='jet', origin='lower', vmin=-50, vmax=50)
            plt.colorbar(img, label='Power (db)')

            num_freqs = window_size
            freqs = np.linspace(-sample_rate / 2, sample_rate / 2, num_freqs) + center_frequency
            freq_scale, freq_unit = get_frequency_scale(center_frequency)
            y_ticks = np.linspace(-sample_rate / 2, sample_rate / 2, 11) + center_frequency
            y_labels = [f"{freq / freq_scale:.4f}" for freq in y_ticks]
            plt.yticks(y_ticks, labels=y_labels)
            plt.ylim([-sample_rate / 2 + center_frequency, sample_rate / 2 + center_frequency])
            plt.ylabel(f'Frequency ({freq_unit})')
        else:
            spectrogram = np.hstack((spectrogram, np.array(data).T))
            # Check if the spectrogram size exceeds the limit
            if spectrogram.shape[1] > spectrogram_size:
                # Remove the first columns
                spectrogram = spectrogram[:, spectrogram.shape[1] - spectrogram_size:]
            img.set_data(spectrogram)
            img.set_clim(vmin=min_value.value, vmax=max_value.value)
            img.set_extent([0, spectrogram.shape[1], -sample_rate / 2 + center_frequency, sample_rate / 2 + center_frequency])  # Aggiorna i limiti dell'asse
        num_windows = spectrogram.shape[1]

        x_tiks_val = total_samples-1
        if x_tiks_val >= spectrogram_size:
            x_tiks_val = spectrogram_size
        x_ticks = np.linspace(0, x_tiks_val, min(num_windows, 7))
        x_labels = np.linspace(total_samples - spectrogram.shape[1], total_samples, min(num_windows, 7))
        plt.xticks(x_ticks, labels=np.round(x_labels).astype(int))

        window_ms = calculate_window_duration_ms(spectrogram.shape[0]*spectrogram.shape[1], spectrogram.shape[1], sample_rate)
        plt.xlabel(f'Window number (x10³) --- Each window is {window_ms} ms')

        plt.subplots_adjust(left=0.2)

        plt.gca()
        plt.gcf().canvas.draw()



def handle_message(data_queue, overlap, window_size, sample_rate, n_processes):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)

    socket.connect("tcp://127.0.0.1:55556")

    pool = multiprocessing.Pool(processes=n_processes)
    while True:
        message = []
        while len(message) < window_size:
            rcv = socket.recv()
            tmp = np.frombuffer(rcv, dtype=np.complex64)
            message.append([float(tmp[0].real), float(tmp[0].imag)])
            tmp = tmp[1:]
        result = pool.apply_async(process_data, (message, overlap, window_size, sample_rate,))
        processed_data = result.get()
        data_queue.put(processed_data)

def main():
    overlap = 0
    window_size = 256
    sample_rate = 1000000
    center_frequency = 1000000000
    spectrogram_size = 25
    n_processes = 20
    data_queue = multiprocessing.Queue()

    fig = plt.figure()
    ani = animation.FuncAnimation(fig, update_graph, fargs=(data_queue, sample_rate, center_frequency, window_size, spectrogram_size, ), interval=0.01)
    p = multiprocessing.Process(target=handle_message, args=(data_queue, overlap, window_size, sample_rate, n_processes))
    p.start()

    plt.show()

if __name__ == "__main__":
    main()