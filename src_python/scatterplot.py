import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

def process_data(iq_sample, data_format):
    real = [x[0] for x in iq_sample]
    imag = [x[1] for x in iq_sample]
    if data_format == 'real-imag':
        return real, imag
    elif data_format == 'magnitude-phase':
        magnitude = []
        phase = []
        for i in range(len(real)):
            magnitude.append(math.sqrt(real[i]**2 + imag[i]**2))
            phase.append(math.atan2(imag[i], real[i]))
        return magnitude, phase

def animated_scatterplot(iq_sample, data_format, interval=100, window=50, grids=False):
    I_data, Q_data = process_data(iq_sample, data_format)

    if data_format == 'real-imag':
        iq_sample = np.array(iq_sample)
        I_data = iq_sample[:, 0]
        Q_data = iq_sample[:, 1]

    fig, ax = plt.subplots(dpi=300)
    fig.set_facecolor('black')
    ax.set_facecolor('black')

    if grids:
        ax1 = fig.add_subplot()
        ax1.xaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax1.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax1.set_facecolor('black')
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax1.set_xticks(np.linspace(-1, 1, 10))
        ax1.set_yticks(np.linspace(-1, 1, 10))

    sc = ax.scatter([], [], s=1, c='lime')

    if data_format == 'real-imag':
        ax.set_title('I/Q Scatter Plot (Real-Imag)', color='white', y=1.07)
        ax.set_xlabel('Q_data (1x10⁵)', color='white', labelpad=1)
        ax.set_ylabel('I_data (1x10⁵)', color='white', labelpad=1)
    elif data_format == 'magnitude-phase':
        ax.set_title('I/Q Scatter Plot (Magnitude-Phase)', color='white', y=1.07)
        ax.set_xlabel('Magnitude (10⁵)', color='white', labelpad=1)
        ax.set_ylabel('Phase', color='white', labelpad=1)

    def update(frame):
        if frame < window:
            xdata = I_data[:frame]
            ydata = Q_data[:frame]
        else:
            xdata = I_data[frame - window:frame]
            ydata = Q_data[frame - window:frame]

        if len(xdata) == 0 or len(ydata) == 0:
            return sc,

        sc.set_offsets(np.c_[xdata, ydata])

        if data_format == 'real-imag':
            I_min, I_max = xdata.min(), xdata.max()
            Q_min, Q_max = ydata.min(), ydata.max()
            I_range = I_max - I_min
            Q_range = Q_max - Q_min

            if I_range == 0:
                I_range = 1e-3
            if Q_range == 0:
                Q_range = 1e-3

            range_max = max(I_range, Q_range) / 2 * 2.4

            ax.set_xlim(-range_max, range_max)
            ax.set_ylim(-range_max, range_max)

            I_ticks = np.linspace(-range_max, range_max, 10) * 1e5
            Q_ticks = np.linspace(-range_max, range_max, 10) * 1e5

            ax.set_xticks(Q_ticks / 1e5)
            ax.set_xticklabels([f'{tick:.0f}' for tick in Q_ticks], color='white')
            ax.set_yticks(I_ticks / 1e5)
            ax.set_yticklabels([f'{tick:.0f}' for tick in I_ticks], color='white')

            ax.tick_params(axis='x', pad=12)

            if grids:
                ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
                for tick in Q_ticks / 1e5:
                    ax.axvline(tick, color='gray', linestyle='--', linewidth=0.5)
                for tick in I_ticks / 1e5:
                    ax.axhline(tick, color='gray', linestyle='--', linewidth=0.5)

        elif data_format == 'magnitude-phase':
            I_max = max(xdata) if max(xdata) > 0 else 1e-5

            Q_min, Q_max = -math.pi, math.pi

            ax.set_xlim(0, I_max)
            ax.set_ylim(Q_min, Q_max)

            I_ticks = np.linspace(0, I_max, 10)
            Q_ticks = np.linspace(Q_min, Q_max, 10)

            ax.set_xticks(I_ticks)
            ax.set_xticklabels([f'{tick*1e5:.2f}' for tick in I_ticks], color='white')
            ax.set_yticks(Q_ticks)
            ax.set_yticklabels([f'{tick:.2f}' for tick in Q_ticks], color='white')

            if grids:
                ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
                for tick in ax.get_xticks():
                    ax.axvline(tick, color='gray', linestyle='--', linewidth=0.5)
                for tick in ax.get_yticks():
                    ax.axhline(tick, color='gray', linestyle='--', linewidth=0.5)

        fig.canvas.draw_idle()

        return sc,

    ani = animation.FuncAnimation(fig, update, frames=len(I_data), interval=interval, blit=True)
    plt.show()

def scatterplot(iq, data_format, grids=False):
    I_data, Q_data = process_data(iq, data_format)

    fig, ax = plt.subplots(dpi=300)
    fig.set_facecolor('black')
    ax.scatter(I_data, Q_data, s=0.1, c='lime')
    ax.set_facecolor('black')

    if data_format == 'real-imag':
        I_min, I_max = min(I_data), max(I_data)
        Q_min, Q_max = min(Q_data), max(Q_data)
        I_range = I_max - I_min
        Q_range = Q_max - Q_min

        if I_range == 0:
            I_range = 1e-3
        if Q_range == 0:
            Q_range = 1e-3

        range_max = max(I_range, Q_range) / 2 * 2.4

        ax.set_xlim(-range_max, range_max)
        ax.set_ylim(-range_max, range_max)

        I_ticks = np.linspace(-range_max, range_max, 10) * 1e5
        Q_ticks = np.linspace(-range_max, range_max, 10) * 1e5

        ax.set_xticks(Q_ticks / 1e5)
        ax.set_xticklabels([f'{tick:.0f}' for tick in Q_ticks], color='white')
        ax.set_yticks(I_ticks / 1e5)
        ax.set_yticklabels([f'{tick:.0f}' for tick in I_ticks], color='white')

        ax.tick_params(axis='x', pad=12)

        if grids:
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
            for tick in Q_ticks / 1e5:
                ax.axvline(tick, color='gray', linestyle='--', linewidth=0.5)
            for tick in I_ticks / 1e5:
                ax.axhline(tick, color='gray', linestyle='--', linewidth=0.5)

        ax.set_title('I/Q Scatter Plot (Real-Imag)', color='white', y=1.07)
        ax.set_xlabel('Q_data (1x10⁵)', color='white', labelpad=1)
        ax.set_ylabel('I_data (1x10⁵)', color='white', labelpad=1)

    elif data_format == 'magnitude-phase':
        I_max = max(I_data) if max(I_data) > 0 else 1e-5
        Q_min, Q_max = -math.pi, math.pi

        ax.set_xlim(0, I_max)
        ax.set_ylim(Q_min, Q_max)

        I_ticks = np.linspace(0, I_max, 10)
        Q_ticks = np.linspace(Q_min, Q_max, 10)

        ax.set_xticks(I_ticks)
        ax.set_xticklabels([f'{tick*1e5:.2f}' for tick in I_ticks], color='white')
        ax.set_yticks(Q_ticks)
        ax.set_yticklabels([f'{tick:.2f}' for tick in Q_ticks], color='white')

        if grids:
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
            for tick in ax.get_xticks():
                ax.axvline(tick, color='gray', linestyle='--', linewidth=0.5)
            for tick in ax.get_yticks():
                ax.axhline(tick, color='gray', linestyle='--', linewidth=0.5)

        ax.set_title('I/Q Scatter Plot (Magnitude-Phase)', color='white', y=1.07)
        ax.set_xlabel('Magnitude (10⁵)', color='white', labelpad=1)
        ax.set_ylabel('Phase', color='white', labelpad=1)

    ax.tick_params(colors='white', which='both')

    plt.show()