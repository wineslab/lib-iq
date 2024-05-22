import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animated_scatterplot(iq, interval=100, window=50, grids=False):
    iq_sample = np.array(iq)
    I_data = iq_sample[:, 0]
    Q_data = iq_sample[:, 1]

    fig, ax = plt.subplots(dpi=300)
    fig.set_facecolor('black')
    ax.set_facecolor('black')

    if grids:
        ax1 = fig.add_subplot(111, polar=True, frame_on=False)
        ax1.set_rticks([])
        ax1.xaxis.grid(True)
        ax1.yaxis.grid(False)
        ax1.set_facecolor('black')
        ax1.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
        ax1.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], color='white')
        ax1.tick_params(axis='x', pad=-3)

    sc = ax.scatter([], [], s=1, c='lime')

    ax.set_title('I/Q Scatter Plot', color='white', y=1.07)
    ax.set_xlabel('Q_data (1x10⁵)', color='white', labelpad=1)
    ax.set_ylabel('I_data (1x10⁵)', color='white', labelpad=1)

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

        fig.canvas.draw_idle()

        return sc,

    ani = animation.FuncAnimation(fig, update, frames=len(I_data), interval=interval, blit=True)
    plt.show()

def scatterplot(iq, grids=False):
    iq_sample = np.array(iq)
    I_data = iq_sample[:, 0]
    Q_data = iq_sample[:, 1]

    fig, ax = plt.subplots(dpi=300)
    fig.set_facecolor('black')
    ax.scatter(I_data, Q_data, s=0.1, c='lime')
    ax.set_facecolor('black')

    I_min, I_max = I_data.min(), I_data.max()
    Q_min, Q_max = Q_data.min(), Q_data.max()
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

    ax.set_title('I/Q Scatter Plot', color='white', y=1.07)
    ax.set_xlabel('Q_data (1x10⁵)', color='white', labelpad=1)
    ax.set_ylabel('I_data (1x10⁵)', color='white', labelpad=1)
    ax.tick_params(colors='white', which='both')

    plt.show()
