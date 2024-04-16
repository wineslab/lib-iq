import numpy as np
import matplotlib.pyplot as plt

def scatterplot(iq):
    iq_sample = np.array(iq)
    I_data = iq_sample[:,0]
    Q_data = iq_sample[:,1]

    IQ_data = I_data + 1j*Q_data

    fig, ax = plt.subplots(dpi=300)
    ax.scatter(IQ_data.real, IQ_data.imag, s=0.1, c='lime')
    ax.set_facecolor('black')

    plt.title('I/Q Scatter Plot', color='white')
    plt.xlabel('Real part', color='white')
    plt.ylabel('Imaginary part', color='white')

    plt.tick_params(colors='white', which='both')

    plt.show()