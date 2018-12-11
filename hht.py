from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import FuncFormatter
from PyEMD import EEMD

col = ['plum', 'darkorchid', 'blue', 'navy', 'deepskyblue', 'darkcyan', 'seagreen', 'darkgreen',
       'olivedrab', 'gold', 'tan', 'red', 'palevioletred', 'm', 'plum']
start_year = 1991
end_year = 2014

def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fY' % (x/365.0)


# col = ['palevioletred', 'm', 'plum', 'darkorchid', 'blue', 'navy', 'deepskyblue', 'darkcyan', 'seagreen', 'darkgreen',
#        'olivedrab', 'gold', 'tan', 'red']


def plot_imfs(signal, imfs, time_samples=None, fig=None, no=1, m=1):
    ''' Author jaidevd https://github.com/jaidevd/pyhht/blob/dev/pyhht/visualization.py Original function from pyhht, but without plt.show()'''
    n_imfs = imfs.shape[0]
    # print(np.abs(imfs[:-1, :]))
    # axis_extent = max(np.max(np.abs(imfs[:-1, :]), axis=0))
    # Plot original signal
    if fig == None:
        fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(n_imfs + 1, m+1, no)

    if no == 1:
        ax.plot(time_samples, signal, color='k')
    else:
        ax.plot(time_samples, signal, color=col[no-1])

    ax.set_yticks([np.ma.median(signal), np.ma.max(np.ma.fix_invalid(signal))])
    ax.set_yticklabels([int(np.ma.median(signal)), int(np.ma.max(np.ma.fix_invalid(signal)))])

    d_d_obs = np.asarray([str(start_year + int(x) / 365) + (
    '0' + str(int(x) % 365 / 31 + 1) if int(x) % 365 / 31 < 9 else str(int(x) % 365 / 31 + 1)) for x in time_samples])

    # ax.xaxis.set_ticks(
    #     [time_samples[0], time_samples[len(time_samples) / 5], time_samples[2 * len(time_samples) / 5], time_samples[3 * len(time_samples) / 5],
    #      time_samples[4 * len(time_samples) / 5]])
    # ax.set_xticklabels(
    #     [d_d_obs[0], d_d_obs[len(d_d_obs) / 5], d_d_obs[2 * len(d_d_obs) / 5], d_d_obs[3 * len(d_d_obs) / 5],
    #      d_d_obs[4 * len(d_d_obs) / 5]])

    # ax.axis([time_samples[0], time_samples[-1], signal.min(), signal.max()])
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('Signal', rotation=90)
    if no == 1:
        ax.set_title('Observation')
    else:
        ax.set_title('Model' + str(no-1))

    # Plot the IMFs
    for i in range(n_imfs - 1):
        # print(i + 2)
        ax = fig.add_subplot(n_imfs + 1, m+1, (i+1)*(m+1) + no)
        if no == 1:
            ax.plot(time_samples, imfs[i, :], color='k')
        else:
            ax.plot(time_samples, imfs[i, :], color=col[no-1])

        # ax.xaxis.set_ticks(
        #     [time_samples[0], time_samples[len(time_samples) / 5], time_samples[2 * len(time_samples) / 5],
        #      time_samples[3 * len(time_samples) / 5],
        #      time_samples[4 * len(time_samples) / 5]])
        # ax.set_xticklabels(
        #     [d_d_obs[0], d_d_obs[len(d_d_obs) / 5], d_d_obs[2 * len(d_d_obs) / 5], d_d_obs[3 * len(d_d_obs) / 5],
        #      d_d_obs[4 * len(d_d_obs) / 5]])
        # ax.axis([time_samples[0], time_samples[-1], -axis_extent, axis_extent])
        ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                       labelbottom=False)
        ax.grid(False)
        ax.set_ylabel('imf' + str(i + 1), rotation=90)
        ax.yaxis.tick_right()
        # formatter = FuncFormatter(millions)
        # ax.yaxis.set_major_formatter(formatter)
        if len(np.ma.fix_invalid(imfs[i, :]).compressed())>0:
            # print(np.ma.max(np.ma.fix_invalid(imfs[i, :])))
            # print(np.ma.median(imfs[i, :]))
            ax.set_yticks([np.ma.median(imfs[i, :]), np.ma.max(np.ma.fix_invalid(imfs[i, :]))])
            ax.set_yticklabels([int(np.ma.median(imfs[i, :])), int(np.ma.max(np.ma.fix_invalid(imfs[i, :])))])

    # Plot the residue
    ax = fig.add_subplot(n_imfs + 1, m+1, n_imfs*(m+1) + no)
    ax.plot(time_samples, imfs[-1, :], 'r')

    ax.xaxis.set_ticks(
        [time_samples[0], time_samples[2 * len(time_samples) / 5],
         time_samples[4 * len(time_samples) / 5]])
    ax.set_xticklabels(
        [d_d_obs[0], d_d_obs[2 * len(d_d_obs) / 5],
         d_d_obs[4 * len(d_d_obs) / 5]])

    ax.axis('tight')
    # ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
    #                labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('res.', rotation=90)
    # ax.set_xticks([time_samples[i*len(time_samples)/5] for i in range(5)])
    # ax.set_xticklabels([int(time_samples[i*len(time_samples)/5]) for i in range(5)])
    ax.set_yticklabels([])

    # plt.xticks(fontsize=3, rotation=45)

    plt.tight_layout()
    return fig


def plot_frequency(signal, imfs, time_samples=None, fig=None, no=1, m=1):
    ''' Author jaidevd https://github.com/jaidevd/pyhht/blob/dev/pyhht/visualization.py '''
    '''Original function from pyhht, but without plt.show()'''
    n_imfs = imfs.shape[0]
    # print(np.abs(imfs[:-1, :]))
    # axis_extent = max(np.max(np.abs(imfs[:-1, :]), axis=0))
    # Plot original signal
    if fig == None:
        fig = plt.figure(figsize=(5, 5))
    d_d_obs = np.asarray([str(start_year + int(x) / 365) + (
    '0' + str(int(x) % 365 / 31 + 1) if int(x) % 365 / 31 < 9 else str(int(x) % 365 / 31 + 1)) for x in time_samples])

    ax = fig.add_subplot(n_imfs + 1, m + 1, no)
    if no == 1:
        ax.plot(time_samples, signal, color='k')
    else:
        ax.plot(time_samples, signal, color=col[no-1])
    # ax.xaxis.set_ticks(
    #     [time_samples[0], time_samples[len(time_samples) / 5], time_samples[2 * len(time_samples) / 5], time_samples[3 * len(time_samples) / 5],
    #      time_samples[4 * len(time_samples) / 5]])
    # ax.set_xticklabels(
    #     [d_d_obs[0], d_d_obs[len(d_d_obs) / 5], d_d_obs[2 * len(d_d_obs) / 5], d_d_obs[3 * len(d_d_obs) / 5],
    #      d_d_obs[4 * len(d_d_obs) / 5]])
    # ax.axis([time_samples[0], time_samples[-1], signal.min(), signal.max()])
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('Signal', rotation=90)
    if no == 1:
        ax.set_title('Observation')
    else:
        ax.set_title('Model' + str(no - 1))

    # Plot the IMFs
    for i in range(n_imfs - 1):
        # print(i + 2)
        ax = fig.add_subplot(n_imfs + 1, m + 1, (i + 1) * (m + 1) + no)
        if no == 1:
            ax.plot(time_samples, imfs[i, :], color='k')
        else:
            ax.plot(time_samples, imfs[i, :], color=col[no-1])
        # ax.xaxis.set_ticks(
        #     [time_samples[0], time_samples[len(time_samples) / 5], time_samples[2 * len(time_samples) / 5],
        #      time_samples[3 * len(time_samples) / 5],
        #      time_samples[4 * len(time_samples) / 5]])
        # ax.set_xticklabels(
        #     [d_d_obs[0], d_d_obs[len(d_d_obs) / 5], d_d_obs[2 * len(d_d_obs) / 5], d_d_obs[3 * len(d_d_obs) / 5],
        #      d_d_obs[4 * len(d_d_obs) / 5]])
        # ax.axis([time_samples[0], time_samples[-1], -axis_extent, axis_extent])
        ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                       labelbottom=False)

        ax.yaxis.tick_right()
        formatter = FuncFormatter(millions)
        ax.yaxis.set_major_formatter(formatter)
        plt.tick_params(axis='right', which='minor', labelsize=2)
        ax.grid(False)
        if len(np.ma.fix_invalid(imfs[i, :]).compressed())>0:
            # print(np.ma.max(np.ma.fix_invalid(imfs[i, :])))
            # print(np.ma.median(imfs[i, :]))
            if np.ma.median(imfs[i, :])<=1:
                ax.set_yticks([1, np.ma.max(np.ma.fix_invalid(imfs[i, :]))])
                ax.set_yticklabels([str(1), str(int(np.ma.max(np.ma.fix_invalid(imfs[i, :]))))+'D' if int(np.ma.max(np.ma.fix_invalid(imfs[i, :])))<365 else str(int(np.ma.max(np.ma.fix_invalid(imfs[i, :]))/365))+'Y'])
            else:
                ax.set_yticks([np.ma.median(imfs[i, :]), np.ma.max(np.ma.fix_invalid(imfs[i, :]))])
                ax.set_yticklabels([str(int(np.ma.median(imfs[i, :])))+'D' if int(np.ma.median(imfs[i, :]))<365 else str(int(np.ma.median(imfs[i, :]))/365)+'Y',str(int(np.ma.max(np.ma.fix_invalid(imfs[i, :]))))+'D' if int(np.ma.max(np.ma.fix_invalid(imfs[i, :])))<365 else str(int(np.ma.max(np.ma.fix_invalid(imfs[i, :]))/365))+'Y'])
            ax.set_ylim(0, np.ma.max(np.ma.fix_invalid(imfs[i, :])))
        ax.set_ylabel('imf' + str(i + 1), rotation=90)

    # Plot the residue
    ax = fig.add_subplot(n_imfs + 1, m + 1, n_imfs * (m + 1) + no)
    ax.plot(time_samples, imfs[-1, :], 'r')
    ax.xaxis.set_ticks(
        [time_samples[0], time_samples[2 * len(time_samples) / 5],
         time_samples[4 * len(time_samples) / 5]])
    ax.set_xticklabels(
        [d_d_obs[0], d_d_obs[2 * len(d_d_obs) / 5],
         d_d_obs[4 * len(d_d_obs) / 5]])
    ax.axis('tight')
    # ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
    #                labelbottom=False, labelsize=2)
    # plt.xticks(fontsize=3, rotation=45)

    ax.grid(False)
    ax.set_ylabel('res.', rotation=90)
    # ax.set_xticks([time_samples[i*len(time_samples)/5] for i in range(5)])
    # ax.set_xticklabels([int(time_samples[i*len(time_samples)/5]) for i in range(5)])
    ax.set_yticklabels([])

    plt.tight_layout()

    return fig


def hilb(s, unwrap=False):
    """
    Performs Hilbert transformation on signal s.
    Returns amplitude and phase of signal.
    Depending on unwrap value phase can be either
    in range [-pi, pi) (unwrap=False) or
    continuous (unwrap=True).
    """
    from scipy.signal import hilbert
    H = hilbert(s)
    amp = np.abs(H)
    phase = np.arctan2(H.imag, H.real)
    if unwrap: phase = np.unwrap(phase)

    return amp, phase


def FAhilbert(imfs, dt):
    n_imfs = imfs.shape[0]
    f = []
    a = []
    for i in range(n_imfs):
        # upper, lower = pyhht.utils.get_envelops(imfs[i, :])
        inst_imf = imfs[i, :]  # /upper
        inst_amp, phase = hilb(inst_imf, unwrap=True)
        inst_freq = (2 * math.pi) / np.diff(phase)  #

        inst_freq = np.insert(inst_freq, len(inst_freq), inst_freq[-1])
        inst_amp = np.insert(inst_amp, len(inst_amp), inst_amp[-1])

        f.append(inst_freq)
        a.append(inst_amp)

    return np.asarray(f).T, np.asarray(a).T


def hht(data, imfs, time, no, fig3, freqsol=33, timesol=50):
    #   freqsol give frequency - axis resolution for hilbert - spectrum
    #   timesol give time - axis resolution for hilbert - spectrum
    t0 = time[0]
    t1 = time[-1]
    dt = (t1 - t0) / (len(time) - 1)

    freq, amp = FAhilbert(imfs, dt)

    tw = t1 - t0
    bins = np.linspace(0, 12, freqsol)  # np.logspace(0, 10, freqsol, base=2.0)
    p = np.digitize(freq, 2 ** bins)
    t = np.ceil((timesol - 1) * (time - t0) / tw)
    t = t.astype(int)

    hilbert_spectrum = np.zeros([timesol, freqsol])
    for i in range(len(time)):
        for j in range(imfs.shape[0] - 1):
            if p[i, j] >= 0 and p[i, j] < freqsol:
                hilbert_spectrum[t[i], p[i, j]] += amp[i, j]

    hilbert_spectrum = abs(hilbert_spectrum)
    d_d_obs = np.asarray([str(start_year + int(x) / 365) + (
    '0' + str(int(x) % 365 / 31 + 1) if int(x) % 365 / 31 < 9 else str(int(x) % 365 / 31 + 1)) for x in time])

    ax = fig3.gca()
    c = ax.contourf(np.linspace(t0, t1, timesol), bins,
                    hilbert_spectrum.T)  # , colors=('whites','lategray','navy','darkgreen','gold','red')
    ax.invert_yaxis()
    ax.set_yticks(np.linspace(1, 11, 11))
    Yticks = [float(math.pow(2, p)) for p in np.linspace(1, 11, 11)]  # make 2^periods
    ax.set_yticklabels(Yticks)
    ax.set_xlabel('Time', fontsize=8)
    ax.set_ylabel('Period(days)', fontsize=8)

    ax.xaxis.set_ticks(
        [time[0], time[len(time) / 5], time[2 * len(time) / 5], time[3 * len(time) / 5],
         time[4 * len(time) / 5]])
    ax.set_xticklabels(
        [d_d_obs[0], d_d_obs[len(d_d_obs) / 5], d_d_obs[2 * len(d_d_obs) / 5], d_d_obs[3 * len(d_d_obs) / 5],
         d_d_obs[4 * len(d_d_obs) / 5]])
    position = fig3.add_axes([0.2, -0., 0.6, 0.01])
    cbar = plt.colorbar(c, cax=position, orientation='horizontal')
    cbar.set_label('Power')

    return fig3, freq

    # plt.show()

#
# f = Dataset('/Users/lli51/Documents/ornl_project/171002_parmmods_monthly_obs.nc')
# # f = Dataset('/Users/lli51/Documents/ornl_project/171002_parmmods_daily_obs.nc')
#
# fsh = f.variables['FSH']
# time = f.variables['time']
# one_site = np.ma.masked_invalid(fsh[0,:])
# time = time[~one_site.mask]
# data = one_site.compressed()
#
# hht(data, time)
