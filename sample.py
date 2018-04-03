# from ILAMB.Variable import Variable

# # pick LE and a model

# obs = Variable(...)
# mod = Variable(...)

# def TimeSeries(obs,mod,site_id):
#     pass
    

# def Wavelet(obs,mod,site_id):
#     pass

# for site in sites:
#     Timeseries(obs,mod,site)
#     Wavelet   (obs,mod,site)   
from ILAMB.Variable import Variable
from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
from taylorDiagram import plot_daylor_graph
import numpy as np
import waipy
import matplotlib.pyplot as plt
from PyEMD import EEMD
from hht import hht
from hht import plot_imfs
from hht import plot_frequency

data = '/Users/lli51/Downloads/ILAMB_sample/DATA/rsus/CERES/rsus_0.5x0.5.nc'
model = '/Users/lli51/Downloads/ILAMB_sample/MODELS/CLM40cn/rsus/rsus_Amon_CLM40cn_historical_r1i1p1_185001-201012.nc'
# obs=Variable(filename=data, variable_name='rsus')
# mod=Variable(filename=model, variable_name='rsus')
m = ModelResult('/Users/lli51/Downloads/ILAMB_sample/MODELS/CLM40cn/', modelname='CLM40cn')
c = Confrontation(source=data, name='CERES',variable='rsus')
obs, mod = c.stageData(m)

def Plot_TimeSeries(obs,mod,site_id):
    print('Process on TimeSeries ' + 'No.' + str(site_id) + '!')
    t = mod.time
    x = (obs.extractDatasites(lat=obs.lat, lon=obs.lat)).data[:,site_id]
    y = (mod.extractDatasites(lat=obs.lat, lon=obs.lat)).data[:,site_id]
    mods = []
    mods.append(y)
    fig0 = plt.figure(figsize=(5, 5))
    plt.suptitle('Time series')
    ax0 = fig0.add_subplot(2,1,1)
    fig0, samples0 = plot_daylor_graph(x, mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)
    ax0.plot(t, x, 'k')
    ax0.plot(t, y)

def Plot_PDF_CDF(obs,mod,site_id):
    print('Process on PDF&CDF ' + 'No.' + str(site_id) + '!')
    t = mod.time
    x = (obs.extractDatasites(lat=obs.lat, lon=obs.lat)).data[:, site_id]
    y = (mod.extractDatasites(lat=obs.lat, lon=obs.lat)).data[:, site_id]
    fig0 = plt.figure(figsize=(5, 5))
    plt.suptitle('PDF and CDF')
    ax0 = fig0.add_subplot(2, 1, 1)
    ax1 = fig0.add_subplot(2, 1, 2)

    h_obs_sorted = np.ma.sort(x).compressed()
    p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
    p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)))  # bin it into n = N/10 bins
    x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
    ax0.plot(x_h, p_h / float(sum(p_h)), 'k-', label='Observed')
    ax1.plot(h_obs_sorted, p1_data, 'k-', label='Observed')

    h_obs_sorted = np.ma.sort(y).compressed()
    p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
    p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)))  # bin it into n = N/10 bins
    x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
    ax0.plot(x_h, p_h / float(sum(p_h)))
    ax1.plot(h_obs_sorted, p1_data)

def Plot_Wavelet(obs,mod,site_id):
    print('Process on Wavelet ' + 'No.' + str(site_id) + '!')
    time_data = mod.time
    data = (obs.extractDatasites(lat=obs.lat, lon=obs.lat)).data[:, site_id]
    y = (mod.extractDatasites(lat=obs.lat, lon=obs.lat)).data[:, site_id]
    fig3 = plt.figure(figsize=(8, 8))
    result = waipy.cwt(data, 1, 1, 0.125, 2, 4 / 0.125, 0.72, 6, mother='Morlet', name='Obs')
    waipy.wavelet_plot('Obs', time_data, data, 0.03125, result, fig3, unit=obs.unit)

def Plot_IMF(obs,mod,site_id):
    print('Process on IMF ' + 'No.' + str(site_id) + '!')
    data = (obs.extractDatasites(lat=obs.lat, lon=obs.lat)).data[:, site_id]
    time = mod.time[~data.mask]
    y = (mod.extractDatasites(lat=obs.lat, lon=obs.lat)).data[:, site_id]
    d_mod = []
    d_mod.append(y)
    fig1 = plt.figure(figsize=(4 * (len(d_mod) + 1), 8))
    fig2 = plt.figure(figsize=(4 * (len(d_mod) + 1), 8))
    fig3 = plt.figure(figsize=(5, 5))
    fig1.subplots_adjust(wspace=0.5, hspace=0.3)
    fig2.subplots_adjust(wspace=0.5, hspace=0.3)
    eemd = EEMD(trials=5)
    imfs = eemd.eemd(data.compressed())
    fig3, freq = hht(data.compressed(), imfs, time, 1, fig3)
    if len(imfs) >= 1:
        fig1 = plot_imfs(data.compressed(), imfs, time_samples=time, fig=fig1, no=1, m=len(d_mod))
        fig2 = plot_frequency(data.compressed(), freq.T, time_samples=time, fig=fig2, no=1, m=len(d_mod))


# Plot_TimeSeries(obs,mod,2)
# Plot_PDF_CDF(obs,mod,2)
# Plot_Wavelet(obs,mod,2)
Plot_IMF(obs,mod,2)
plt.show()
