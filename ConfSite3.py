from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
from ILAMB.Variable import Variable
from ILAMB.Confrontation import getVariableList
from ILAMB.Regions import Regions
import ILAMB.ilamblib as il
import ILAMB.Post as post
from netCDF4 import Dataset, num2date, date2num, date2index
import matplotlib.pyplot as plt, numpy as np
from matplotlib.collections import LineCollection
import matplotlib.ticker as mticker

# from matplotlib.dates import date2num
import matplotlib
from scipy import stats, linalg
from scipy import signal

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import proj3d

import pandas as pd
import seaborn as sns;

sns.set()
import os, glob, math, calendar, PIL
from PIL import Image
import collections
import math
# self-package
from taylorDiagram import plot_Taylor_graph_time_basic
from taylorDiagram import plot_Taylor_graph_season_cycle
from taylorDiagram import plot_Taylor_graph
from taylorDiagram import plot_Taylor_graph_day_cycle
from taylorDiagram import plot_Taylor_graph_three_cycle
import waipy

import matplotlib.pyplot as plt

SMALL_SIZE = 17
MEDIUM_SIZE = 22
BIGGER_SIZE = 22
TitleLocation = 0.95
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.style.use('seaborn-colorblind')
ColorBook = ['plum', 'darkorchid', 'blue', 'navy', 'deepskyblue', 'darkcyan', 'seagreen', 'darkgreen',
       'olivedrab', 'gold', 'tan', 'red', 'palevioletred', 'm', 'plum']
# To add IMF analysis, refer to https://github.com/liangliannie/hht-spectrum
# from PyEMD import EEMD
# from hht import hht
# from hht import plot_imfs
# from hht import plot_frequency

def BaseMap_Plot(LatList, LonList, SiteNameList, RegionName='Region', MapLatL=-90, MapLatR=90, MapLonL=-180,
                MapLonR=180):
    """A function to plot BaseMap for different Region

    :param LatList: List of Lat
    :param LonList: List of Lon
    :param SiteNameList: Name of each Site
    :param MapLatL: Size of the Map, use default to get the earth map
    :param MapLatR: Size of the Map, use default to get the earth map
    :param MapLonL: Size of the Map, use default to get the earth map
    :param MapLonR: Size of the Map, use default to get the earth map

    :returns:  Figure or return NULL
    """
    fig = plt.figure(figsize=(15, 7))
    m = Basemap(projection='mill', llcrnrlat=MapLatL, urcrnrlat=MapLatR, llcrnrlon=MapLonL, urcrnrlon= MapLonR, resolution='c')
    m.drawcoastlines()
    # m.drawcountries()
    # m.drawstates()
    # m.fillcontinents(color='#04BAE3', lake_color='#FFFFFF')
    # m.drawmapboundary(fill_color='#FFFFFF')

    for i, (lat, lon) in enumerate(zip(LatList, LonList)):
        x, y = m(lon, lat)
        m.plot(x, y, 'o', markersize=12, alpha=.5, label=SiteNameList[i])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    # plt.annotate('Annotate', xy=(0,0.01), xycoords='axes fraction')
    plt.suptitle(RegionName,y=TitleLocation)

    return fig


def TimeSeriesOneModel_Plot(ObsData, ModData, FigTitle= None, XLabel='Obs.Name', YLabel='Obs.Unit', ModLegend='Mod',
                            FigName=None, fig=None):
    """A function to plot timeseries figures

    :param ObsData: Pandas.DataFrame[] of one column (defined observation data of one site)
    :param ModData: Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param FigTitle: Title of the figure
    :param XLabel: X label of the figure (default OBS.Name)
    :param YLabel: Y label of the figure (default OBS.Unit)
    :param ModLegend: The Name of Model Legend
    :param FigName: FigName including the relative path to save the Fig.

    :returns:  Figure or return NULL
    """
    Dates = ObsData.index
    if fig is None:
        fig = plt.figure(figsize=(15, 7))
    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=2)
    if FigTitle is not None:
        plt.suptitle(FigTitle, y=TitleLocation)
    ax1.scatter(Dates, np.ones(len(Dates)), c=~pd.isna(ObsData), marker='s', cmap='binary', s=1)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set_yticklabels([])
    plt.ylabel('Existence')
    ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, colspan=2, sharex=ax1)
    ax2.plot(Dates, ObsData, 'k-', label='OBS', linewidth=1.0)
    ax2.plot(Dates, np.ma.masked_where(pd.isna(ObsData), ModData), '-', label=ModLegend, linewidth=1.0)
    # ax2.fill_between(Dates, ObsData, ModData, where=(ObsData > ModData), color='r')
    # ax2.fill_between(Dates, ObsData, ModData, where=(ModData < ObsData), color='g')
    plt.ylabel(YLabel)
    plt.xlabel(XLabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)

    if FigName is not None:
        fig.savefig(FigName)
    else:
        return fig


def CyclesOneModel_Plot(ObsData0, ModData0, FrequencyRule=None, FigTitle='SiteName', XLabel='Hours of A Day',
                        YLabel='Obs.Name(Obs.Unit)', ModLegend='MOD',
                        FigName=None):
    """A function to plot Cycles figures

    :param ObsData0: Pandas.DataFrame[] of one column (defined observation data of one site)
    :param ModData0: Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param FrequencyRule: Define the Cycles, e.g. 'H' is Hourly of a Day; 'D' is Day of A year; 'M' is Month of A year;
    :param FigTitle: Title of the figure
    :param XLabel: X label of the figure (default OBS.Name)
    :param YLabel: Y label of the figure (default OBS.Unit)
    :param ModLegend: The Name of Model Legend
    :param FigName: FigName including the relative path to save the Fig.

    :returns:  Figure or return NULL
    """

    # import time
    # timestamp = time.strftime('%H:%M:%S')
    # timestamp
    if FrequencyRule is not None:
        if FrequencyRule == 'S':
            ObsData = ObsData0.resample('M').mean()
            ModData = ModData0.resample('M').mean()
        else:
            ObsData = ObsData0.resample(FrequencyRule).mean()
            ModData = ModData0.resample(FrequencyRule).mean()


    if FrequencyRule == 'H':
        obs_Cycle_mean = [ObsData[ObsData.index.hour == i].mean() for i in range(24)]
        obs_Cycle_var = [ObsData[ObsData.index.hour == i].std() for i in range(24)]
        mod_Cycle_mean = [ModData[ModData.index.hour == i].mean() for i in range(24)]
        mod_Cycle_var = [ModData[ModData.index.hour == i].std() for i in range(24)]
    elif FrequencyRule == 'D':
        obs_Cycle_mean = [ObsData[ObsData.index.dayofyear == i].mean() for i in range(365)]
        obs_Cycle_var = [ObsData[ObsData.index.dayofyear == i].std() for i in range(365)]
        mod_Cycle_mean = [ModData[ModData.index.dayofyear == i].mean() for i in range(365)]
        mod_Cycle_var = [ModData[ModData.index.dayofyear == i].std() for i in range(365)]
    elif FrequencyRule == 'M':
        obs_Cycle_mean = [ObsData[ObsData.index.month == i].mean() for i in range(12)]
        obs_Cycle_var = [ObsData[ObsData.index.month == i].std() for i in range(12)]
        mod_Cycle_mean = [ModData[ModData.index.month == i].mean() for i in range(12)]
        mod_Cycle_var = [ModData[ModData.index.month == i].std() for i in range(12)]

    elif FrequencyRule == 'S':
        data1 = [ObsData[ObsData.index.month == i].mean() for i in range(12)]
        data2 = [ObsData[ObsData.index.month == i].std() for i in range(12)]
        data3 = [ModData[ModData.index.month == i].mean() for i in range(12)]
        data4 = [ModData[ModData.index.month == i].std() for i in range(12)]

        obs_Cycle_mean = [np.ma.mean([data1[11], data1[0], data1[1]]), np.ma.mean([data1[4], data1[2], data1[3]]),
                 np.ma.mean([data1[7], data1[5], data1[6]]), np.ma.mean([data1[10], data1[8], data1[9]])]
        obs_Cycle_var = [np.ma.mean([data2[11], data2[0], data2[1]]), np.ma.mean([data2[4], data2[2], data2[3]]),
                 np.ma.mean([data2[7], data2[5], data2[6]]), np.ma.mean([data2[10], data2[8], data2[9]])]
        mod_Cycle_mean = [np.ma.mean([data3[11], data3[0], data3[1]]), np.ma.mean([data3[4], data3[2], data3[3]]),
                          np.ma.mean([data3[7], data3[5], data3[6]]), np.ma.mean([data3[10], data3[8], data3[9]])]
        mod_Cycle_var = [np.ma.mean([data4[11], data4[0], data4[1]]), np.ma.mean([data4[4], data4[2], data4[3]]),
                          np.ma.mean([data4[7], data4[5], data4[6]]), np.ma.mean([data4[10], data4[8], data4[9]])]

    fig = plt.figure(figsize=(15, 7))

    CycleTime = range(len(obs_Cycle_mean))

    if FigTitle is not None:
        plt.suptitle(FigTitle, y=TitleLocation)

    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=2)
    ax1.scatter(CycleTime, np.ones(len(CycleTime)), c=pd.isna(obs_Cycle_mean), marker='s', cmap='binary', s=1)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set_yticklabels([])
    plt.ylabel('Existence')

    ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, colspan=2, sharex=ax1)
    ax2.plot(CycleTime, obs_Cycle_mean, 'k-', label='OBS', linewidth=1.0)
    ax2.plot(CycleTime, np.ma.masked_where(pd.isna(obs_Cycle_mean), mod_Cycle_mean), '-', label=ModLegend,
             linewidth=1.0)
    ax2.fill_between(CycleTime, np.asarray(obs_Cycle_mean) - np.asarray(obs_Cycle_var),
                     np.asarray(obs_Cycle_mean) + np.asarray(obs_Cycle_var), alpha=0.2, edgecolor='#1B2ACC',
                     linewidth=0.5, linestyle='dashdot', antialiased=True)

    ax2.fill_between(CycleTime, np.asarray(mod_Cycle_mean) - np.asarray(mod_Cycle_var),
                     np.asarray(mod_Cycle_mean) + np.asarray(mod_Cycle_var), alpha=0.2, edgecolor='#1B2ACC',
                     linewidth=0.5, linestyle='dashdot', antialiased=True)

    if XLabel[0] == 'H':
        plt.xticks(range(24), [str(i) + '' for i in range(24)])
    elif XLabel[0] == 'D':
        plt.xticks(range(0, 365, 60), [str(i) for i in range(0, 365, 60)])
    elif XLabel[0] == 'M':
        plt.xticks(range(13), calendar.month_name[1:13], rotation=20)
    elif XLabel[0] == 'S':
        plt.xticks(range(4), ('DJF', 'MAM', 'JJA', 'SON'))

    plt.ylabel(YLabel)
    plt.xlabel(XLabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)

    if FigName is not None:
        fig.savefig(FigName)
    else:
        return fig


def PDFCDFOneModel_Plot(obs_one, mod_one, FigTitle='SiteName', XLabel='Obs.Name(Obs.Unit)', YLabel=['Density Distribution(PDF)', 'Cumulative Distribution(CDF)'], ModLegend='MOD',FigName=None):
    """A function to plot timeseries figures

    :param ObsData: Pandas.DataFrame[] of one column (defined observation data of one site)
    :param ModData: Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param FigTitle: Title of the figure
    :param XLabel: X label of the figure (default OBS.Name)
    :param YLabel: Y label of the figure (default OBS.Unit)
    :param ModLegend: The Name of Model Legend
    :param FigName: FigName including the relative path to save the Fig.

    :returns:  Figure or return NULL
    """
# from scipy.stats import gaussian_kde
    fig = plt.figure(figsize=(9, 9))
    plt.suptitle(FigTitle, y=TitleLocation+0.2)
    ax = plt.subplot2grid((4,2), (0,0), rowspan=2, colspan=2)
    sns.kdeplot(obs_one.dropna(), shade=True, ax=ax,  color="k", label='OBS')
    sns.kdeplot(mod_one[~pd.isna(obs_one)].dropna(), shade=True, ax=ax, label=ModLegend)
    # sns.kdeplot(mod_one.dropna(), shade=True, ax=ax, label='MOD(PDF)')
    plt.xlabel(XLabel)
    plt.ylabel(YLabel[0])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)

    ax2 = plt.subplot2grid((4,2), (2,0), rowspan=2, colspan=2, sharex=ax)
    ax2.hist(obs_one.dropna(), density = 1, cumulative=1, histtype='stepfilled', bins=300, label='OBS', alpha=0.6, edgecolor='black')
    ax2.hist(mod_one[~pd.isna(obs_one)].dropna(), density = 1, cumulative=1, histtype='stepfilled', bins=300, label=ModLegend, alpha=0.5)

    plt.xlabel(XLabel)
    plt.ylabel(YLabel[1])
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    if FigName is not None:
        fig.savefig(FigName)
    else:
        return fig


def Plot_Wavelet(Data, Time, Label='Obs.Name(Obs.Unit)', FigTitle='OBS' + 'SiteName', FigName=None):
    """A function to plot wavelet figures

    :param Data: Pandas.DataFrame[] of one column (defined observation data of one site)
    :param Time: Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param Label: YLabel for the Siginal
    :param FigTitle: SupTitle for the figure
    :param FigName: FigName including the relative path to save the Fig.

    :returns:  Figure or return NULL
    """
    fig = plt.figure(figsize=(12, 12))
    result = waipy.cwt(Data, 1, 1, 0.125, 2, 4 / 0.125, 0.72, 6, mother='Morlet', name=FigTitle)
    ax1, ax2, ax3, ax5 = waipy.wavelet_plot(FigTitle, Time, Data, 0.03125, result, fig, unit=Label)
    step = len(Data.dropna().index.date) / 5
    ax2.set_xticks(matplotlib.dates.date2num(Data.dropna().index.date)[:-step:step])
    ax2.set_xticklabels(Data.dropna().index.year[:-step:step])

    if FigName is not None:
        fig.savefig(FigName)
    else:
        return fig


def Plot_TimeSeries_TaylorGram(obs, mmod, col_num=-1, FigName='Time Series(H, D, M)'):
    """A function to plot timeseries TaylorGram

    :param ObsData: Pandas.DataFrame[] of one column (defined observation data of one site)
    :param ModData: List of Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param col_num: -1 means models, and >=0 means the model number
    :param FigName: FigName including the relative path to save the Fig.

    :returns:  Figure or return NULL
    """

    data1, data2, data3 = np.asarray(obs.resample('H').mean()), np.asarray(obs.resample('D').mean()), np.asarray(
        obs.resample('M').mean())
    #     data1,data2,data3 = (obs.resample('H').mean()), (obs.resample('D').mean()), (obs.resample('M').mean())
    data1, data2, data3 = np.ma.masked_invalid(data1), np.ma.masked_invalid(data2), np.ma.masked_invalid(data3)
    fig0 = plt.figure(figsize=(7, 7))
    models1, models2, models3 = [], [], []
    for i, mod in enumerate(mmod):
        y1 = np.asarray(mod.resample('H').mean())
        models1.append(np.ma.masked_invalid((y1)))
        y2 = np.asarray(mod.resample('D').mean())
        models2.append(np.ma.masked_invalid((y2)))
        y3 = np.asarray(mod.resample('M').mean())
        models3.append(np.ma.masked_invalid((y3)))
    if col_num < 0:
        fig0, samples1, samples2, samples3 = plot_Taylor_graph_time_basic(data1, data2, data3, models1, models2,
                                                                          models3, fig0, rect=111, ref_times=10,
                                                                          bbox_to_anchor=(1, 1))
    else:
        fig0, samples1, samples2, samples3 = plot_Taylor_graph_time_basic(data1, data2, data3, models1, models2,
                                                                          models3, fig0, rect=111, ref_times=10,
                                                                          bbox_to_anchor=(1, 1),
                                                                          modnumber=col_num + 1)
    plt.suptitle(FigName, y=TitleLocation)
    return fig0


def Plot_TimeSeries_TaylorGram_annual(obs, mmod, col_num=-1, FigName='Time Series(Four seasons)'):
    """A function to plot timeseries TaylorGram

    :param ObsData: Pandas.DataFrame[] of one column (defined observation data of one site)
    :param ModData: List of Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param col_num: -1 means models, and >=0 means the model number
    :param FigName: FigName including the relative path to save the Fig.

    :returns:  Figure or return NULL
    """
    obs_h = obs.resample('H').mean()
    obs_h_DJF, obs_h_MAM, obs_h_JJA, obs_h_SON = obs_h.copy(), obs_h.copy(), obs_h.copy(), obs_h.copy()  # PD is mutable

    obs_h_DJF[(obs_h_DJF.index.month >= 2) & (obs_h_DJF.index.month <= 11)] = None
    obs_h_MAM[(obs_h_MAM.index.month <= 2) | (obs_h_MAM.index.month >= 5)] = None
    obs_h_JJA[(obs_h_JJA.index.month <= 5) | (obs_h_JJA.index.month >= 9)] = None
    obs_h_SON[(obs_h_SON.index.month <= 8) | (obs_h_SON.index.month == 12)] = None

    data1 = obs_h_DJF.resample('A').mean()
    data2 = obs_h_MAM.resample('A').mean()
    data3 = obs_h_JJA.resample('A').mean()
    data4 = obs_h_SON.resample('A').mean()
    data0 = obs_h.resample('A').mean()
    data1, data2, data3, data4, data0 = np.asarray(data1), np.asarray(data2), np.asarray(data3), np.asarray(
        data4), np.asarray(data0)
    data1, data2, data3, data4, data0 = np.ma.masked_invalid(data1), np.ma.masked_invalid(data2), np.ma.masked_invalid(
        data3), np.ma.masked_invalid(data4), np.ma.masked_invalid(data0)
    fig0 = plt.figure(figsize=(7, 7))
    models1, models2, models3, models4, models5 = [], [], [], [], []
    for i, mod in enumerate(mmod):
        mod_h = mod.resample('H').mean()
        mod_h_DJF, mod_h_MAM, mod_h_JJA, mod_h_SON = mod_h.copy(), mod_h.copy(), mod_h.copy(), mod_h.copy()  # Hard COPY!!!
        mod_h_DJF[(mod_h_DJF.index.month >= 2) & (mod_h_DJF.index.month <= 11)] = None
        mod_h_MAM[(mod_h_MAM.index.month <= 2) | (mod_h_MAM.index.month >= 5)] = None
        mod_h_JJA[(mod_h_JJA.index.month <= 5) | (mod_h_JJA.index.month >= 9)] = None
        mod_h_SON[(mod_h_SON.index.month <= 8) | (mod_h_SON.index.month == 12)] = None
        mod1 = mod_h_DJF.resample('A').mean()
        mod2 = mod_h_MAM.resample('A').mean()
        mod3 = mod_h_JJA.resample('A').mean()
        mod4 = mod_h_SON.resample('A').mean()
        mod5 = mod_h.resample('A').mean()
        y1 = np.asarray(mod1)
        models1.append(np.ma.masked_invalid((y1)))
        y2 = np.asarray(mod2)
        models2.append(np.ma.masked_invalid((y2)))
        y3 = np.asarray(mod3)
        models3.append(np.ma.masked_invalid((y3)))
        y4 = np.asarray(mod4)
        models4.append(np.ma.masked_invalid((y4)))
        y5 = np.asarray(mod5)
        models5.append(np.ma.masked_invalid((y5)))

    if col_num <= 0:
        fig0, samples1, samples2, samples3, samples4, samples5 = plot_Taylor_graph_season_cycle(data1, data2, data3,
                                                                                                data4,
                                                                                                data0, models1, models2,
                                                                                                models3, models4,
                                                                                                models5,
                                                                                                fig0, rect=111,
                                                                                                ref_times=10,
                                                                                                bbox_to_anchor=(
                                                                                                1,1))
    else:
        fig0, samples1, samples2, samples3, samples4, samples5 = plot_Taylor_graph_season_cycle(data1, data2, data3,
                                                                                                data4,
                                                                                                data0, models1, models2,
                                                                                                models3, models4,
                                                                                                models5,
                                                                                                fig0, rect=111,
                                                                                                ref_times=10,
                                                                                                bbox_to_anchor=(
                                                                                                    1,1),
                                                                                                modnumber=col_num + 1)
    plt.suptitle(FigName, y=TitleLocation)
    return fig0


def Plot_TimeSeries_TaylorGram_hourofday(obs, mmod, col_num=-1, FigName='Cycles with Hours of a Day'):
    """A function to plot timeseries TaylorGram

    :param ObsData: Pandas.DataFrame[] of one column (defined observation data of one site)
    :param ModData: List of Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param col_num: -1 means models, and >=0 means the model number
    :param FigName: FigName including the relative path to save the Fig.

    :returns:  Figure or return NULL
    """
    obs_h = obs.resample('H').mean()
    obs_h_DJF, obs_h_MAM, obs_h_JJA, obs_h_SON = obs_h.copy(), obs_h.copy(), obs_h.copy(), obs_h.copy()  # PD is mutable

    obs_h_DJF[(obs_h_DJF.index.month >= 2) & (obs_h_DJF.index.month <= 11)] = None
    obs_h_MAM[(obs_h_MAM.index.month <= 2) | (obs_h_MAM.index.month >= 5)] = None
    obs_h_JJA[(obs_h_JJA.index.month <= 5) | (obs_h_JJA.index.month >= 9)] = None
    obs_h_SON[(obs_h_SON.index.month <= 8) | (obs_h_SON.index.month == 12)] = None

    data1 = [obs_h_DJF[obs_h_DJF.index.hour == i].mean() for i in range(24)]
    data2 = [obs_h_MAM[obs_h_MAM.index.hour == i].mean() for i in range(24)]
    data3 = [obs_h_JJA[obs_h_JJA.index.hour == i].mean() for i in range(24)]
    data4 = [obs_h_SON[obs_h_SON.index.hour == i].mean() for i in range(24)]
    data0 = [obs_h[obs_h.index.hour == i].mean() for i in range(24)]
    data1, data2, data3, data4, data0 = np.asarray(data1), np.asarray(data2), np.asarray(data3), np.asarray(
        data4), np.asarray(data0)
    data1, data2, data3, data4, data0 = np.ma.masked_invalid(data1), np.ma.masked_invalid(data2), np.ma.masked_invalid(
        data3), np.ma.masked_invalid(data4), np.ma.masked_invalid(data0)
    fig0 = plt.figure(figsize=(7, 7))
    models1, models2, models3, models4, models5 = [], [], [], [], []
    for i, mod in enumerate(mmod):
        mod_h = mod.resample('H').mean()
        mod_h_DJF, mod_h_MAM, mod_h_JJA, mod_h_SON = mod_h.copy(), mod_h.copy(), mod_h.copy(), mod_h.copy()  # Hard COPY!!!
        mod_h_DJF[(mod_h_DJF.index.month >= 2) & (mod_h_DJF.index.month <= 11)] = None
        mod_h_MAM[(mod_h_MAM.index.month <= 2) | (mod_h_MAM.index.month >= 5)] = None
        mod_h_JJA[(mod_h_JJA.index.month <= 5) | (mod_h_JJA.index.month >= 9)] = None
        mod_h_SON[(mod_h_SON.index.month <= 8) | (mod_h_SON.index.month == 12)] = None
        mod1 = [mod_h_DJF[mod_h_DJF.index.hour == i].mean() for i in range(24)]
        mod2 = [mod_h_MAM[mod_h_MAM.index.hour == i].mean() for i in range(24)]
        mod3 = [mod_h_JJA[mod_h_JJA.index.hour == i].mean() for i in range(24)]
        mod4 = [mod_h_SON[mod_h_SON.index.hour == i].mean() for i in range(24)]
        mod5 = [mod_h[mod_h.index.hour == i].mean() for i in range(24)]
        y1 = np.asarray(mod1)
        models1.append(np.ma.masked_invalid((y1)))
        y2 = np.asarray(mod2)
        models2.append(np.ma.masked_invalid((y2)))
        y3 = np.asarray(mod3)
        models3.append(np.ma.masked_invalid((y3)))
        y4 = np.asarray(mod4)
        models4.append(np.ma.masked_invalid((y4)))
        y5 = np.asarray(mod5)
        models5.append(np.ma.masked_invalid((y5)))

    if col_num <= 0:
        fig0, samples1, samples2, samples3, samples4, samples5 = plot_Taylor_graph_season_cycle(data1, data2, data3,
                                                                                                data4,
                                                                                                data0, models1, models2,
                                                                                                models3, models4,
                                                                                                models5,
                                                                                                fig0, rect=111,
                                                                                                ref_times=10,
                                                                                                bbox_to_anchor=(
                                                                                                1,1))
    else:
        fig0, samples1, samples2, samples3, samples4, samples5 = plot_Taylor_graph_season_cycle(data1, data2, data3,
                                                                                                data4,
                                                                                                data0, models1, models2,
                                                                                                models3, models4,
                                                                                                models5,
                                                                                                fig0, rect=111,
                                                                                                ref_times=10,
                                                                                                bbox_to_anchor=(
                                                                                                    1,1),
                                                                                                modnumber=col_num + 1)
    plt.suptitle(FigName, y=TitleLocation)
    return fig0

def Plot_TimeSeries_TaylorGram_cycles(obs, mmod, col_num=-1, FigName='Three Different Cycles'):
    """A function to plot timeseries TaylorGram

    :param ObsData: Pandas.DataFrame[] of one column (defined observation data of one site)
    :param ModData: List of Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param col_num: -1 means models, and >=0 means the model number
    :param FigName: FigName including the relative path to save the Fig.

    :returns:  Figure or return NULL
    """
    data1 = [obs[obs.index.dayofyear == i].mean() for i in range(365)]
    data2 = [obs[obs.index.month == i].mean() for i in range(12)]
    data3 = [np.ma.mean([data2[11],data2[0],data2[1]]),np.ma.mean([data2[4],data2[2],data2[3]]),np.ma.mean([data2[7],data2[5],data2[6]]),np.ma.mean([data2[10],data2[8],data2[9]])]
    data1,data2,data3 = np.asarray(data1), np.asarray(data2), np.asarray(data3)
    data1,data2,data3 = np.ma.masked_invalid(data1), np.ma.masked_invalid(data2), np.ma.masked_invalid(data3)
    fig0 = plt.figure(figsize=(7, 7))
    models1,models2,models3 = [],[],[]
    for i, mod in enumerate(mmod):
        y1 = [mod[mod.index.dayofyear == i].mean() for i in range(365)]
        y2 = [mod[mod.index.month == i].mean() for i in range(12)]
        y3 = [np.ma.mean([y2[11],y2[0],y2[1]]),np.ma.mean([y2[4],y2[2],y2[3]]),np.ma.mean([y2[7],y2[5],y2[6]]),np.ma.mean([y2[10],y2[8],y2[9]])]
        y1,y2,y3 = np.asarray(y1), np.asarray(y2), np.asarray(y3)
        models1.append(np.ma.masked_invalid((y1)))
        models2.append(np.ma.masked_invalid((y2)))
        models3.append(np.ma.masked_invalid((y3)))
    if col_num <0:
        fig0, samples1, samples2, samples3 = plot_Taylor_graph_time_basic(data1, data2, data3, models1, models2,
                                                                          models3, fig0, rect=111, ref_times=10,
                                                                          bbox_to_anchor=(1,1))
    else:
        fig0, samples1, samples2, samples3 = plot_Taylor_graph_time_basic(data1, data2, data3, models1, models2, models3, fig0, rect=111, ref_times=10, bbox_to_anchor=(1,1), modnumber=col_num+1)
    plt.suptitle(FigName, y=TitleLocation)
    return fig0


def TimeSeriesMoreModel_Plot(ObsData, ModData, FigTitle= None, XLabel='Obs.Name', YLabel='Obs.Unit', ModLegend='MOD',
                            FigName=None):
    """A function to plot timeseries figures

    :param ObsData: Pandas.DataFrame[] of one column (defined observation data of one site)
    :param ModData: Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param FigTitle: Title of the figure
    :param XLabel: X label of the figure (default OBS.Name)
    :param YLabel: Y label of the figure (default OBS.Unit)
    :param ModLegend: The Name of Model Legend
    :param FigName: FigName including the relative path to save the Fig.

    :returns:  Figure or return NULL
    """
    Dates = ObsData.index
    fig = plt.figure(figsize=(15, 7))
    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=2)
    if FigTitle is not None:
        plt.suptitle(FigTitle, y=TitleLocation)
    ax1.scatter(Dates, np.ones(len(Dates)), c=~pd.isna(ObsData), marker='s', cmap='binary', s=1)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set_yticklabels([])
    plt.ylabel('Existence')
    ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, colspan=2, sharex=ax1)
    ax2.plot(Dates, ObsData, 'k-', label='OBS', linewidth=1.0)
    for i, m in enumerate(ModData):
        ax2.plot(Dates, np.ma.masked_where(pd.isna(ObsData), m), '-', label=ModLegend+str(i+1), linewidth=1.0)
    # ax2.fill_between(Dates, ObsData, ModData, where=(ObsData > ModData), color='r')
    # ax2.fill_between(Dates, ObsData, ModData, where=(ModData < ObsData), color='g')
    plt.ylabel(YLabel)
    plt.xlabel(XLabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)

    if FigName is not None:
        fig.savefig(FigName)
    else:
        return fig


def CyclesMoreModel_Plot(ObsData0, ModData0, FrequencyRule=None, FigTitle='SiteName', XLabel='Hours of A Day',
                        YLabel='Obs.Name(Obs.Unit)', ModLegend='MOD',
                        FigName=None):
    """A function to plot Cycles figures

    :param ObsData0: Pandas.DataFrame[] of one column (defined observation data of one site)
    :param ModData0: Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param FrequencyRule: Define the Cycles, e.g. 'H' is Hourly of a Day; 'D' is Day of A year; 'M' is Month of A year;
    :param FigTitle: Title of the figure
    :param XLabel: X label of the figure (default OBS.Name)
    :param YLabel: Y label of the figure (default OBS.Unit)
    :param ModLegend: The Name of Model Legend
    :param FigName: FigName including the relative path to save the Fig.

    :returns:  Figure or return NULL
    """

    # import time
    # timestamp = time.strftime('%H:%M:%S')
    # timestamp
    if FrequencyRule is not None:
        if FrequencyRule == 'S':
            ObsData = ObsData0.resample('M').mean()
            ModData = [m.resample('M').mean() for m in ModData0]
        else:
            ObsData = ObsData0.resample(FrequencyRule).mean()
            ModData = [m.resample(FrequencyRule).mean() for m in ModData0]

    if FrequencyRule == 'H':
        obs_Cycle_mean = [ObsData[ObsData.index.hour == i].mean() for i in range(24)]
        obs_Cycle_var = [ObsData[ObsData.index.hour == i].std() for i in range(24)]
        mod_Cycle_mean = [[m[m.index.hour == i].mean() for i in range(24)] for m in ModData]
        mod_Cycle_var = [[m[m.index.hour == i].std() for i in range(24)] for m in ModData]
    elif FrequencyRule == 'D':
        obs_Cycle_mean = [ObsData[ObsData.index.dayofyear == i].mean() for i in range(365)]
        obs_Cycle_var = [ObsData[ObsData.index.dayofyear == i].std() for i in range(365)]
        mod_Cycle_mean = [[m[m.index.dayofyear == i].mean() for i in range(365)] for m in ModData]
        mod_Cycle_var = [[m[m.index.dayofyear == i].std() for i in range(365)] for m in ModData]
    elif FrequencyRule == 'M':
        obs_Cycle_mean = [ObsData[ObsData.index.month == i].mean() for i in range(12)]
        obs_Cycle_var = [ObsData[ObsData.index.month == i].std() for i in range(12)]
        mod_Cycle_mean = [[m[m.index.month == i].mean() for i in range(12)] for m in ModData]
        mod_Cycle_var = [[m[m.index.month == i].std() for i in range(12)] for m in ModData]
    elif FrequencyRule == 'S':
        data1 = [ObsData[ObsData.index.month == i].mean() for i in range(12)]
        data2 = [ObsData[ObsData.index.month == i].std() for i in range(12)]
        data3 = [[m[m.index.month == i].mean() for i in range(12)] for m in ModData]
        data4 = [[m[m.index.month == i].std() for i in range(12)] for m in ModData]
        obs_Cycle_mean = [np.ma.mean([data1[11], data1[0], data1[1]]), np.ma.mean([data1[4], data1[2], data1[3]]),
                 np.ma.mean([data1[7], data1[5], data1[6]]), np.ma.mean([data1[10], data1[8], data1[9]])]
        obs_Cycle_var = [np.ma.mean([data2[11], data2[0], data2[1]]), np.ma.mean([data2[4], data2[2], data2[3]]),
                 np.ma.mean([data2[7], data2[5], data2[6]]), np.ma.mean([data2[10], data2[8], data2[9]])]
        mod_Cycle_mean = [[np.ma.mean([d[11], d[0], d[1]]), np.ma.mean([d[4], d[2], d[3]]),
                          np.ma.mean([d[7], d[5], d[6]]), np.ma.mean([d[10], d[8], d[9]])] for d in data3]
        mod_Cycle_var = [[np.ma.mean([d[11], d[0], d[1]]), np.ma.mean([d[4], d[2], d[3]]),
                          np.ma.mean([d[7], d[5], d[6]]), np.ma.mean([d[10], d[8], d[9]])] for d in data4]



    fig = plt.figure(figsize=(15, 7))

    CycleTime = range(len(obs_Cycle_mean))

    if FigTitle is not None:
        plt.suptitle(FigTitle, y=TitleLocation)

    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=2)
    ax1.scatter(CycleTime, np.ones(len(CycleTime)), c=pd.isna(obs_Cycle_mean), marker='s', cmap='binary', s=1)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set_yticklabels([])
    plt.ylabel('Existence')

    ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, colspan=2, sharex=ax1)
    ax2.plot(CycleTime, obs_Cycle_mean, 'k-', label='OBS', linewidth=1.0)

    ax2.fill_between(CycleTime, np.asarray(obs_Cycle_mean) - np.asarray(obs_Cycle_var),
                     np.asarray(obs_Cycle_mean) + np.asarray(obs_Cycle_var), alpha=0.2, edgecolor='#1B2ACC',
                     linewidth=0.5, linestyle='dashdot', antialiased=True)

    for i in range(len(mod_Cycle_mean)):
        ax2.plot(CycleTime, np.ma.masked_where(pd.isna(obs_Cycle_mean), mod_Cycle_mean[i]), '-',
                 label=ModLegend + str(i + 1),
                 linewidth=1.0)
        # ax2.fill_between(CycleTime, np.asarray(mod_Cycle_mean[i]) - np.asarray(mod_Cycle_var[i]),
        #                  np.asarray(mod_Cycle_mean) + np.asarray(mod_Cycle_var), alpha=0.2, edgecolor='#1B2ACC',
        #                  linewidth=0.5, linestyle='dashdot', antialiased=True)

    if XLabel[0] == 'H':
        plt.xticks(range(24), [str(i) + '' for i in range(24)])
    elif XLabel[0] == 'D':
        plt.xticks(range(0, 365, 60), [str(i) for i in range(0, 365, 60)])
    elif XLabel[0] == 'M':
        plt.xticks(range(13), calendar.month_name[1:13], rotation=20)
    elif XLabel[0] == 'S':
        plt.xticks(range(4), ('DJF', 'MAM', 'JJA', 'SON'))

    plt.ylabel(YLabel)
    plt.xlabel(XLabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)

    if FigName is not None:
        fig.savefig(FigName)
    else:
        return fig


def PDFCDFMoreModel_Plot(obs_one, mod_one, FigTitle='SiteName', XLabel='Obs.Name(Obs.Unit)', YLabel=['Density Distribution(PDF)', 'Cumulative Distribution(CDF)'], ModLegend='MOD',FigName=None):
    """A function to plot timeseries figures

    :param ObsData: Pandas.DataFrame[] of one column (defined observation data of one site)
    :param ModData: Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param FigTitle: Title of the figure
    :param XLabel: X label of the figure (default OBS.Name)
    :param YLabel: Y label of the figure (default OBS.Unit)
    :param ModLegend: The Name of Model Legend
    :param FigName: FigName including the relative path to save the Fig.

    :returns:  Figure or return NULL
    """
# from scipy.stats import gaussian_kde
    fig = plt.figure(figsize=(9, 9))
    plt.suptitle(FigTitle, y=TitleLocation+0.2)
    ax = plt.subplot2grid((4,2), (0,0), rowspan=2, colspan=2)
    sns.kdeplot(obs_one.dropna(), shade=True, ax=ax,  color="k", label='OBS')
    for i, m in enumerate(mod_one):
        sns.kdeplot(m[~pd.isna(obs_one)].dropna(), shade=True, ax=ax, label=ModLegend+str(i+1))
    # sns.kdeplot(mod_one.dropna(), shade=True, ax=ax, label='MOD(PDF)')
    plt.xlabel(XLabel)
    plt.ylabel(YLabel[0])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)

    ax2 = plt.subplot2grid((4,2), (2,0), rowspan=2, colspan=2, sharex=ax)
    ax2.hist(obs_one.dropna(), density = 1, cumulative=1, histtype='stepfilled', bins=300, label='OBS', alpha=0.6, edgecolor='black')
    for i, m in enumerate(mod_one):
        ax2.hist(m[~pd.isna(obs_one)].dropna(), density = 1, cumulative=1, histtype='stepfilled', bins=300, label=ModLegend+str(i+1), alpha=0.5)

    plt.xlabel(XLabel)
    plt.ylabel(YLabel[1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    if FigName is not None:
        fig.savefig(FigName)
    else:
        return fig


def Plot_response2(obs1, mod1, obs2, mod2, XLabel='obs1.name +(obs1.unit)', YLabel='obs2.name +(obs2.unit)',
                   FigTitle='Title', s=None, col=-1):
    """A function to plot Response of two variables

    :param obs1: Variable 1 Pandas.DataFrame[] of one column (defined observation data of one site)
    :param mod1: List of Variable 1 Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param obs2: Variable 2 Pandas.DataFrame[] of one column (defined observation data of one site)
    :param mod2: List of Variable 2 Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param XLabel: X label of the figure (default OBS1.Name)
    :param YLabel: Y label of the figure (default OBS2.Name)
    :param FigTitle: SupTitle of the Figure
    :param s: Season plot, s=1: DJF, s=2: MAM, s=3: JJA, s=4:SON

    :returns:  Figure or return NULL
    """
    fig0 = plt.figure(figsize=(7, 7))
    ax0 = fig0.add_subplot(1, 1, 1)
    x1 = np.ma.masked_invalid(obs1)
    x2 = np.ma.masked_invalid(obs2)
    ax0.plot(x1, x2, 'k.', label='OBS')
    if len(mod1) > 1:
        for i, (m1, m2) in enumerate(zip(mod1, mod2)):
            y1 = np.ma.masked_where(pd.isna(x1), m1)
            y2 = np.ma.masked_where(pd.isna(x2), m2)
            ax0.plot(y1, y2, '.', label='MOD' + str(i + 1), color=ColorBook[i])
    else:
        for i, (m1, m2) in enumerate(zip(mod1, mod2)):
            y1 = np.ma.masked_where(pd.isna(x1), m1)
            y2 = np.ma.masked_where(pd.isna(x2), m2)
            ax0.plot(y1, y2, '.', label='MOD' + str(col + 1), color=ColorBook[i])


    ax0.set_xlabel(XLabel)
    ax0.set_ylabel(YLabel)
    if s != None:
        seasonlist = ['DJF', 'MAM', 'JJA', 'SON']
        plt.suptitle(FigTitle + ' (' + seasonlist[s - 1] + ')', y=TitleLocation)
    else:
        plt.suptitle(FigTitle, y=TitleLocation)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)

    return fig0


def binPlot(X, Y, lab=None, ax=None, numBins=8, xmin=None, xmax=None, c=None):
    """A function to do the bin plot
    Adopted from  http://peterthomasweir.blogspot.com/2012/10/plot-binned-mean-and-mean-plusminus-std.html

    :param X: x axis
    :param Y: Y axis
    :param lab: Label
    :param ax:
    :param numBins: default 8
    :param xmin:
    :param xmax:
    :param c: line color

    :returns:  Figure or return NULL
    """
    if xmin is None:
        xmin = X.min()
    if xmax is None:
        xmax = X.max()
    bins = np.linspace(xmin, xmax, numBins + 1)
    xx = np.array([np.ma.mean((bins[binInd], bins[binInd + 1])) for binInd in range(numBins)])
    yy = np.array([np.ma.mean(Y[(X > bins[binInd]) & (X <= bins[binInd + 1])]) for binInd in range(numBins)])
    yystd = 0.5 * np.array([np.std(Y[(X > bins[binInd]) & (X <= bins[binInd + 1])]) for binInd in range(numBins)])
    if lab == 'OBS':
        ax.plot(xx, yy, 'k-', label=lab)
        ax.errorbar(xx, yy, yerr=yystd, fmt='o', elinewidth=2, capthick=1, capsize=4, color='k')
    else:
        import random
        #         c=random.randint(0, 256)
        ax.plot(xx, yy, '-', label=lab, color=c)
        ax.errorbar(xx, yy, yerr=yystd, fmt='o', elinewidth=2, capthick=1, capsize=4, color=c)


def Plot_response2_error(obs1, mod1, obs2, mod2, XLabel='obs1.name +(obs1.unit)', YLabel='obs2.name +(obs2.unit)',
                         FigTitle='Title', s=None, col=-1, score=None):
    """A function to plot Response of two variables errorbar

    :param obs1: Variable 1 Pandas.DataFrame[] of one column (defined observation data of one site)
    :param mod1: List of Variable 1 Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param obs2: Variable 2 Pandas.DataFrame[] of one column (defined observation data of one site)
    :param mod2: List of Variable 2 Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param XLabel: X label of the figure (default OBS1.Name)
    :param YLabel: Y label of the figure (default OBS2.Name)
    :param FigTitle: SupTitle of the Figure
    :param s: Season plot, s=1: DJF, s=2: MAM, s=3: JJA, s=4:SON

    :returns:  Figure or return NULL
    """

    fig0 = plt.figure(figsize=(7, 7))
    ax0 = fig0.add_subplot(1, 1, 1)
    x1 = np.ma.masked_invalid(obs1)
    x2 = np.ma.masked_invalid(obs2)
    #     ax0.plot(x1, x2, 'k.', label='Obs')
    binPlot(x1, x2, ax=ax0, numBins=10, lab='OBS')

    if len(mod1)>1:
        for i, (m1, m2) in enumerate(zip(mod1, mod2)):
            #         y1 = np.ma.masked_where(pd.isna(x1),m1)
            #         y2 = np.ma.masked_where(pd.isna(x2),m2)
            #         ax0.plot(y1, y2,'.', label='mod'+str(i+1))
            y1, y2 = m1, m2
            binPlot(y1, y2, ax=ax0, numBins=10, lab='MOD' + str(i + 1), c=ColorBook[i])
    else:
        for i, (m1, m2) in enumerate(zip(mod1, mod2)):
            #         y1 = np.ma.masked_where(pd.isna(x1),m1)
            #         y2 = np.ma.masked_where(pd.isna(x2),m2)
            #         ax0.plot(y1, y2,'.', label='mod'+str(i+1))
            y1, y2 = m1, m2
            binPlot(y1, y2, ax=ax0, numBins=10, lab='MOD' + str(col + 1), c=ColorBook[i])

    ax0.set_xlabel(XLabel)
    ax0.set_ylabel(YLabel)
    if s != None:
        seasonlist = ['DJF', 'MAM', 'JJA', 'SON']
        plt.suptitle(FigTitle + ' (' + seasonlist[s - 1] + ')', y=TitleLocation)
    else:
        plt.suptitle(FigTitle, y=TitleLocation)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    return fig0


def Plot_response4(obs1, mod1, obs2, mod2, obs3, mod3, obs4, mod4, FigTitle='obs4.name+(obs4.unit)',
                   XLabel='obs1.name+(obs1.unit)', YLabel='obs2.name+(obs2.unit)', ZLabel='obs3.name+(obs3.unit)',
                   s=None, col_num=-1, Frequency ='D'):
    """A function to plot Response of four variables

    :param obs1: Variable 1 Pandas.DataFrame[] of one column (defined observation data of one site)
    :param mod1: List of Variable 1 Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param obs2: Variable 2 Pandas.DataFrame[] of one column (defined observation data of one site)
    :param mod2: List of Variable 2 Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param obs3: Variable 3 Pandas.DataFrame[] of one column (defined observation data of one site)
    :param mod3: List of Variable 3 Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param obs4: Variable 4 Pandas.DataFrame[] of one column (defined observation data of one site)
    :param mod4: List of Variable 4 Pandas.DataFrame[] of one column (defined Mode data of one site)

    :param XLabel: X label of the figure (default OBS1.Name)
    :param YLabel: Y label of the figure (default OBS2.Name)
    :param FigTitle: SupTitle of the Figure
    :param s: Season plot, s=1: DJF, s=2: MAM, s=3: JJA, s=4:SON

    :returns:  Figure or return NULL
    """
    ResampleFrequency = Frequency
    fig0 = plt.figure(figsize=(9, 9 * (len(mod1) + 1)))
    if s != None:
        seasonlist = ['DJF', 'MAM', 'JJA', 'SON']
        plt.suptitle(FigTitle + '(' + seasonlist[s - 1] + ')'+'['+Frequency+']', y=1.)
    else:
        plt.suptitle(FigTitle+'['+Frequency+']', y=TitleLocation)

    ax0 = fig0.add_subplot(len(mod1) + 1, 1, 1, projection='3d')
    x1,x2,x3,x4=obs1.resample(ResampleFrequency).mean(),obs2.resample(ResampleFrequency).mean(),obs3.resample(ResampleFrequency).mean(),obs4.resample(ResampleFrequency).mean()
    colorax = ax0.scatter(x1, x2, x3, c=x4, cmap=plt.hot())
    ax0.set_xlabel(XLabel)
    ax0.set_ylabel(YLabel)
    ax0.set_zlabel(ZLabel)
    ax0.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax0.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax0.zaxis.set_major_locator(plt.MaxNLocator(2))
    ax0.tick_params(pad=-5)
    format_func = lambda x, t: "{:.1E}".format(x)
    ax0.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax0.zaxis.set_major_formatter(plt.FuncFormatter(format_func))
    for label in ax0.zaxis.get_ticklabels():
        label.set_rotation(-40)

    # fig0.subplots_adjust(right=0.7)
    z = x4
    axes = fig0.add_axes([1.1, 0.25, 0.02, 0.5])
    cbar = fig0.colorbar(colorax, ticks=[min(z), max(z)], orientation='vertical',
                         label=obs4.name, cax=axes)
    cbar.ax.set_yticklabels(['Low', 'High'])  # horizontal colorbar

    for i in range(len(mod1)):
        y1, y2, y3, y4 = mod1[i].resample(ResampleFrequency).mean(), mod2[i].resample(ResampleFrequency).mean(), mod3[i].resample(ResampleFrequency).mean(), mod4[i].resample(ResampleFrequency).mean()
        ax0 = fig0.add_subplot(len(mod1) + 1, 1, i + 2, projection='3d')
        ax0.scatter(y1, y2, y3, c=y4, cmap=plt.hot())
        if col_num < 0:
            ax0.set_title('MOD' + str(i + 1) + ':' + FigTitle)
        else:
            ax0.set_title('MOD' + str(col + 1) + ':' + FigTitle)
        ax0.set_xlabel(XLabel)
        ax0.set_ylabel(YLabel)
        ax0.set_zlabel(ZLabel)
        ax0.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax0.yaxis.set_major_locator(plt.MaxNLocator(2))
        ax0.zaxis.set_major_locator(plt.MaxNLocator(2))
        ax0.tick_params(pad=-5)
        ax0.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax0.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax0.zaxis.set_major_formatter(plt.FuncFormatter(format_func))
        for label in ax0.zaxis.get_ticklabels():
            label.set_rotation(-40)

            #     plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
    plt.tight_layout()

    return fig0


def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.

    # Partial Correlation in Python (clone of Matlab's partialcorr)
    # This uses the linear regression approach to compute the partial
    # correlation (might be slow for a huge number of variables).The code is adopted from
    # https://gist.github.com/fabianp/9396204419c7b638d38f
    # Date: Nov 2014
    # Author: Fabian Pedregosa-Izquierdo, f@bianp.net
    # Testing: Valentina Borghesani, valentinaborghesani@gmail.com

    """

    C = np.column_stack([C, np.ones(C.shape[0])])

    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr[0:C.shape[1] - 1, 0:C.shape[1] - 1]

def PlotFourVariableCorr(obs1, mod1, obs2, mod2, obs3, mod3, obs4, mod4, FigTitle='obs4.name+(obs4.unit)',
                         XLabel='obs1.name+(obs1.unit)', YLabel='obs2.name+(obs2.unit)', ZLabel='obs3.name+(obs3.unit)',
                         s=None, col_num=-1, Frequency='M', score=None):
    """A function to plot Partial Correlation of four variables

    :param obs1: Variable 1 Pandas.DataFrame[] of one column (defined observation data of one site)
    :param mod1: List of Variable 1 Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param obs2: Variable 2 Pandas.DataFrame[] of one column (defined observation data of one site)
    :param mod2: List of Variable 2 Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param obs3: Variable 3 Pandas.DataFrame[] of one column (defined observation data of one site)
    :param mod3: List of Variable 3 Pandas.DataFrame[] of one column (defined Mode data of one site)
    :param obs4: Variable 4 Pandas.DataFrame[] of one column (defined observation data of one site)
    :param mod4: List of Variable 4 Pandas.DataFrame[] of one column (defined Mode data of one site)

    :param XLabel: X label of the figure (default OBS1.Name)
    :param YLabel: Y label of the figure (default OBS2.Name)
    :param FigTitle: SupTitle of the Figure
    :param s: Season plot, s=1: DJF, s=2: MAM, s=3: JJA, s=4:SON

    :returns:  Figure or return NULL
    """

    ResampleFrequency = Frequency
    x1, x2, x3, x4 = obs1.resample(ResampleFrequency).mean(), obs2.resample(ResampleFrequency).mean(), obs3.resample(
        ResampleFrequency).mean(), obs4.resample(ResampleFrequency).mean()
    frames = [x1, x2, x3, x4]
    result = pd.concat(frames, axis=1)
    corr0 = partial_corr(result.dropna())
    fig0 = plt.figure(figsize=(9, 9))
    ax0 = fig0.add_subplot(1, 1, 1, projection='3d')
    ax0.scatter(corr0[0, 1], corr0[0, 2], corr0[0, 3], label='OBS', s=60)
    if isinstance(score, list):
        score.append((corr0[0, 1], corr0[0, 2], corr0[0, 3]))

    if s != None:
        seasonlist = ['DJF', 'MAM', 'JJA', 'SON']
        plt.suptitle(FigTitle + '(' + seasonlist[s - 1] + ')'+'['+Frequency+']', y=TitleLocation)
    else:
        plt.suptitle(FigTitle+'['+Frequency+']', y=TitleLocation)

    if col_num <= 0:
        for i in range(len(mod1)):
            y1, y2, y3, y4 = mod1[i].resample(ResampleFrequency).mean(), mod2[i].resample(ResampleFrequency).mean(), \
                             mod3[i].resample(ResampleFrequency).mean(), mod4[i].resample(ResampleFrequency).mean()
            frames = [y1, y2, y3, y4]
            result = pd.concat(frames, axis=1)
            corr0 = partial_corr(result.dropna())
            ax0.scatter(corr0[0, 1], corr0[0, 2], corr0[0, 3], label='MOD' + str(i + 1), s=60)
            if isinstance(score, list):
                score.append((corr0[0, 1], corr0[0, 2], corr0[0, 3]))
    else:
        for i in range(len(mod1)):
            y1, y2, y3, y4 = mod1[i].resample(ResampleFrequency).mean(), mod2[i].resample(ResampleFrequency).mean(), \
                             mod3[i].resample(ResampleFrequency).mean(), mod4[i].resample(ResampleFrequency).mean()
            frames = [y1, y2, y3, y4]
            result = pd.concat(frames, axis=1)
            corr0 = partial_corr(result.dropna())
            ax0.scatter(corr0[0, 1], corr0[0, 2], corr0[0, 3], label='MOD' + str(col_num), s=60)

    ax0.set_xlabel(XLabel)
    ax0.set_ylabel(YLabel)
    ax0.set_zlabel(ZLabel)
    ax0.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax0.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax0.zaxis.set_major_locator(plt.MaxNLocator(2))
    ax0.tick_params(pad=-5)
    format_func = lambda x, t: "{:.1E}".format(x)
    ax0.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax0.zaxis.set_major_formatter(plt.FuncFormatter(format_func))

    for label in ax0.zaxis.get_ticklabels():
        label.set_rotation(-40)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    return fig0

def plot_variable_matrix_trend_and_detrend(data, dtrend_data, variable_list, col_num=-1, site_name = None):
    """A function to plot matrix correlation board

    :param data: Correlation
    :param dtrend_data: Detrend Data's Correlation
    :param variable_list: Name of Data List

    :param s: Season plot, s=1: DJF, s=2: MAM, s=3: JJA, s=4:SON

    :returns:  Figure or return NULL
    """
    fig, axes = plt.subplots(len(variable_list), len(variable_list), sharex=True, sharey=True,
                             figsize=(15, 15))
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    plt.suptitle(site_name, y=TitleLocation)
    plt.figtext(0.99, 0.01, 'Regular trending data while * denotes for being detrended', horizontalalignment='right')
    ax_cb = fig.add_axes([.91, .3, .03, .4])

    for j in range(len(variable_list)):  # models
        for k in range(j, len(variable_list)):
            array = data[:, k, j]  ## Put n sites in one picture
            nam = ['OBS('+str(array[0])[:6]+')']
            if col_num ==2:
                nam.extend(['MOD'+str(i+1)+'('+str(array[i+1])[:4]+')' for i in range(len(array)-1)])
            else:
                nam.extend(['MOD'+str(col_num+1)+'('+str(array[i+1])[:4]+')' for i in range(len(array)-1)])
            df_cm = pd.DataFrame(array)
            annot = [nam[i] for i in range(len(array))]
            # ax.pie(array, autopct=lambda(p): '{v:d}'.format(p * sum(list(array)) / 100), startangle=90,colors=my_cmap(my_norm(color_vals)))
            sns.heatmap(df_cm, annot=np.array([annot]).T, cmap='Spectral', cbar=k == 0, ax=axes[k][j],
                       vmin=-1, vmax=1, fmt = '',
                       cbar_ax=None if k else ax_cb)
            # print(i, j)
            if j == 0:
                axes[k][j].set_ylabel((variable_list[k]))
            if k == len(variable_list) - 1:
                axes[k][j].set_xlabel(variable_list[j])
            # axes[i][j].axis('off')
            axes[k][j].set_yticklabels([])
            axes[k][j].set_xticklabels([])

    for j in range(len(variable_list)):  # models
        for k in range(0, j):
            array = dtrend_data[:, k, j]  ## Put n sites in one picture
            nam = ['OBS*('+str(array[0])[:6]+')']
            if col_num ==2:
                nam.extend(['MOD'+str(i+1)+'*('+str(array[i+1])[:4]+')' for i in range(len(array)-1)])
            else:
                nam.extend(['MOD'+str(col_num+1)+'*('+str(array[i+1])[:4]+')' for i in range(len(array)-1)])
            df_cm = pd.DataFrame(array)
            annot = [nam[i] for i in range(len(array))]
            # ax.pie(array, autopct=lambda(p): '{v:d}'.format(p * sum(list(array)) / 100), startangle=90,colors=my_cmap(my_norm(color_vals)))
            sns.heatmap(df_cm, annot=np.array([annot]).T, cmap='Spectral', ax=axes[k][j],cbar=False,
                       vmin=-1, vmax=1, fmt = '',
                       cbar_ax=None)
            # print(i, j)
            # if j == 0:
            #     axes[k][j].set_ylabel((variable_list[k]))
            # if k == len(variable_list) - 1:
            #     axes[k][j].set_xlabel(variable_list[j])
            # axes[i][j].axis('off')
            axes[k][j].set_yticklabels([])
            axes[k][j].set_xticklabels([])

    return fig


def CorrelationMatrix_TrendAndDetrend(list_, ResampleFrequency='M'):
    """A function to compute data correlation and detrend correlation

    :param list_: List of pandas dataframe
    :param ResampleFrequency: Pandas resample frequency
    :param s: Season plot, s=1: DJF, s=2: MAM, s=3: JJA, s=4:SON: TO DO>>>>

    :returns:  Figure or return NULL
    """
    frames =[x.resample(ResampleFrequency).mean() for x in list_]
    dframes = []
    for y in frames:
        not_nan_ind = ~np.isnan(y)
#         print(len(not_nan_ind), len(y.dropna()))
        m, b, _, _, _ = stats.linregress(np.arange(len(y))[not_nan_ind],y.dropna())
        dy = y - (m*np.arange(len(y)) + b)
        dframes.append(dy)
    result = pd.concat(frames, axis=1)
    dresult = pd.concat(dframes, axis=1)
    A = result.corr(method='pearson', min_periods=1)
    corr = np.ma.corrcoef(A)
    mask = np.zeros_like(A)
    mask[np.triu_indices_from(mask)] = True
    dA = dresult.corr(method='pearson', min_periods=1)
    dcorr = np.ma.corrcoef(dA)
    dmask = np.zeros_like(dA)
    dmask[np.triu_indices_from(dmask)] = True
    return corr, mask, dcorr, dmask


def pil_grid(images, max_horiz=np.iinfo(int).max):
    '''This is a function used to list all the pictures'''
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid

def MaskSeasonData(obs_h, mod_h):

    """This is a function used to mask all different seasons based on pandas's month timestamp

    :param obs_h: Variable 1 Pandas.DataFrame[] of one column (defined observation data of one site)
    :param mod_h: List of Variable 1 Pandas.DataFrame[] of one column (defined Mode data of one site)

    :returns:  Seasons Pandas DateFrame
    """
    obs_h_DJF, obs_h_MAM, obs_h_JJA, obs_h_SON = obs_h.copy(), obs_h.copy(), obs_h.copy(), obs_h.copy()  # PD is mutable
    mod_h_DJF, mod_h_MAM, mod_h_JJA, mod_h_SON = [m.copy() for m in mod_h], [m.copy() for m in mod_h], [
        m.copy() for m in mod_h], [m.copy() for m in mod_h]  # Hard COPY!!!

    obs_h_DJF[(obs_h_DJF.index.month >= 2) & (obs_h_DJF.index.month <= 11)] = None
    for m in mod_h_DJF:
        m[(m.index.month >= 2) & (m.index.month <= 11)] = None

    obs_h_MAM[(obs_h_MAM.index.month <= 2) | (obs_h_MAM.index.month >= 5)] = None
    for m in mod_h_DJF:
        m[(m.index.month <= 2) | (m.index.month >= 5)] = None

    obs_h_JJA[(obs_h_JJA.index.month <= 5) | (obs_h_JJA.index.month >= 9)] = None
    for m in mod_h_JJA:
        m[(m.index.month <= 5) | (m.index.month >= 9)] = None

    obs_h_SON[(obs_h_SON.index.month <= 8) | (obs_h_SON.index.month == 12)] = None
    for m in mod_h_SON:
        m[(m.index.month <= 8) | (m.index.month == 12)] = None

    return obs_h_DJF, obs_h_MAM, obs_h_JJA, obs_h_SON, mod_h_DJF, mod_h_MAM, mod_h_JJA, mod_h_SON

class ConfSite(Confrontation):
    """

    A confrontation for examining the site based on iLAMB Confrontation

    (Here we assume iLAMB has passed out the correct obs and mods

    If not, observations and Models could be manually input in the begining of __init__ )

    """

    def __init__(self, **keywords):

        # Calls the regular constructor
        super(ConfSite, self).__init__(**keywords)
        # Setup a html layout for generating web views of the results
        print(' ....... Imported all the required Packages .......')

        # obs = Variable(filename=self.source, variable_name=self.variable, alternate_vars=self.alternate_vars)

        lat0 = np.array([47.1167, -35.6557, 51.3092, 50.3055, -2.8567,
         48.2167, 49.5026, 51.0793, 50.9636, 55.4869,
         74.4732, 61.8474, 43.7414, 5.2788, 45.9553,
         46.5878, 42.3903, 52.1679, 70.6167, 56.4917,
         72.3738, 38.3953, 42.5378, 46.0826, 41.5545,
         45.9459, 31.8214, 46.242, 38.1159, 45.5598,
         38.4067, 45.8059, 31.7365, -25.0197])
        lon0 = np.array([ 11.31750011,  148.1519928 ,    4.52059984,
                      5.99679995,  -54.95890045,  -82.1556015 ,
                     18.53840065,   10.45199966,   13.56690025,
                     11.64579964,  -20.5503006 ,   24.28479958,
                      3.59579992,  -52.92490005,   11.28120041,
                     11.43470001,   11.92090034,    5.74399996,
                    147.88299561,   32.9239006 ,  126.4957962 ,
                   -120.63279724,  -72.17150116,  -89.97920227,
                    -83.84380341,  -90.27230072, -110.8660965 ,
                    -89.34770203, -120.96600342,  -84.71379852,
                   -120.95069885,  -90.07990265, -109.94190216,
                     31.49690056])
        # obs = Variable(filename='/Users/lli51/Downloads/alldata/obs_FSH_model_ilamb.nc4',
        #                  variable_name="FSH")
        RegionsFile = Dataset('/Users/lli51/Desktop/IGBPa_1198.map.nc')  # Supposed to change dir based on different directory/Or put subdirectory

        # Current Site Name & Site Location information i.e. lat, lon based on Observations ## Dec. 13. 2018.
        self.sitenumber = 3  # len(regions)
        self.sitename = ['SITE: ' + str(i) for i in
                         range(self.sitenumber)]  # Supposed to be learned or input from .nc file
        self.lats = lat0[:self.sitenumber]
        self.lons = lon0[:self.sitenumber]

        # Define the Regions(Sinle Site, Geographic Group, or District IGBP(or Other Groups) )
        reg = Regions()
        regions = []
        self.RegionsName = {}
        self.ind_regions = collections.defaultdict(set)
        ### -----------------  1. Multiple Site in one group  ----------------------------------- ####
        # Setup the region for multiple site regions based on IGBP
        self.RegionsIGBP = {'I1': 'Evergreen Needleleaf Forest', 'I2': 'Evergreen Broadleaf Forest',
                            'I2': 'Deciduous Needleleaf Forest', 'I4': 'Deciduous Broadleaf Forest', 'I5': ' Mixed Forest',
                            'I6': 'Closed Shrublands', 'I7': 'Open Shrublands', 'I8': ' Woody Savannas',
                            'I9': 'Savannas', 'I10': 'Grasslands ', 'I11': 'Permanent Wetlands',
                            'I12': 'Croplands', 'I13': 'Urban and Built-up', 'I14': 'Cropland Mosaics',
                            'I15': 'Snow and Ice (permanent)', 'I16': 'Bare Soil and Rocks', 'I17': 'Water Bodies', 'I18': 'Tundra'}

        lat_region, lon_region = RegionsFile['lat'][:], RegionsFile['lon'][:]
        RegionsLabel = RegionsFile['CLASS']
        # Given the sites for the certain region
        for i, (lat, lon) in enumerate(zip(self.lats, self.lons)):
            ind_lat = (np.abs(lat_region - lat)).argmin()
            ind_lon = (np.abs(lon_region - lon)).argmin()
            ind_label = RegionsLabel[ind_lat][ind_lon]
            self.ind_regions['I'+str(ind_label)].add(i)
            self.RegionsName['I'+str(ind_label)] = self.RegionsIGBP['I'+str(ind_label)]

        for i, r in enumerate(self.ind_regions.keys()):
            reg.addRegionLatLonBounds(r, self.RegionsIGBP[r], (90 + i - 0.5, 90 + i + 0.5), (180 + i - 0.5, 180 + i + 0.5))
            regions.append(r)


        ### -----------------  2. Single Site in one group location based ----------------------------------- ####
        # Setup the name of the sites and regions' bounds
        for i, (lat, lon) in enumerate(zip(self.lats, self.lons)):
            reg.addRegionLatLonBounds('site'+str(i), self.sitename[i], (lat - 0.01, lat + 0.01), (lon - 0.01, lon + 0.01))
            regions.append('site'+str(i))
            self.ind_regions['site'+str(i)].add(i) #add response sites to the regions
            self.RegionsName['site'+str(i)] =self.sitename[i]


        ### -----------------  3. Multiple Site in one group location based -----------------------------------####
        # To do, setup the region for multiple site regions based on Geographic Locations
        Geo_regions = []
        self.lowerlatbound = []
        self.upperlatbound = []
        self.lowerlonbound = []
        self.upperlonbound = []

        regions.append("bona")
        Geo_regions.append("bona")
        self.lowerlatbound.append(49.75)
        self.upperlatbound.append(79.75)
        self.lowerlonbound.append(-170.25)
        self.upperlonbound.append(-60.25)

        regions.append("tena")
        Geo_regions.append("tena")
        self.lowerlatbound.append(30.25)
        self.upperlatbound.append(49.75)
        self.lowerlonbound.append(-125.25)
        self.upperlonbound.append(-66.25)

        self.RegionsName['bona'] = "Boreal North America"
        self.RegionsName['tena'] = "Temperate North America"
        self.RegionsName['global'] = "Globe"

        reg.addRegionLatLonBounds("bona", "Boreal North America", (49.75, 79.75), (-170.25, - 60.25))
        reg.addRegionLatLonBounds("tena","Temperate North America", (30.25, 49.75),(-125.25,- 66.25))
        #         reg.addRegionLatLonBounds("ceam","Central America",                  (  9.75, 30.25),(-115.25,- 80.25))
        #         reg.addRegionLatLonBounds("nhsa","Northern Hemisphere South America",(  0.25, 12.75),(- 80.25,- 50.25))
        #         reg.addRegionLatLonBounds("shsa","Southern Hemisphere South America",(-59.75,  0.25),(- 80.25,- 33.25))
        #         reg.addRegionLatLonBounds("euro","Europe",                           ( 35.25, 70.25),(- 10.25,  30.25))
        #         reg.addRegionLatLonBounds("mide","Middle East",                      ( 20.25, 40.25),(- 10.25,  60.25))
        #         reg.addRegionLatLonBounds("nhaf","Northern Hemisphere Africa",       (  0.25, 20.25),(- 20.25,  45.25))
        #         reg.addRegionLatLonBounds("shaf","Southern Hemisphere Africa",       (-34.75,  0.25),(  10.25,  45.25))
        #         reg.addRegionLatLonBounds("boas","Boreal Asia",                      ( 54.75, 70.25),(  30.25, 179.75))
        #         reg.addRegionLatLonBounds("ceas","Central Asia",                     ( 30.25, 54.75),(  30.25, 142.58))
        #         reg.addRegionLatLonBounds("seas","Southeast Asia",                   (  5.25, 30.25),(  65.25, 120.25))
        #         reg.addRegionLatLonBounds("eqas","Equatorial Asia",                  (-10.25, 10.25),(  99.75, 150.25))
        #         reg.addRegionLatLonBounds("aust","Australia",                        (-41.25,-10.50),( 112.00, 154.00))
        #
        reg.addRegionLatLonBounds("global", "Globe", (-90, 90), (-180, 180))
        regions.append("global")
        Geo_regions.append("global")
        self.lowerlatbound.append(-90)
        self.upperlatbound.append(90)
        self.lowerlonbound.append(-180)
        self.upperlonbound.append(180)

        # self.regions = regions
        for regionid, regionname in enumerate(Geo_regions):
            for j, (lat, lon) in enumerate(zip(self.lats, self.lons)):
                if lat < self.upperlatbound[regionid] and lat > self.lowerlatbound[regionid] and lon < self.upperlonbound[regionid] and lon > self.lowerlonbound[regionid]:
                    self.ind_regions[regionname].add(j)
        self.regions = self.ind_regions.keys()
        # hello
        # Above finished all the regions' setup
        ### ---------------------------------------------------- ####

        pages = []
        pages.append(post.HtmlPage("MeanState", "Mean State"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections(
            ["Time series", "Time series(Annually)", "Cycles mean", "Cycles mean(seasonly)", "PDF CDF", "Frequency",
             "Response two variables", "Response four variables", "Correlations"])
        pages[-1].setRegions(self.regions)  # self.regions
        pages.append(post.HtmlAllModelsPage("AllModels", "All Models"))
        pages[-1].setHeader("CNAME / RNAME")
        # pages[-1].setSections(["Time series", "Cycles mean", "Frequency", "Response"])
        pages[-1].setSections([])
        pages[-1].setRegions(self.regions)

        pages.append(post.HtmlPage("DataInformation", "Data Information"))
        pages[-1].setSections([])
        pages[-1].text = "\n"

        with Dataset(self.source) as dset:
            for attr in dset.ncattrs():
                pages[-1].text += "<p><b>&nbsp;&nbsp;%s:&nbsp;</b>%s</p>\n" % (
                    attr, dset.getncattr(attr).encode('ascii', 'ignore'))
        self.layout = post.HtmlLayout(pages, self.longname)

    def stageData(self, m):

        obs = Variable(filename=self.source,
                       variable_name=self.variable,
                       alternate_vars=self.alternate_vars)

        # the model data needs integrated over the globe
        mod = m.extractTimeSeries(self.variable, alt_vars=self.alternate_vars)
        # mod = mod.integrateInSpace().convert(obs.unit)

        obs, mod = il.MakeComparable(obs, mod, clip_ref=True)

        obs = Variable(name=obs.name,unit=obs.unit,time=obs.time,data=obs.data[:,0,0:self.sitenumber])
        mod = Variable(name=mod.name, unit=mod.unit, time=mod.time, data=mod.data[:, 0, 0:self.sitenumber])

        return obs, mod

    def confront(self, m):

        output_path = './test_output/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)


        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]

        print(' .......  Loading DataSet ....... ')

        # # Below We Test the Date of one Variable, v1obs is our current root variable
        # v1obs = Variable(filename='/Users/lli51/Downloads/alldata/obs_FSH_model_ilamb.nc4',
        #                  variable_name="FSH")
        # v1mod1 = Variable(filename='/Users/lli51/Downloads/alldata/171206_ELMv0_CN_FSH_model_ilamb.nc4',
        #                   variable_name="FSH")
        # v1mod2 = Variable(filename='/Users/lli51/Downloads/alldata/171206_ELMv1_CN_FSH_model_ilamb.nc4',
        #                   variable_name="FSH")
        #
        # v2obs = Variable(filename='/Users/lli51/Downloads/alldata/obs_GPP_model_ilamb.nc4',
        #                  variable_name="GPP")
        # v2mod1 = Variable(filename='/Users/lli51/Downloads/alldata/171206_ELMv0_CN_GPP_model_ilamb.nc4',
        #                   variable_name="GPP")
        # v2mod2 = Variable(filename='/Users/lli51/Downloads/alldata/171206_ELMv1_CN_GPP_model_ilamb.nc4',
        #                   variable_name="GPP")
        #
        # v3obs = Variable(filename='/Users/lli51/Downloads/alldata/obs_NEE_model_ilamb.nc4',
        #                  variable_name="NEE")
        # v3mod1 = Variable(filename='/Users/lli51/Downloads/alldata/171206_ELMv0_CN_NEE_model_ilamb.nc4',
        #                   variable_name="NEE")
        # v3mod2 = Variable(filename='/Users/lli51/Downloads/alldata/171206_ELMv1_CN_NEE_model_ilamb.nc4',
        #                   variable_name="NEE")
        #
        # v4obs = Variable(filename='/Users/lli51/Downloads/alldata/obs_ER_model_ilamb.nc4',
        #                  variable_name="ER")
        # v4mod1 = Variable(filename='/Users/lli51/Downloads/alldata/171206_ELMv0_CN_ER_model_ilamb.nc4',
        #                   variable_name="ER")
        # v4mod2 = Variable(filename='/Users/lli51/Downloads/alldata/171206_ELMv1_CN_ER_model_ilamb.nc4',
        #                   variable_name="ER")

        # Grab the data
        obs, mod = self.stageData(m)
        self.score = collections.defaultdict(list)
        # v1obs, v1mod1, v1mod2 = obs, mod, mod
        # v2obs, v2mod1, v2mod2 = obs, mod, mod
        # v3obs, v3mod1, v3mod2 = obs, mod, mod
        # v4obs, v4mod1, v4mod2 = obs, mod, mod
        # varsobs = [v1obs, v2obs, v3obs, v4obs]
        # varsmod = [[v1mod1, v1mod2], [v2mod1, v2mod2], [v3mod1, v3mod2], [v4mod1, v4mod2]]
        # v1dates = num2date(v1obs.time, units='days since 1850-01-01 00:00:00')
        # varsobsDF = [pd.DataFrame(m.data[:], index=v1dates) for m in varsobs]
        # varsmodDF = [[pd.DataFrame(m.data[:], index=v1dates) for m in mlist] for mlist in varsmod]

        # Read the data Faster
        v1obs = Dataset('/Users/lli51/Downloads/alldata/obs_FSH_model_ilamb.nc4')['FSH']
        v1mod1 = Dataset('/Users/lli51/Downloads/alldata/171206_ELMv0_CN_FSH_model_ilamb.nc4')['FSH']
        v1mod2 = Dataset('/Users/lli51/Downloads/alldata/171206_ELMv1_CN_FSH_model_ilamb.nc4')['FSH']

        v2obs = Dataset('/Users/lli51/Downloads/alldata/obs_GPP_model_ilamb.nc4')['GPP']
        v2mod1 = Dataset('/Users/lli51/Downloads/alldata/171206_ELMv0_CN_GPP_model_ilamb.nc4')['GPP']
        v2mod2 = Dataset('/Users/lli51/Downloads/alldata/171206_ELMv1_CN_GPP_model_ilamb.nc4')['GPP']

        v3obs = Dataset('/Users/lli51/Downloads/alldata/obs_NEE_model_ilamb.nc4')['NEE']
        v3mod1 = Dataset('/Users/lli51/Downloads/alldata/171206_ELMv0_CN_NEE_model_ilamb.nc4')['NEE']
        v3mod2 = Dataset('/Users/lli51/Downloads/alldata/171206_ELMv1_CN_NEE_model_ilamb.nc4')['NEE']

        v4obs = Dataset('/Users/lli51/Downloads/alldata/obs_ER_model_ilamb.nc4')['ER']
        v4mod1 = Dataset('/Users/lli51/Downloads/alldata/171206_ELMv0_CN_ER_model_ilamb.nc4')['ER']
        v4mod2 = Dataset('/Users/lli51/Downloads/alldata/171206_ELMv1_CN_ER_model_ilamb.nc4')['ER']
        times = Dataset('/Users/lli51/Downloads/alldata/obs_FSH_model_ilamb.nc4')['time']

        dates = num2date(times[:], units=times.units)
        varsobs = [v1obs, v2obs, v3obs, v4obs]
        varsmod = [[v1mod1, v1mod2], [v2mod1, v2mod2], [v3mod1, v3mod2], [v4mod1, v4mod2]]

        varsobsDF = [pd.DataFrame(m[:], index=dates) for m in varsobs]
        varsmodDF = [[pd.DataFrame(m[:], index=dates) for m in mlist] for mlist in varsmod]


        Var1_obs_DF = varsobsDF[0]
        Var1_Mod_DF_List = varsmodDF[0]
        Var2_obs_DF = varsobsDF[1]
        Var2_Mod_DF_List = varsmodDF[1]
        Var3_obs_DF = varsobsDF[2]
        Var3_Mod_DF_List = varsmodDF[2]
        Var4_obs_DF = varsobsDF[3]
        Var4_Mod_DF_List = varsmodDF[3]

        ILAMB_Var_OBS = v1obs
        IterNameModList = ["Model1", "Model2", "Model"]
        self.IterNameModList = IterNameModList
        for ModNum, ModName in enumerate(IterNameModList):
            # if ModName != "Models":
            #     continue

            if ModName == "Models":
                Var1_IterModList_DF = Var1_Mod_DF_List
                Var2_IterModList_DF = Var2_Mod_DF_List
                Var3_IterModList_DF = Var3_Mod_DF_List
                Var4_IterModList_DF = Var4_Mod_DF_List

            else:
                Var1_IterModList_DF = [Var1_Mod_DF_List[ModNum - 1]]
                Var2_IterModList_DF = [Var2_Mod_DF_List[ModNum - 1]]
                Var3_IterModList_DF = [Var3_Mod_DF_List[ModNum - 1]]
                Var4_IterModList_DF = [Var4_Mod_DF_List[ModNum - 1]]


            # Plot Response2
            # print(self.regions)
            # hello

            print('.......  Processing Response Post ....... ')
            for region in self.regions:  # len(obsDF.columns)

                v1_obs_one, v1_mod_one = Var1_obs_DF[list(self.ind_regions[region])].mean(axis=1), [m[list(self.ind_regions[region])].mean(axis=1) for m in Var1_IterModList_DF]
                v2_obs_one, v2_mod_one = Var2_obs_DF[list(self.ind_regions[region])].mean(axis=1), [m[list(self.ind_regions[region])].mean(axis=1) for m in Var2_IterModList_DF]
                v3_obs_one, v3_mod_one = Var3_obs_DF[list(self.ind_regions[region])].mean(axis=1), [m[list(self.ind_regions[region])].mean(axis=1) for m in Var3_IterModList_DF]
                v4_obs_one, v4_mod_one = Var4_obs_DF[list(self.ind_regions[region])].mean(axis=1), [m[list(self.ind_regions[region])].mean(axis=1) for m in Var4_IterModList_DF]

                v1_obs_h_DJF, v1_obs_h_MAM, v1_obs_h_JJA, v1_obs_h_SON, v1_mod_h_DJF, v1_mod_h_MAM, v1_mod_h_JJA, v1_mod_h_SON = MaskSeasonData(
                    v1_obs_one, v1_mod_one)
                v2_obs_h_DJF, v2_obs_h_MAM, v2_obs_h_JJA, v2_obs_h_SON, v2_mod_h_DJF, v2_mod_h_MAM, v2_mod_h_JJA, v2_mod_h_SON = MaskSeasonData(
                    v2_obs_one, v2_mod_one)

                v3_obs_h_DJF, v3_obs_h_MAM, v3_obs_h_JJA, v3_obs_h_SON, v3_mod_h_DJF, v3_mod_h_MAM, v3_mod_h_JJA, v3_mod_h_SON = MaskSeasonData(
                    v3_obs_one, v3_mod_one)
                v4_obs_h_DJF, v4_obs_h_MAM, v4_obs_h_JJA, v4_obs_h_SON, v4_mod_h_DJF, v4_mod_h_MAM, v4_mod_h_JJA, v4_mod_h_SON = MaskSeasonData(
                    v4_obs_one, v4_mod_one)

                # print('Plot Data Correlation Boards')
                variable_list = [v1obs.name, v2obs.name, v3obs.name, v4obs.name]

                corr_1, _, dcorr_1, _ = CorrelationMatrix_TrendAndDetrend([v1_obs_one, v2_obs_one, v3_obs_one, v4_obs_one])
                array, darray = [corr_1], [dcorr_1]

                for i in range(len(Var1_IterModList_DF)):
                    corr_2, _, dcorr_2, _ = CorrelationMatrix_TrendAndDetrend( [v1_mod_one[i], v2_mod_one[i], v3_mod_one[i], v4_mod_one[i]])
                    array.append(corr_2)
                    darray.append(dcorr_2)

                array = np.asarray(array)
                darray = np.asarray(darray)
                self.score[region + 'CorrelationTrend'] = [i for i in np.abs((array)).sum(axis=1).sum(axis=1)/(np.prod((array).shape)/3.0)]
                self.score[region + 'CorrelationDetrend'] = [i for i in np.abs((darray)).sum(axis=1).sum(axis=1) /(np.prod((darray).shape)/3.0)]



                fig = plot_variable_matrix_trend_and_detrend(array, darray, variable_list, col_num=ModNum, site_name=self.RegionsName[region]+"[H]")
                fig.savefig(os.path.join(output_path, "%s_%s_correlation_box.png" % (ModName, region)),
                            bbox_inches='tight')

                self.score[region+'TimeSeries'] = [np.ma.mean(np.ma.masked_invalid(np.abs(v1_obs_one.values-x.values))) for x in v1_mod_one]

                self.score[region+'response2'] = [np.ma.mean(np.ma.masked_invalid(np.abs(v1_obs_one.values-x.values) + np.abs(v2_obs_one.values-y.values))) for x,y in zip(v1_mod_one, v2_mod_one)]

                self.score[region+'response4'] = [np.ma.mean(np.ma.masked_invalid(np.abs(v1_obs_one.values-x.values) + np.abs(v2_obs_one.values-y.values)+ np.abs(v3_obs_one.values-z.values)+ np.abs(v4_obs_one.values-k.values))) for x,y,z,k in zip(v1_mod_one, v2_mod_one, v3_mod_one, v4_mod_one)]



                fig2_variable = Plot_response2(v1_obs_one, v1_mod_one, v2_obs_one, v2_mod_one, XLabel=v1obs.name + '(' + v1obs.units + ')',
                           YLabel=v2obs.name + '(' + v2obs.units + ')', FigTitle=self.RegionsName[region], col=ModNum)
                fig2_variable.savefig(os.path.join(output_path, "%s_%s_response.png" % (ModName, region)),
                                      bbox_inches='tight')

                fig2_variable_error = Plot_response2_error(v1_obs_one, v1_mod_one, v2_obs_one, v2_mod_one,
                                     XLabel=v1obs.name + '(' + v1obs.units + ')',
                                     YLabel=v2obs.name + '(' + v2obs.units + ')', FigTitle=self.RegionsName[region], col=ModNum)
                fig2_variable_error.savefig(os.path.join(output_path, "%s_%s_response_error.png" % (ModName, region)),
                                            bbox_inches='tight')
                ### Seasonal Cases_response2
                fig2_variables1 = Plot_response2(v1_obs_h_DJF, v1_mod_h_DJF, v2_obs_h_DJF, v2_mod_h_DJF, XLabel=v1obs.name + '(' + v1obs.units + ')',
                           YLabel=v2obs.name + '(' + v2obs.units + ')', FigTitle='Response',s=1, col=ModNum)
                fig2_variables1.savefig(os.path.join(output_path, "%s_%s_response_s1.png" % (ModName, region)),
                                      bbox_inches='tight')

                fig2_variable_errors1 = Plot_response2_error(v1_obs_h_DJF, v1_mod_h_DJF, v2_obs_h_DJF, v2_mod_h_DJF,
                                     XLabel=v1obs.name + '(' + v1obs.units + ')',
                                     YLabel=v2obs.name + '(' + v2obs.units + ')', FigTitle=self.RegionsName[region],s=1, col=ModNum)
                fig2_variable_errors1.savefig(os.path.join(output_path, "%s_%s_response_error_s1.png" % (ModName, region)),
                                            bbox_inches='tight')

                fig2_variables2 = Plot_response2(v1_obs_h_MAM, v1_mod_h_MAM, v2_obs_h_MAM, v2_mod_h_MAM, XLabel=v1obs.name + '(' + v1obs.units + ')',
                           YLabel=v2obs.name + '(' + v2obs.units + ')', FigTitle='Response',s=2, col=ModNum)
                fig2_variables2.savefig(os.path.join(output_path, "%s_%s_response_s2.png" % (ModName, region)),
                                      bbox_inches='tight')

                fig2_variable_errors2 = Plot_response2_error(v1_obs_h_MAM, v1_mod_h_MAM, v2_obs_h_MAM, v2_mod_h_MAM,
                                     XLabel=v1obs.name + '(' + v1obs.units + ')',
                                     YLabel=v2obs.name + '(' + v2obs.units + ')', FigTitle=self.RegionsName[region],s=2, col=ModNum)
                fig2_variable_errors2.savefig(os.path.join(output_path, "%s_%s_response_error_s2.png" % (ModName, region)),
                                            bbox_inches='tight')

                fig2_variables3 = Plot_response2(v1_obs_h_JJA, v1_mod_h_JJA, v2_obs_h_JJA, v2_mod_h_JJA, XLabel=v1obs.name + '(' + v1obs.units + ')',
                           YLabel=v2obs.name + '(' + v2obs.units + ')', FigTitle=self.RegionsName[region],s=3, col=ModNum)
                fig2_variables3.savefig(os.path.join(output_path, "%s_%s_response_s3.png" % (ModName, region)),
                                      bbox_inches='tight')

                fig2_variable_errors3 = Plot_response2_error(v1_obs_h_JJA, v1_mod_h_JJA, v2_obs_h_JJA, v2_mod_h_JJA,
                                     XLabel=v1obs.name + '(' + v1obs.units + ')',
                                     YLabel=v2obs.name + '(' + v2obs.units + ')', FigTitle=self.RegionsName[region],s=3, col=ModNum)
                fig2_variable_errors3.savefig(os.path.join(output_path, "%s_%s_response_error_s3.png" % (ModName, region)),
                                            bbox_inches='tight')

                fig2_variables4 = Plot_response2(v1_obs_h_SON, v1_mod_h_SON, v2_obs_h_SON, v2_mod_h_SON, XLabel=v1obs.name + '(' + v1obs.units + ')',
                           YLabel=v2obs.name + '(' + v2obs.units + ')', FigTitle=self.RegionsName[region],s=4, col=ModNum)
                fig2_variables4.savefig(os.path.join(output_path, "%s_%s_response_s4.png" % (ModName, region)),
                                      bbox_inches='tight')

                fig2_variable_errors4 = Plot_response2_error(v1_obs_h_SON, v1_mod_h_SON, v2_obs_h_SON, v2_mod_h_SON,
                                     XLabel=v1obs.name + '(' + v1obs.units + ')',
                                     YLabel=v2obs.name + '(' + v2obs.units + ')', FigTitle=self.RegionsName[region],s=4, col=ModNum)
                fig2_variable_errors4.savefig(os.path.join(output_path, "%s_%s_response_error_s4.png" % (ModName, region)),
                                            bbox_inches='tight')

                plt.close('all')

                # continue

                print('Plot Response 4')
                fig4_variable = Plot_response4(v1_obs_one, v1_mod_one, v2_obs_one, v2_mod_one,v3_obs_one, v3_mod_one, v4_obs_one, v4_mod_one, FigTitle=self.RegionsName[region]+'\n'+v4obs.name + '(' + v4obs.units + ')',
                               XLabel=v1obs.name + '(' + v1obs.units + ')', YLabel=v2obs.name + '(' + v2obs.units + ')',
                               ZLabel=v3obs.name + '(' + v3obs.units + ')',
                               s=None, col_num=-1, Frequency ='M')
                fig4_variable.savefig(os.path.join(output_path, "%s_%s_response4.png" % (ModName, region)),
                                      bbox_inches='tight')

                fig10_s1 = Plot_response4(v1_obs_h_DJF, v1_mod_h_DJF, v2_obs_h_DJF, v2_mod_h_DJF,v3_obs_h_DJF, v3_mod_h_DJF, v4_obs_h_DJF, v4_mod_h_DJF, FigTitle=self.RegionsName[region]+'\n'+v4obs.name + '(' + v4obs.units + ')',
                               XLabel=v1obs.name + '(' + v1obs.units + ')', YLabel=v2obs.name + '(' + v2obs.units + ')',
                               ZLabel=v3obs.name + '(' + v3obs.units + ')',
                               s=1, col_num=-1, Frequency ='M')
                fig10_s1.savefig(os.path.join(output_path, "%s_%s_response4_s1.png" % (ModName, region)),
                                bbox_inches='tight')
                fig10_s2 = Plot_response4(v1_obs_h_MAM, v1_mod_h_MAM, v2_obs_h_MAM, v2_mod_h_MAM,v3_obs_h_MAM, v3_mod_h_MAM, v4_obs_h_MAM, v4_mod_h_MAM, FigTitle=self.RegionsName[region]+'\n'+v4obs.name + '(' + v4obs.units + ')',
                               XLabel=v1obs.name + '(' + v1obs.units + ')', YLabel=v2obs.name + '(' + v2obs.units + ')',
                               ZLabel=v3obs.name + '(' + v3obs.units + ')',
                               s=2, col_num=-1, Frequency ='M')
                fig10_s2.savefig(os.path.join(output_path, "%s_%s_response4_s2.png" % (ModName, region)),
                                bbox_inches='tight')

                fig10_s3 = Plot_response4(v1_obs_h_JJA, v1_mod_h_JJA, v2_obs_h_JJA, v2_mod_h_JJA,v3_obs_h_JJA, v3_mod_h_JJA, v4_obs_h_JJA, v4_mod_h_JJA, FigTitle=self.RegionsName[region]+'\n'+v4obs.name + '(' + v4obs.units + ')',
                               XLabel=v1obs.name + '(' + v1obs.units + ')', YLabel=v2obs.name + '(' + v2obs.units + ')',
                               ZLabel=v3obs.name + '(' + v3obs.units + ')',
                               s=3, col_num=-1, Frequency ='M')
                fig10_s3.savefig(os.path.join(output_path, "%s_%s_response4_s3.png" % (ModName, region)),
                                bbox_inches='tight')

                fig10_s4 = Plot_response4(v1_obs_h_SON, v1_mod_h_SON, v2_obs_h_SON, v2_mod_h_SON,v3_obs_h_SON, v3_mod_h_SON, v4_obs_h_SON, v4_mod_h_SON, FigTitle=self.RegionsName[region]+'\n'+v4obs.name + '(' + v4obs.units + ')',
                               XLabel=v1obs.name + '(' + v1obs.units + ')', YLabel=v2obs.name + '(' + v2obs.units + ')',
                               ZLabel=v3obs.name + '(' + v3obs.units + ')',
                               s=4, col_num=-1, Frequency ='M')
                fig10_s4.savefig(os.path.join(output_path, "%s_%s_response4_s4.png" % (ModName, region)),
                                bbox_inches='tight')

                plt.close('all')
                outputlist = []
                fig4_variable = PlotFourVariableCorr(v1_obs_one, v1_mod_one, v2_obs_one, v2_mod_one, v3_obs_one, v3_mod_one,
                                               v4_obs_one, v4_mod_one, FigTitle=self.RegionsName[region]+'\n'+v4obs.name + '(' + v4obs.units + ')',
                                               XLabel=v1obs.name + '(' + v1obs.units + ')',
                                               YLabel=v2obs.name + '(' + v2obs.units + ')',
                                               ZLabel=v3obs.name + '(' + v3obs.units + ')',
                                               s=None, col_num=-1, Frequency='M', score=outputlist)
                self.score[region + 'PartialCorrelation'] = [i for i in outputlist]


                fig4_variable.savefig(os.path.join(output_path, "%s_%s_corr4.png" % (ModName, region)),
                                      bbox_inches='tight')

                fig10_s1 = PlotFourVariableCorr(v1_obs_h_DJF, v1_mod_h_DJF, v2_obs_h_DJF, v2_mod_h_DJF, v3_obs_h_DJF,
                                          v3_mod_h_DJF, v4_obs_h_DJF, v4_mod_h_DJF,
                                          FigTitle=v4obs.name + '(' + v4obs.units + ')',
                                          XLabel=v1obs.name + '(' + v1obs.units + ')',
                                          YLabel=v2obs.name + '(' + v2obs.units + ')',
                                          ZLabel=v3obs.name + '(' + v3obs.units + ')',
                                          s=1, col_num=-1, Frequency='M')
                fig10_s1.savefig(os.path.join(output_path, "%s_%s_corr4_s1.png" % (ModName, region)),
                                 bbox_inches='tight')
                fig10_s2 = PlotFourVariableCorr(v1_obs_h_MAM, v1_mod_h_MAM, v2_obs_h_MAM, v2_mod_h_MAM, v3_obs_h_MAM,
                                          v3_mod_h_MAM, v4_obs_h_MAM, v4_mod_h_MAM,
                                          FigTitle=v4obs.name + '(' + v4obs.units + ')',
                                          XLabel=v1obs.name + '(' + v1obs.units + ')',
                                          YLabel=v2obs.name + '(' + v2obs.units + ')',
                                          ZLabel=v3obs.name + '(' + v3obs.units + ')',
                                          s=2, col_num=-1, Frequency='M')
                fig10_s2.savefig(os.path.join(output_path, "%s_%s_corr4_s2.png" % (ModName, region)),
                                 bbox_inches='tight')

                fig10_s3 = PlotFourVariableCorr(v1_obs_h_JJA, v1_mod_h_JJA, v2_obs_h_JJA, v2_mod_h_JJA, v3_obs_h_JJA,
                                          v3_mod_h_JJA, v4_obs_h_JJA, v4_mod_h_JJA,
                                          FigTitle=v4obs.name + '(' + v4obs.units + ')',
                                          XLabel=v1obs.name + '(' + v1obs.units + ')',
                                          YLabel=v2obs.name + '(' + v2obs.units + ')',
                                          ZLabel=v3obs.name + '(' + v3obs.units + ')',
                                          s=3, col_num=-1, Frequency='M')
                fig10_s3.savefig(os.path.join(output_path, "%s_%s_corr4_s3.png" % (ModName, region)),
                                 bbox_inches='tight')

                fig10_s4 = PlotFourVariableCorr(v1_obs_h_SON, v1_mod_h_SON, v2_obs_h_SON, v2_mod_h_SON, v3_obs_h_SON,
                                          v3_mod_h_SON, v4_obs_h_SON, v4_mod_h_SON,
                                          FigTitle=v4obs.name + '(' + v4obs.units + ')',
                                          XLabel=v1obs.name + '(' + v1obs.units + ')',
                                          YLabel=v2obs.name + '(' + v2obs.units + ')',
                                          ZLabel=v3obs.name + '(' + v3obs.units + ')',
                                          s=4, col_num=-1, Frequency='M')
                fig10_s4.savefig(os.path.join(output_path, "%s_%s_corr4_s4.png" % (ModName, region)),
                                 bbox_inches='tight')

                plt.close('all')

            # continue

            if len(Var1_IterModList_DF) == 1:
                obsDF, modDF = Var1_obs_DF, Var1_IterModList_DF[0]
                print('.......  Processing Single Site Post ....... ')

                for region in self.regions: #len(obsDF.columns)

                    print('Draw the BaseMap for the region')
                    LatList = [self.lats[siteid] for siteid in self.ind_regions[region]]
                    LonList = [self.lons[siteid] for siteid in self.ind_regions[region]]
                    SiteNameList = ['Lat: ' + str(lat)[:5] + ', Lon: ' + str(lon)[:5] for lat, lon in
                                    zip(LatList, LonList)]

                    obs_one, mod_one = obsDF[list(self.ind_regions[region])].mean(axis=1), modDF[list(self.ind_regions[region])].mean(axis=1) # Original Data for one column, which stands for one site

                    figLegend = BaseMap_Plot(LatList, LonList, SiteNameList, RegionName=self.RegionsName[region])
                    figLegend.savefig(os.path.join(output_path, "%s_%s_timeseries_legend.png" % (ModName, region)),
                                      bbox_inches='tight')


                    plt.close("all")
                    print('Processing_TimeSeries: SITE('+str(siteid)+')')
                    # Print Basically Hourly Data/Orignal Time Scale
                    obs_h, mod_h = obs_one.resample('H').mean(), mod_one.resample('H').mean()  #  Hourly Mean
                    fig_TimeSeries_hourly = TimeSeriesOneModel_Plot(obs_h, mod_h, FigTitle=self.RegionsName[region],
                                                                    XLabel='Year' + '(Hourly)',
                                                                    YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                    ModLegend=ModName)
                    fig_TimeSeries_hourly.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_hourly.png" % (ModName, region)),
                        bbox_inches='tight')

                    obs_d, mod_d = obs_one.resample('D').mean(), mod_one.resample('D').mean() #  Daily Mean
                    fig_TimeSeries_daily = TimeSeriesOneModel_Plot(obs_d, mod_d, FigTitle=self.RegionsName[region],
                                                                   XLabel='Year' + '(Daily)',
                                                                   YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                   ModLegend=ModName)
                    fig_TimeSeries_daily.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_daily.png" % (ModName, region)),
                        bbox_inches='tight')

                    obs_m, mod_m = obs_one.resample('M').mean(), mod_one.resample('M').mean() #  Monthly Mean
                    fig_TimeSeries_monthly = TimeSeriesOneModel_Plot(obs_m, mod_m, FigTitle=self.RegionsName[region],
                                                                     XLabel='Year' + '(Monthly)',
                                                                     YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                     ModLegend=ModName)
                    fig_TimeSeries_monthly.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_monthly.png" % (ModName, region)),
                        bbox_inches='tight')

                    obs_a, mod_a = obs_one.resample('A').mean(), mod_one.resample('A').mean() #  Yearly Mean
                    fig_TimeSeries_yearly = TimeSeriesOneModel_Plot(obs_a, mod_a, FigTitle=self.RegionsName[region],
                                                                    XLabel='Year' + '(Annual)',
                                                                    YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                    ModLegend=ModName)
                    fig_TimeSeries_yearly.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_yearly.png" % (ModName, region)),
                        bbox_inches='tight')

                    obs_s, mod_s = obs_one.resample('BQS').mean(), mod_one.resample('BQS').mean() #  Yearly Mean
                    fig_TimeSeries_seasonly = TimeSeriesOneModel_Plot(obs_s, mod_s,
                                                                      FigTitle=self.RegionsName[region],
                                                                      XLabel='Year' + '(Seasonly)',
                                                                      YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                      ModLegend=ModName)
                    fig_TimeSeries_seasonly.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_seasonly.png" % (ModName, region)),
                        bbox_inches='tight')
                    m


                    plt.close("all")

                    ########### Mask all the Seasonal Hourly Data  ###########
                    obs_h_DJF, obs_h_MAM, obs_h_JJA, obs_h_SON = obs_h.copy(), obs_h.copy(), obs_h.copy(), obs_h.copy() # PD is mutable
                    mod_h_DJF, mod_h_MAM, mod_h_JJA, mod_h_SON = mod_h.copy(), mod_h.copy(), mod_h.copy(), mod_h.copy() # Hard COPY!!!

                    obs_h_DJF[(obs_h_DJF.index.month >= 2) & (obs_h_DJF.index.month <= 11)] = None
                    mod_h_DJF[(mod_h_DJF.index.month >= 2) & (mod_h_DJF.index.month <= 11)] = None

                    obs_h_MAM[(obs_h_MAM.index.month <= 2) | (obs_h_MAM.index.month >= 5)] = None
                    mod_h_MAM[(mod_h_MAM.index.month <= 2) | (mod_h_MAM.index.month >= 5)] = None

                    obs_h_JJA[(obs_h_JJA.index.month <= 5) | (obs_h_JJA.index.month >= 9)] = None
                    mod_h_JJA[(mod_h_JJA.index.month <= 5) | (mod_h_JJA.index.month >= 9)] = None

                    obs_h_SON[(obs_h_SON.index.month <= 8) | (obs_h_SON.index.month == 12)] = None
                    mod_h_SON[(mod_h_SON.index.month <= 8) | (mod_h_SON.index.month == 12)] = None


                    print('Processing_Season_TimeSeries: SITE(' + str(siteid) + ')')

                    obs_a_DJF, mod_a_DJF = obs_h_DJF.resample('A').mean(), mod_h_DJF.resample('A').mean()
                    obs_a_MAM, mod_a_MAM = obs_h_MAM.resample('A').mean(), mod_h_MAM.resample('A').mean()
                    obs_a_JJA, mod_a_JJA = obs_h_JJA.resample('A').mean(), mod_h_JJA.resample('A').mean()
                    obs_a_SON, mod_a_SON = obs_h_SON.resample('A').mean(), mod_h_SON.resample('A').mean()

                    figannual_s1 = TimeSeriesOneModel_Plot(obs_a_DJF, mod_a_DJF, FigTitle=self.RegionsName[region],
                                                                    XLabel='Year' + '(Annual, DJF)',
                                                                    YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                    ModLegend=ModName)
                    figannual_s1.savefig(os.path.join(output_path, "%s_%s_timeseries_s1.png" % (ModName, region)),
                                         bbox_inches='tight')

                    figannual_s2 = TimeSeriesOneModel_Plot(obs_a_MAM, mod_a_MAM, FigTitle=self.RegionsName[region],
                                                                    XLabel='Year' + '(Annual, MAM)',
                                                                    YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                    ModLegend=ModName)

                    figannual_s2.savefig(os.path.join(output_path, "%s_%s_timeseries_s2.png" % (ModName, region)),
                                         bbox_inches='tight')
                    figannual_s3 = TimeSeriesOneModel_Plot(obs_a_JJA, mod_a_JJA, FigTitle=self.RegionsName[region],
                                                                    XLabel='Year' + '(Annual, JJA)',
                                                                    YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units+ ')',
                                                                    ModLegend=ModName)

                    figannual_s3.savefig(os.path.join(output_path, "%s_%s_timeseries_s3.png" % (ModName, region)),
                                         bbox_inches='tight')
                    figannual_s4 = TimeSeriesOneModel_Plot(obs_a_SON, mod_a_SON, FigTitle=self.RegionsName[region],
                                                                    XLabel='Year' + '(Annual, SON)',
                                                                    YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                    ModLegend=ModName)
                    figannual_s4.savefig(os.path.join(output_path, "%s_%s_timeseries_s4.png" % (ModName, region)),
                                         bbox_inches='tight')
                    plt.close("all")

                    print('Processing_Cycles: SITE(' + str(siteid) + ')')

                    fig_TimeSeries_cycle_hourofday = CyclesOneModel_Plot(obs_one, mod_one, FrequencyRule='H',
                                                                         FigTitle=self.RegionsName[region],
                                                                         XLabel='Hours of A Day(Annual)',
                                                                         YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                         ModLegend=ModName)
                    fig_TimeSeries_cycle_hourofday.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_hourofday.png" % (ModName, region)),
                        bbox_inches='tight')

                    fig_TimeSeries_cycle_dayofyear = CyclesOneModel_Plot(obs_one, mod_one, FrequencyRule='D', FigTitle=self.RegionsName[region],
                                        XLabel='Days of A Year', YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')', ModLegend=ModName)
                    fig_TimeSeries_cycle_dayofyear.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_dayofyear.png" % (ModName, region)),
                        bbox_inches='tight')

                    fig_TimeSeries_cycle_monthofyear = CyclesOneModel_Plot(obs_one, mod_one, FrequencyRule='M',
                                                                         FigTitle=self.RegionsName[region],
                                                                         XLabel='Months of A Year',
                                                                         YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                         ModLegend=ModName)
                    fig_TimeSeries_cycle_monthofyear.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_monthofyear.png" % (ModName, region)),
                        bbox_inches='tight')

                    fig_TimeSeries_cycle_seasonofyear =CyclesOneModel_Plot(obs_one, mod_one, FrequencyRule='S',
                                                                         FigTitle=self.RegionsName[region],
                                                                         XLabel='Seasons of A Year',
                                                                         YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                         ModLegend=ModName)
                    fig_TimeSeries_cycle_seasonofyear.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_seasonofyear.png" % (ModName, region)),
                        bbox_inches='tight')

                    plt.close("all")



                    fig_TimeSeries_cycle_hourofday = CyclesOneModel_Plot(obs_h_DJF, mod_h_DJF, FrequencyRule='H',
                                                                         FigTitle=self.RegionsName[region],
                                                                         XLabel='Hours of A Day(DJF)',
                                                                         YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                         ModLegend=ModName)
                    fig_TimeSeries_cycle_hourofday.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_hourofday_s1.png" % (ModName, region)),
                        bbox_inches='tight')

                    fig_TimeSeries_cycle_hourofday = CyclesOneModel_Plot(obs_h_MAM, mod_h_MAM, FrequencyRule='H',
                                                                         FigTitle=self.RegionsName[region],
                                                                         XLabel='Hours of A Day(MAM)',
                                                                         YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                         ModLegend=ModName)
                    fig_TimeSeries_cycle_hourofday.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_hourofday_s2.png" % (ModName, region)),
                        bbox_inches='tight')

                    fig_TimeSeries_cycle_hourofday = CyclesOneModel_Plot(obs_h_JJA, mod_h_JJA, FrequencyRule='H',
                                                                         FigTitle=self.RegionsName[region],
                                                                         XLabel='Hours of A Day(JJA)',
                                                                         YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                         ModLegend=ModName)
                    fig_TimeSeries_cycle_hourofday.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_hourofday_s3.png" % (ModName, region)),
                        bbox_inches='tight')

                    fig_TimeSeries_cycle_hourofday = CyclesOneModel_Plot(obs_h_SON, obs_h_SON, FrequencyRule='H',
                                                                         FigTitle=self.RegionsName[region],
                                                                         XLabel='Hours of A Day(SON)',
                                                                         YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                         ModLegend=ModName)
                    fig_TimeSeries_cycle_hourofday.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_hourofday_s4.png" % (ModName, region)),
                        bbox_inches='tight')

                    plt.close("all")

                    print('Processing_PDFCDF: SITE(' + str(siteid) + ')')
                    fig_PDF_CDF = PDFCDFOneModel_Plot(obs_h, mod_h, FigTitle=self.RegionsName[region],
                                        XLabel='Hourly ' + ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')', ModLegend='MOD')
                    fig_PDF_CDF.savefig(os.path.join(output_path, "%s_%s_PDFCDF.png" % (ModName, region)),
                                        bbox_inches='tight')
                    plt.close("all")

                    print('Processing_TaylorGram: SITE(' + str(siteid) + ')')

                    fig_TimeSeries_TaylorGram = Plot_TimeSeries_TaylorGram(obs_h, [mod_h], col_num=ModNum, FigName='Time Series(H, D, M)')
                    fig_TimeSeries_TaylorGram.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_taylorgram.png" % (ModName, region)))

                    figannual_taylorgram = Plot_TimeSeries_TaylorGram_annual(obs_h, [mod_h], col_num=ModNum, FigName='Time Series(Four seasons)')
                    figannual_taylorgram.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_taylorgram_annual.png" % (ModName, region)))

                    fighourofda_taylorgram = Plot_TimeSeries_TaylorGram_hourofday(obs_h, [mod_h], col_num=ModNum, FigName='Cycles(Hours of A Day)')
                    fighourofda_taylorgram.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_taylorgram_hourofday.png" % (ModName, region)))

                    figcycles_taylorgram = Plot_TimeSeries_TaylorGram_cycles(obs_h, [mod_h], col_num=ModNum, FigName='Cycles')
                    figcycles_taylorgram.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_taylorgram_cycles.png" % (ModName, region)))

                    plt.close("all")


            elif len(IterNameModList) >= 2:
                obsDF = Var1_obs_DF
                modDF = Var1_IterModList_DF
                print('.......  Processing Multi-Models Post ....... ')


                for region in self.regions: #len(obsDF.columns)

                    obs_one, mod_one = obsDF[list(self.ind_regions[region])].mean(axis=1), [m[list(self.ind_regions[region])].mean(axis=1) for m in modDF]# Original Data for one column, which stands for one site

                    print('Draw the BaseMap for the region')
                    LatList = [self.lats[siteid] for siteid in self.ind_regions[region]]
                    LonList = [self.lons[siteid] for siteid in self.ind_regions[region]]
                    SiteNameList = ['Lat: ' + str(lat)[:5] + ', Lon: ' + str(lon)[:5] for lat, lon in
                                    zip(LatList, LonList)]


                    figLegend = BaseMap_Plot(LatList, LonList, SiteNameList, RegionName=self.RegionsName[region])
                    figLegend.savefig(os.path.join(output_path, "%s_%s_timeseries_legend.png" % (ModName, region)),
                                      bbox_inches='tight')


                    print('Processing_MoreModel_TimeSeries: SITE(' + str(siteid) + ')')
                    obs_h, mod_h = obs_one.resample('H').mean(), [m.resample('H').mean() for m in mod_one]  # Hourly Mean
                    fig_TimeSeries_hourly = TimeSeriesMoreModel_Plot(obs_h, mod_h, FigTitle=self.RegionsName[region],
                                                                    XLabel='Year' + '(Hourly)',
                                                                    YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                    ModLegend=ModName)
                    fig_TimeSeries_hourly.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_hourly.png" % (ModName, region)),
                        bbox_inches='tight')

                    obs_d, mod_d = obs_one.resample('D').mean(), [m.resample('D').mean() for m in mod_one]  # Daily Mean
                    fig_TimeSeries_daily = TimeSeriesMoreModel_Plot(obs_d, mod_d, FigTitle=self.RegionsName[region],
                                                                   XLabel='Year' + '(Daily)',
                                                                   YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                   ModLegend=ModName)
                    fig_TimeSeries_daily.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_daily.png" % (ModName, region)),
                        bbox_inches='tight')

                    obs_m, mod_m = obs_one.resample('M').mean(), [m.resample('M').mean()  for m in mod_one]  # Monthly Mean
                    fig_TimeSeries_monthly = TimeSeriesMoreModel_Plot(obs_m, mod_m, FigTitle=self.RegionsName[region],
                                                                     XLabel='Year' + '(Monthly)',
                                                                     YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                     ModLegend=ModName)
                    fig_TimeSeries_monthly.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_monthly.png" % (ModName, region)),
                        bbox_inches='tight')

                    obs_a, mod_a = obs_one.resample('A').mean(), [m.resample('A').mean()  for m in mod_one]  # Yearly Mean
                    fig_TimeSeries_yearly = TimeSeriesMoreModel_Plot(obs_a, mod_a, FigTitle=self.RegionsName[region],
                                                                    XLabel='Year' + '(Annual)',
                                                                    YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                    ModLegend=ModName)
                    fig_TimeSeries_yearly.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_yearly.png" % (ModName, region)),
                        bbox_inches='tight')

                    obs_s, mod_s = obs_one.resample('BQS').mean(), [m.resample('BQS').mean()  for m in mod_one]  # Yearly Mean
                    fig_TimeSeries_seasonly = TimeSeriesMoreModel_Plot(obs_s, mod_s,
                                                                      FigTitle=self.RegionsName[region],
                                                                      XLabel='Year' + '(Seasonly)',
                                                                      YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                      ModLegend=ModName)
                    fig_TimeSeries_seasonly.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_seasonly.png" % (ModName, region)),
                        bbox_inches='tight')
                    m

                    plt.close("all")

                    ########### Mask all the Seasonal Hourly Data  ###########

                    obs_h_DJF, obs_h_MAM, obs_h_JJA, obs_h_SON, mod_h_DJF, mod_h_MAM, mod_h_JJA, mod_h_SON = MaskSeasonData(obs_h, mod_h)

                    # obs_h_DJF, obs_h_MAM, obs_h_JJA, obs_h_SON = obs_h.copy(), obs_h.copy(), obs_h.copy(), obs_h.copy()  # PD is mutable
                    # mod_h_DJF, mod_h_MAM, mod_h_JJA, mod_h_SON = [m.copy() for m in mod_h], [m.copy() for m in mod_h], [m.copy() for m in mod_h], [m.copy() for m in mod_h] # Hard COPY!!!
                    #
                    # obs_h_DJF[(obs_h_DJF.index.month >= 2) & (obs_h_DJF.index.month <= 11)] = None
                    # for m in mod_h_DJF:
                    #     m[(m.index.month >= 2) & (m.index.month <= 11)] = None
                    #
                    # obs_h_MAM[(obs_h_MAM.index.month <= 2) | (obs_h_MAM.index.month >= 5)] = None
                    # for m in mod_h_DJF:
                    #     m[(m.index.month <= 2) | (m.index.month >= 5)] = None
                    #
                    # obs_h_JJA[(obs_h_JJA.index.month <= 5) | (obs_h_JJA.index.month >= 9)] = None
                    # for m in mod_h_JJA:
                    #     m[(m.index.month <= 5) | (m.index.month >= 9)] = None
                    #
                    # obs_h_SON[(obs_h_SON.index.month <= 8) | (obs_h_SON.index.month == 12)] = None
                    # for m in mod_h_SON:
                    #     m[(m.index.month <= 8) | (m.index.month == 12)] = None

                    print('Processing_MoreModel_Season_TimeSeries: SITE(' + str(siteid) + ')')

                    obs_a_DJF, mod_a_DJF = obs_h_DJF.resample('A').mean(), [m.resample('A').mean() for m in mod_h_DJF]
                    obs_a_MAM, mod_a_MAM = obs_h_MAM.resample('A').mean(), [m.resample('A').mean() for m in mod_h_MAM]
                    obs_a_JJA, mod_a_JJA = obs_h_JJA.resample('A').mean(), [m.resample('A').mean() for m in mod_h_JJA]
                    obs_a_SON, mod_a_SON = obs_h_SON.resample('A').mean(), [m.resample('A').mean() for m in mod_h_SON]

                    figannual_s1 = TimeSeriesMoreModel_Plot(obs_a_DJF, mod_a_DJF, FigTitle=self.RegionsName[region],
                                                           XLabel='Year' + '(Annual, DJF)',
                                                           YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                           ModLegend=ModName)
                    figannual_s1.savefig(os.path.join(output_path, "%s_%s_timeseries_s1.png" % (ModName, region)),
                                         bbox_inches='tight')

                    figannual_s2 = TimeSeriesMoreModel_Plot(obs_a_MAM, mod_a_MAM, FigTitle=self.RegionsName[region],
                                                           XLabel='Year' + '(Annual, MAM)',
                                                           YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                           ModLegend=ModName)

                    figannual_s2.savefig(os.path.join(output_path, "%s_%s_timeseries_s2.png" % (ModName, region)),
                                         bbox_inches='tight')
                    figannual_s3 = TimeSeriesMoreModel_Plot(obs_a_JJA, mod_a_JJA, FigTitle=self.RegionsName[region],
                                                           XLabel='Year' + '(Annual, JJA)',
                                                           YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                           ModLegend=ModName)

                    figannual_s3.savefig(os.path.join(output_path, "%s_%s_timeseries_s3.png" % (ModName, region)),
                                         bbox_inches='tight')
                    figannual_s4 = TimeSeriesMoreModel_Plot(obs_a_SON, mod_a_SON, FigTitle=self.RegionsName[region],
                                                           XLabel='Year' + '(Annual, SON)',
                                                           YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                           ModLegend=ModName)
                    figannual_s4.savefig(os.path.join(output_path, "%s_%s_timeseries_s4.png" % (ModName, region)),
                                         bbox_inches='tight')
                    plt.close("all")

                    print('Processing_MoreModel_Cycles: SITE(' + str(siteid) + ')')

                    fig_TimeSeries_cycle_hourofday = CyclesMoreModel_Plot(obs_one, mod_one, FrequencyRule='H',
                                                                         FigTitle=self.RegionsName[region],
                                                                         XLabel='Hours of A Day',
                                                                         YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                         ModLegend=ModName)
                    fig_TimeSeries_cycle_hourofday.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_hourofday.png" % (ModName, region)),
                        bbox_inches='tight')
                    # print('Hours of A day')
                    fig_TimeSeries_cycle_dayofyear = CyclesMoreModel_Plot(obs_one, mod_one, FrequencyRule='D',
                                                                         FigTitle=self.RegionsName[region],
                                                                         XLabel='Days of A Year(Annual)',
                                                                         YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                         ModLegend=ModName)
                    fig_TimeSeries_cycle_dayofyear.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_dayofyear.png" % (ModName, region)),
                        bbox_inches='tight')
                    # print('Days of A Year')
                    fig_TimeSeries_cycle_monthofyear = CyclesMoreModel_Plot(obs_one, mod_one, FrequencyRule='M',
                                                                           FigTitle=self.RegionsName[region],
                                                                           XLabel='Months of A Year',
                                                                           YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                           ModLegend=ModName)
                    fig_TimeSeries_cycle_monthofyear.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_monthofyear.png" % (ModName, region)),
                        bbox_inches='tight')
                    # print('Months of A Year')
                    fig_TimeSeries_cycle_seasonofyear = CyclesMoreModel_Plot(obs_one, mod_one, FrequencyRule='S',
                                                                            FigTitle=self.RegionsName[region],
                                                                            XLabel='Seasons of A Year',
                                                                            YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                            ModLegend=ModName)
                    fig_TimeSeries_cycle_seasonofyear.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_seasonofyear.png" % (ModName, region)),
                        bbox_inches='tight')
                    plt.close("all")

                    fig_TimeSeries_cycle_hourofday = CyclesMoreModel_Plot(obs_h_DJF, mod_h_DJF, FrequencyRule='H',
                                                                         FigTitle=self.RegionsName[region],
                                                                         XLabel='Hours of A Day(DJF)',
                                                                         YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                         ModLegend=ModName)
                    fig_TimeSeries_cycle_hourofday.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_hourofday_s1.png" % (ModName, region)),
                        bbox_inches='tight')

                    fig_TimeSeries_cycle_hourofday = CyclesMoreModel_Plot(obs_h_MAM, mod_h_MAM, FrequencyRule='H',
                                                                         FigTitle=self.RegionsName[region],
                                                                         XLabel='Hours of A Day(MAM)',
                                                                         YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                         ModLegend=ModName)
                    fig_TimeSeries_cycle_hourofday.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_hourofday_s2.png" % (ModName, region)),
                        bbox_inches='tight')

                    fig_TimeSeries_cycle_hourofday = CyclesMoreModel_Plot(obs_h_JJA, mod_h_JJA, FrequencyRule='H',
                                                                         FigTitle=self.RegionsName[region],
                                                                         XLabel='Hours of A Day(JJA)',
                                                                         YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                         ModLegend=ModName)
                    fig_TimeSeries_cycle_hourofday.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_hourofday_s3.png" % (ModName, region)),
                        bbox_inches='tight')

                    fig_TimeSeries_cycle_hourofday = CyclesMoreModel_Plot(obs_h_SON, mod_h_SON, FrequencyRule='H',
                                                                         FigTitle=self.RegionsName[region],
                                                                         XLabel='Hours of A Day(SON)',
                                                                         YLabel=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                                         ModLegend=ModName)
                    fig_TimeSeries_cycle_hourofday.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_hourofday_s4.png" % (ModName, region)),
                        bbox_inches='tight')

                    plt.close("all")

                    print('Processing_MoreModel_PDFCDF: SITE(' + str(siteid) + ')')
                    fig_PDF_CDF = PDFCDFMoreModel_Plot(obs_h, mod_h, FigTitle=self.RegionsName[region],
                                                      XLabel='Hourly ' + ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                                      ModLegend='MOD')
                    fig_PDF_CDF.savefig(os.path.join(output_path, "%s_%s_PDFCDF.png" % (ModName, region)),
                                        bbox_inches='tight')
                    plt.close("all")

                    print('Processing_MoreModel_TaylorGram: SITE(' + str(siteid) + ')')

                    fig_TimeSeries_TaylorGram = Plot_TimeSeries_TaylorGram(obs_h, mod_h, col_num=-1,
                                                                           FigName='Time Series(H, D, M)')
                    fig_TimeSeries_TaylorGram.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_taylorgram.png" % (ModName, region)))

                    figannual_taylorgram = Plot_TimeSeries_TaylorGram_annual(obs_h, mod_h, col_num=-1,
                                                                             FigName='Time Series(Four seasons)')
                    figannual_taylorgram.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_taylorgram_annual.png" % (ModName, region)))

                    fighourofda_taylorgram = Plot_TimeSeries_TaylorGram_hourofday(obs_h, mod_h, col_num=-1,
                                                                                  FigName='Cycles(Hours of A Day)')
                    fighourofda_taylorgram.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_taylorgram_hourofday.png" % (ModName, region)))

                    figcycles_taylorgram = Plot_TimeSeries_TaylorGram_cycles(obs_h, mod_h, col_num=-1,
                                                                             FigName='Cycles')
                    figcycles_taylorgram.savefig(
                        os.path.join(output_path, "%s_%s_timeseries_taylorgram_cycles.png" % (ModName, region)))

                    plt.close("all")



                    obs_m, mod_m = obs_one.resample('M').mean(), [m.resample('M').mean() for m in mod_one]

                    fig_Wavelet = Plot_Wavelet(obs_m.dropna(), matplotlib.dates.date2num(obs_m.dropna().index.date),
                                               Label=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')',
                                            FigTitle='OBS:' + self.RegionsName[region])
                    for m in IterNameModList:
                        fig_Wavelet.savefig(os.path.join(output_path, "%s_%s_wavelet.png" % (m, region)),
                                            bbox_inches='tight')
                    list_im = []
                    for i, mod in enumerate(mod_m):
                        fig_Wavelet_mod = Plot_Wavelet(mod.dropna(), matplotlib.dates.date2num(mod.dropna().index.date),
                                                   Label=ILAMB_Var_OBS.name + '(' + ILAMB_Var_OBS.units + ')', FigTitle=IterNameModList[i]+': ' + self.RegionsName[region])
                        fig_Wavelet_mod.savefig(
                            os.path.join(output_path, ("%s_%s_wavelet_Mod0.png") % (IterNameModList[i], region)),
                            bbox_inches='tight')
                        list_im.append((output_path + "%s_%s_wavelet_Mod0.png") % (IterNameModList[i], region))

                    imgs = [PIL.Image.open(i) for i in list_im]
                    x_axis_pictures_number = 1
                    while len(imgs) % x_axis_pictures_number != 0:
                        im = Image.new('RGB', (0, 0), color=tuple((np.random.rand(3) * 255).astype(np.uint8)))
                        imgs.append(im)
                    imgs_comb = pil_grid(imgs, x_axis_pictures_number)
                    # imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
                    imgs_comb.save((output_path + "/%s_%s_wavelet_Mod0.png") % (IterNameModList[-1], region))
                    plt.close("all")

                    #
                #         for modnumber, mname in enumerate(mmname):
                #             for regionid, bigregion in enumerate(self.regions[self.sitenumber:]):
                #                 timesseries = ['_timeseries_hourly.png', '_timeseries_daily.png', '_timeseries_monthly.png',
                #                                '_timeseries_yearly.png', '_timeseries_seasonly.png', "_timeseries_s1.png",
                #                                "_timeseries_s2.png", "_timeseries_s3.png", "_timeseries_s4.png"]
                #                 Taylorgrams =['_timeseries_taylorgram.png', '_timeseries_taylorgram_annual.png','_timeseries_taylorgram_hourofday.png', '_timeseries_taylorgram_cycles.png']
                #                 cycles = ['_timeseries_hourofday.png', '_timeseries_dayofyear.png', '_timeseries_monthofyear.png',
                #                           '_timeseries_seasonofyear.png', '_timeseries_hourofday_s1.png',
                #                           '_timeseries_hourofday_s2.png',
                #                           '_timeseries_hourofday_s3.png', '_timeseries_hourofday_s4.png']
                #                 pdfcdf = ['_PDFCDF.png']
                #                 frequency = ['_wavelet.png',
                #                              '_wavelet_Mod0.png']  # '_wavelet_Mod2.png', # '_wavelet_Mod3.png'  '_IMF1.png', '_IMF2.png', '_IMF3.png', '_IMF4.png'
                #                 response = ['_response.png', '_response_s1.png', '_response_s2.png', '_response_s3.png',
                #                             '_response_s4.png', '_response_error_s1.png'
                #                     , '_response_error_s2.png', '_response_error_s3.png', '_response_error_s4.png',
                #                             '_response_error.png',
                #                             '_response4.png', '_response4_s1.png', '_response4_s2.png',
                #                             '_response4_s3.png', '_response4_s4.png', '_corr4.png', '_correlation_box.png']
                #                 allmetric = timesseries + cycles + pdfcdf + frequency + response + Taylorgrams
                #                 for metric in allmetric:
                #                     list_im = []
                #                     for siteid, siteregion in enumerate(self.regions[:self.sitenumber]):
                #                         siteregion = str(siteid)
                #                         # print('site lat, lon', self.lats[siteid], self.lons[siteid])
                #                         # print('lat bound', self.lowerlatbound[regionid], self.upperlatbound[regionid])
                #                         # print('lon bound', self.lowerlonbound[regionid], self.upperlonbound[regionid])
                #
                #                         if self.lats[siteid] < self.upperlatbound[regionid] and self.lats[siteid] > self.lowerlatbound[
                #                             regionid] and self.lons[siteid] < self.upperlonbound[regionid] and self.lons[siteid] > \
                #                                 self.lowerlonbound[regionid]:
                #                             # print('In region',siteid, siteregion, regionid, siteregion)
                #                             list_im.append((output_path + "%s_%s" + metric) % (mname, siteregion))
                #                     imgs = [PIL.Image.open(i) for i in list_im]
                #                     x_axis_pictures_number = 2
                #                     while len(imgs) % x_axis_pictures_number != 0:
                #                         im = Image.new('RGB', (0, 0), color=tuple((np.random.rand(3) * 255).astype(np.uint8)))
                #                         imgs.append(im)
                #                     imgs_comb = pil_grid(imgs, x_axis_pictures_number)
                #                     imgs_comb.save((output_path + "%s_%s" + metric) % (mname, bigregion))
                #                 list_im = []
                #                 list_im.append((output_path + "%s_%s" + '_timeseries_legend.png') % (mname, siteregion))
                #                 imgs = [PIL.Image.open(i) for i in list_im]
                #                 imgs_comb = pil_grid(imgs, x_axis_pictures_number)
                #                 imgs_comb.save((output_path + "%s_%s" + '_timeseries_legend.png') % (mname, bigregion))

        page.addFigure("Time series", ('(Time Series)' + "Legend"), output_path + "MNAME_RNAME_timeseries_legend.png",
                       legend=False)

        page.addFigure("Time series", ('(Time Series)' + "Hourly"), output_path + "MNAME_RNAME_timeseries_hourly.png",
                       legend=False)

        page.addFigure("Time series", ('(Time Series)' + "Daily"), output_path + "MNAME_RNAME_timeseries_daily.png",
                       legend=False)

        page.addFigure("Time series", ('(Time Series)' + "Monthly"), output_path + "MNAME_RNAME_timeseries_monthly.png",
                       legend=False)

        page.addFigure("Time series", ('(Time Series)' + "Seasonly"),
                       output_path + "MNAME_RNAME_timeseries_seasonly.png",
                       legend=False)

        page.addFigure("Time series", ('(Time Series)' + "Taylor"),
                       output_path + "MNAME_RNAME_timeseries_taylorgram.png",
                       legend=False)

        page.addFigure("Time series(Annually)", ('(Time Series)' + "Yearly"),
                       output_path + "MNAME_RNAME_timeseries_yearly.png",
                       legend=False)

        page.addFigure("Time series(Annually)", ('(Time Series)' + "Season 1"),
                       output_path + "MNAME_RNAME_timeseries_s1.png",
                       legend=False)

        page.addFigure("Time series(Annually)", ('(Time Series)' + "Season 2"),
                       output_path + "MNAME_RNAME_timeseries_s2.png",
                       legend=False)

        page.addFigure("Time series(Annually)", ('(Time Series)' + "Season 3"),
                       output_path + "MNAME_RNAME_timeseries_s3.png",
                       legend=False)

        page.addFigure("Time series(Annually)", ('(Time Series)' + "Season 4"),
                       output_path + "MNAME_RNAME_timeseries_s4.png",
                       legend=False)

        page.addFigure("Time series(Annually)", ('(Time Series)' + "TaylorAnnual"),
                       output_path + "MNAME_RNAME_timeseries_taylorgram_annual.png",
                       legend=False)

        page.addFigure("PDF CDF", ('(PDF & CDF)' + "PDF & CDF"), output_path + "MNAME_RNAME_PDFCDF.png",
                       legend=False)

        page.addFigure("Cycles mean(seasonly)", ("Time Series"), output_path + "MNAME_RNAME_timeseries_hourofday.png",
                       legend=False)

        page.addFigure("Cycles mean(seasonly)", ("Season 1"), output_path + "MNAME_RNAME_timeseries_hourofday_s1.png",
                       legend=False)

        page.addFigure("Cycles mean(seasonly)", ("Season 2"), output_path + "MNAME_RNAME_timeseries_hourofday_s2.png",
                       legend=False)

        page.addFigure("Cycles mean(seasonly)", ("Season 3"), output_path + "MNAME_RNAME_timeseries_hourofday_s3.png",
                       legend=False)

        page.addFigure("Cycles mean(seasonly)", ("Season 4"), output_path + "MNAME_RNAME_timeseries_hourofday_s4.png",
                       legend=False)

        page.addFigure("Cycles mean(seasonly)", ("Taylor Graph"),
                       output_path + "MNAME_RNAME_timeseries_taylorgram_hourofday.png",
                       legend=False)

        page.addFigure("Cycles mean", ('(Cycle 1)' + "Time Series"),
                       output_path + "MNAME_RNAME_timeseries_dayofyear.png",
                       legend=False)

        page.addFigure("Cycles mean", ('(Cycle 2)' + "Time Series"),
                       output_path + "MNAME_RNAME_timeseries_monthofyear.png",
                       legend=False)

        page.addFigure("Cycles mean", ('(Cycle 3)' + "Time Series"),
                       output_path + "MNAME_RNAME_timeseries_seasonofyear.png",
                       legend=False)

        page.addFigure("Cycles mean", ('(Cycle 4)' + "Time Series"),
                       output_path + "MNAME_RNAME_timeseries_taylorgram_cycles.png",
                       legend=False)

        page.addFigure("Frequency", ('(Wavelet)' + "Wavelet Obs"), output_path + "MNAME_RNAME_wavelet.png",
                       legend=False)
        page.addFigure("Frequency", ('(Wavelet)' + "Wavelet Mod0"), output_path + "MNAME_RNAME_wavelet_Mod0.png",
                       legend=False)

        page.addFigure("Response two variables", ('(Time Series)' + "Legend2"),
                       output_path + "MNAME_RNAME_timeseries_legend.png",
                       legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response"), output_path + "MNAME_RNAME_response.png",
                       legend=False)

        page.addFigure("Response two variables", ('(Response)' + "Response_error"),
                       output_path + "MNAME_RNAME_response_error.png",
                       legend=False)

        page.addFigure("Response two variables", ('(Response)' + "Response(DJF)"),
                       output_path + "MNAME_RNAME_response_s1.png",
                       legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(DJF)1"),
                       output_path + "MNAME_RNAME_response_error_s1.png",
                       legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(MAM)"),
                       output_path + "MNAME_RNAME_response_s2.png",
                       legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(MAM)2"),
                       output_path + "MNAME_RNAME_response_error_s2.png",
                       legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(JJA)"),
                       output_path + "MNAME_RNAME_response_s3.png",
                       legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(JJA)2"),
                       output_path + "MNAME_RNAME_response_error_s3.png",
                       legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(SON)"),
                       output_path + "MNAME_RNAME_response_s4.png",
                       legend=False)

        page.addFigure("Response two variables", ('(Response)' + "Response(SON)2"),
                       output_path + "MNAME_RNAME_response_error_s4.png",
                       legend=False)

        page.addFigure("Response four variables", ('(Response)' + "Response 4 Variable"),
                       output_path + "MNAME_RNAME_response4.png",
                       legend=False)
        page.addFigure("Response four variables", ('(Response)' + "Response 4 Variable(DJF)"),
                       output_path + "MNAME_RNAME_response4_s1.png",
                       legend=False)
        page.addFigure("Response four variables", ('(Response)' + "Response 4 Variable(MAM)"),
                       output_path + "MNAME_RNAME_response4_s2.png",
                       legend=False)
        page.addFigure("Response four variables", ('(Response)' + "Response 4 Variable(JJA)"),
                       output_path + "MNAME_RNAME_response4_s3.png",
                       legend=False)
        page.addFigure("Response four variables", ('(Response)' + "Response 4 Variable(SON)"),
                       output_path + "MNAME_RNAME_response4_s4.png",
                       legend=False)

        page.addFigure("Correlations", ('(Correlations)' + "Correlations"), output_path + "MNAME_RNAME_corr4.png",
                       legend=False)
        page.addFigure("Correlations", ('(Correlations)' + "Correlations(DJF)"), output_path + "MNAME_RNAME_corr4_s1.png",
                       legend=False)
        page.addFigure("Correlations", ('(Correlations)' + "Correlations(MAM)"), output_path + "MNAME_RNAME_corr4_s2.png",
                       legend=False)
        page.addFigure("Correlations", ('(Correlations)' + "Correlations(JJA)"), output_path + "MNAME_RNAME_corr4_s3.png",
                       legend=False)
        page.addFigure("Correlations", ('(Correlations)' + "Correlations(SON)"), output_path + "MNAME_RNAME_corr4_s4.png",
                       legend=False)

        page.addFigure("Correlations", ('(Correlations)' + "Correlations_box"),
                       output_path + "MNAME_RNAME_correlation_box.png",
                       legend=False)

    def generateHtml(self):
        """Generate the HTML for the results of this confrontation.

        This routine opens all netCDF files and builds a table of
        metrics. Then it passes the results to the HTML generator and
        saves the result in the output directory. This only occurs on
        the confrontation flagged as master.

        """
        # only the master processor needs to do this
        output_path = self.output_path  # '/Users/lli51/Documents/ILAMB_sample/'
        # output_path ='/Users/lli51/Documents/ILAMB_sample/'
        import random


        for modnumber, modname in enumerate(self.IterNameModList):  #
            if modnumber < len(self.IterNameModList) - 1:
                # output_path = '/Users/lli51/Documents/ILAMB_sample/'
                results = Dataset(os.path.join(output_path, "%s_%s.nc" % (self.name, modname)), mode="w")
                results.setncatts({"name": modname, "color": m.color})
                for r, region in enumerate(self.regions):
                    Variable(name=(str(r) + ",TimeSeries"), unit="Average Error", data=self.score[region+'TimeSeries'][modnumber]).toNetCDF4(results, group="MeanState")
                    # Variable(name=(str(r) + ",CycleMeans"), unit="0-1", data=self.score[region+'CycleMeans'][modnumber]).toNetCDF4(results, group="MeanState")
                    # Variable(name=(str(r) + ",Frequency"), unit="0-1", data=self.score[region+'Frequency'][modnumber]).toNetCDF4(results, group="MeanState")
                    Variable(name=(str(r) + ",Response2"), unit="Average Error", data=self.score[region+'response2'][modnumber]).toNetCDF4(results, group="MeanState")
                    Variable(name=(str(r) + ",Response4"), unit="Average Error", data=self.score[region+'response4'][modnumber]).toNetCDF4(results, group="MeanState")
                    Variable(name=(str(r) + ",PartialCorrelation"), unit="0-1", data=np.mean(self.score[region+'PartialCorrelation'][modnumber+1])).toNetCDF4(results, group="MeanState")
                    Variable(name=(str(r) + ",CorrelationTrend"), unit="0-1", data=self.score[region+"CorrelationTrend"][modnumber+1]).toNetCDF4(results, group="MeanState")
                    Variable(name=(str(r) + ",CorrelationDetrend"), unit="0-1", data=self.score[region+"CorrelationDetrend"][modnumber+1]).toNetCDF4(results, group="MeanState")

                results.close()
            if modnumber == len(self.IterNameModList)-1:
                results = Dataset(os.path.join(output_path, "%s_%s.nc" % (self.name, modname)), mode="w")
                results.setncatts({"name": modname, "color": m.color})
                for r, region in enumerate(self.regions):
                    Variable(name=(str(r) + ",TimeSeries"), unit="Average Error", data=self.score[region+'TimeSeries'][1:]).toNetCDF4(results, group="MeanState")
                    # Variable(name=(str(r) + ",CycleMeans"), unit="0-1", data='-').toNetCDF4(results, group="MeanState")
                    # Variable(name=(str(r) + ",Frequency"), unit="0-1", data='-').toNetCDF4(results, group="MeanState")
                    Variable(name=(str(r) + ",Response2"), unit="Average Error", data=self.score[region+'Response2'][1:]).toNetCDF4(results, group="MeanState")
                    Variable(name=(str(r) + ",Response4"), unit="Average Error", data=self.score[region+'Response4'][1:]).toNetCDF4(results, group="MeanState")
                    Variable(name=(str(r) + ",PartialCorrelation"), unit="0-1", data=0).toNetCDF4(results, group="MeanState")
                    Variable(name=(str(r) + ",CorrelationTrend"), unit="0-1",
                             data=np.mean(self.score[region + "CorrelationTrend"][1:])).toNetCDF4(results,
                                                                                                    group="MeanState")
                    Variable(name=(str(r) + ",CorrelationDetrend"), unit="0-1",
                             data=np.mean(self.score[region + "CorrelationDetrend"][1:])).toNetCDF4(results,
                                                                                                      group="MeanState")

                results.close()

            if self.master:
                results = Dataset(os.path.join(output_path, "%s_Benchmark.nc" % (self.name)), mode="w")
                results.setncatts({"name": "Benchmark", "color": np.asarray([0.5, 0.5, 0.5])})
                for r, region in enumerate(self.regions):
                    Variable(name=(str(r) + ",TimeSeries"), unit="Average Error", data=0).toNetCDF4(results, group="MeanState")
                    # Variable(name=(str(r) + ",CycleMeans"), unit="0-1", data=0).toNetCDF4(results, group="MeanState")
                    # Variable(name=(str(r) + ",Frequency"), unit="0-1", data=0).toNetCDF4(results, group="MeanState")
                    Variable(name=(str(r) + ",Response2"), unit="Average Error", data=0).toNetCDF4(results, group="MeanState")
                    Variable(name=(str(r) + ",Response4"), unit="Average Error", data=0).toNetCDF4(results, group="MeanState")
                    Variable(name=(str(r) + ",PartialCorrelation"), unit="0-1", data=np.mean(self.score[region+'PartialCorrelation'][0])).toNetCDF4(results, group="MeanState")
                    Variable(name=(str(r) + ",CorrelationTrend"), unit="0-1",
                             data=self.score[region + "CorrelationTrend"][0]).toNetCDF4(results,
                                                                                                    group="MeanState")
                    Variable(name=(str(r) + ",CorrelationDetrend"), unit="0-1",
                             data=self.score[region + "CorrelationDetrend"][0]).toNetCDF4(results,
                                                                                                    group="MeanState")
                results.close()

        if not self.master: return

        for page in self.layout.pages:

            # build the metric dictionary
            metrics = {}
            page.models = []
            for fname in glob.glob(os.path.join(output_path, "*.nc")):
                with Dataset(fname) as dataset:
                    mname = dataset.getncattr("name")
                    if mname != "Benchmark": page.models.append(mname)
                    if not dataset.groups.has_key(page.name): continue
                    group = dataset.groups[page.name]

                    # if the dataset opens, we need to add the model (table row)
                    metrics[mname] = {}

                    # each model will need to have all regions
                    for region in self.regions: metrics[mname][region] = {}

                    # columns in the table will be in the scalars group
                    if not group.groups.has_key("scalars"): continue

                    # we add scalars to the model/region based on the region
                    # name being in the variable name. If no region is found,
                    # we assume it is the global region.
                    grp = group.groups["scalars"]
                    for vname in grp.variables.keys():
                        found = False
                        # print(vname)
                        vnamelist = vname.split(',')
                        for r, region in enumerate(self.regions):
                            if str(r) in vnamelist:
                                found = True
                                var = grp.variables[vname]
                                vnamelist.remove(str(r))
                                name = vnamelist[0]#vname.replace(str(r), "")
                                # print(mname, region, name)
                                metrics[mname][region][name] = Variable(name=name,
                                                                        unit=var.units,
                                                                        data=var[...])
                        if not found:
                            var = grp.variables[vname]
                            metrics[mname]["global"][vname] = Variable(name=vname,
                                                                       unit=var.units,
                                                                       data=var[...])
            # print(metrics)
            page.setMetrics(metrics)

        # write the HTML page
        f = file(os.path.join(output_path, "%s.html" % (self.name)), "w")
        f.write(str(self.layout))
        f.close()


if __name__ == "__main__":
    #s
    data_file = '/Users/lli51/Documents/ILAMB_sample/DATA/rsus/CERES/rsus_0.5x0.5.nc'
    m = ModelResult('/Users/lli51/Documents/ILAMB_sample/MODELS/', modelname='CERES')
    #
    # data_file = '/Users/lli51/Downloads/alldata/obs_FSH_model_ilamb.nc4'
    # m = ModelResult('/Users/lli51/Downloads/alldata/171206_ELMv0_CN_FSH_model_ilamb.nc4', modelname='mod1')
    # # #
    c = ConfSite(source=data_file, name='CERES', variable='rsus')
    c.confront(m)
    c.generateHtml()
