from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
from ILAMB.Variable import Variable
from ILAMB.Confrontation import getVariableList
from ILAMB.Regions import Regions
import ILAMB.ilamblib as il
import ILAMB.Post as post

from netCDF4 import Dataset
import matplotlib.pyplot as plt, numpy as np
from matplotlib.collections import LineCollection
# import ILAMB.Post1 as post
import ILAMB.Post3 as post
# import ILAMB.Post as post
from scipy.interpolate import CubicSpline
from scipy import stats, linalg
from scipy import signal

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import proj3d

import pandas as pd
import seaborn as sns; sns.set()
import os, glob, math, calendar, PIL
from PIL import Image

# self-package
from taylorDiagram import plot_Taylor_graph_time_basic
from taylorDiagram import plot_Taylor_graph_season_cycle
from taylorDiagram import plot_Taylor_graph
from taylorDiagram import plot_Taylor_graph_day_cycle
from taylorDiagram import plot_Taylor_graph_three_cycle
import waipy
from PyEMD import EEMD
from hht import hht
from hht import plot_imfs
from hht import plot_frequency




# List of colors for all different models, which can be set separately
col = ['plum', 'darkorchid', 'blue', 'navy', 'deepskyblue', 'darkcyan', 'seagreen', 'darkgreen',
       'olivedrab', 'gold', 'tan', 'red', 'palevioletred', 'm', 'plum']

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

def change_x_tick(obs, tt, site_id, ax0):
    tmask = np.where(obs.mask == False)[0]
    if tmask.size > 0:
        tmin, tmax = tmask[[0, -1]]
    else:
        tmin = 0;
        tmax = obs.time.size - 1
    t = tt[tmin:(tmax + 1)]
    ind = np.where(t % 365 < 30.)[0]
    ticks= t[ind] - (t[ind] % 365)
    nums = int(len(ticks)/6)
    if nums>=1:
        ticks = ticks[::nums]
    ticklabels = (ticks / 365. + 1850.).astype(int)
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(ticklabels)

def CycleReshape(var, cycle_length=None):  # cycle_length [days]
    if cycle_length == None:
        cycle_length = 1.
    dt = ((var.time_bnds[:, 1] - var.time_bnds[:, 0]).mean())
    spd = int(round(cycle_length / dt))

    if spd <= 0:
        print('The cycle number is smaller than the given data cycle, original data is replicated')
        repli = int(round(dt / cycle_length))
        data = np.repeat(var.data, repli, axis=0)
        cycle = data.reshape((-1, repli) + var.data.shape[1:])
        tbnd = var.time_bnds  # % cycle_length
        t = tbnd.mean(axis=1)
    else:
        if spd == 1:
            begin = 0
        else:
            begin = np.argmin(var.time[:(spd - 1)] % spd) # begin starts from 8?
        end = begin + int(var.data[begin:,0].size / float(spd)) * spd # Time
        shp = (-1, spd) + var.data.shape[1:]  # reshape (-1, spd, site)
        # print(begin, end, var.data[begin:end, :].shape, shp)
        cycle = var.data[begin:end, :].reshape(shp)  # reshape(time, cycle, site)
        # what is the meaning of tbnd? var.time_bnds[begin:end, :].shape  (time, 2)
        tbnd = var.time_bnds[begin:end, :].reshape((-1, spd, 2)) # % cycle_length
        tbnd = tbnd[:, 0, :]
        # tbnd[-1, 1] = 1.
        t = tbnd.mean(axis=1)
    return cycle, t, tbnd  # bounds on time

def GetSeasonMask(obs, mmod, site_id, season_kind):
    # print('This function is used to mask seasons!')
    # The input data is annual data
    print('Process on mask Seasonal ' + 'No.' + str(site_id) + '!')
    odata, ot, otb = CycleReshape(obs, cycle_length=365.)

    x = odata[:, :, site_id]
    begin = 0
    end = 0 + int(x[0, :].size / float(4)) * 4
    mask = np.ones_like(x)
    if season_kind == 1:
        mask[:,begin:end/4] = 0
        xx = np.ma.masked_where(mask, x)
    elif season_kind == 2:
        mask[:,end/4:2*end/4]=0
        xx = np.ma.masked_where(mask, x)
    elif season_kind == 3:
        mask[:,2*end/4:3*end/4]=0
        xx = np.ma.masked_where(mask, x)
    else:
        mask[:, 3 * end / 4:end] = 0
        xx = np.ma.masked_where(mask, x)
    obs2 = Variable(name=obs.name, unit=obs.unit, time=obs.time, data=(xx.reshape(1,-1).T))
    mmod2 = []
    for i, mod in enumerate(mmod):
        mdata, mt, mtb = CycleReshape(mod, cycle_length=365.)
        y = mdata[:, :, site_id]
        mask = np.ones_like(x)
        if season_kind == 1:
            mask[:,begin:end/4]=0
            yy = np.ma.masked_where(mask, y)
        elif season_kind == 2:
            mask[:,end/4:2*end/4]=0
            yy = np.ma.masked_where(mask, y)
        elif season_kind == 3:
            mask[:,2*end/4:3*end/4]=0
            yy = np.ma.masked_where(mask, y)
        else:
            mask[:, 3 * end / 4:end] = 0
            yy = np.ma.masked_where(mask, y)
        # newmod.append(yy.reshape(1,-1))
        mmod2.append(Variable(name=mod.name, unit=mod.unit, time=mod.time, data=(yy.reshape(1, -1).T)))
    # print("obs_data", (obs2.data[:,0]).shape)
    # print("obs_time1", (obs.data[:,0]).shape)
    return obs2, mmod2

def binPlot(X, Y, label=None, ax=None, numBins=8, xmin=None, xmax=None, c=None):

    '''  Adopted from  http://peterthomasweir.blogspot.com/2012/10/plot-binned-mean-and-mean-plusminus-std.html '''
    if xmin is None:
        xmin = X.min()
    if xmax is None:
        xmax = X.max()
    bins = np.linspace(xmin, xmax, numBins + 1)
    xx = np.array([np.mean((bins[binInd], bins[binInd + 1])) for binInd in range(numBins)])
    yy = np.array([np.mean(Y[(X > bins[binInd]) & (X <= bins[binInd + 1])]) for binInd in range(numBins)])
    yystd = 0.5*np.array([np.std(Y[(X > bins[binInd]) & (X <= bins[binInd + 1])]) for binInd in range(numBins)])
    if label == 'Observed':
        ax.plot(xx, yy, 'k-')
        ax.errorbar(xx, yy, yerr=yystd, fmt='o', elinewidth=2, capthick=1, capsize=4, color='k')
    else:
        ax.plot(xx, yy, '-', c=c)
        ax.errorbar(xx, yy, yerr=yystd, fmt='o', elinewidth=2, capthick=1, capsize=4, color=c)

def detrend_corr(C):
    # This function caculates the detrend correlation between pairs of variables in C
    # C = np.column_stack([C, np.ones(C.shape[0])])
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            mask = C[:, i].mask | C[:, j].mask
            if len(C[:, i][~mask])>0:
                res_i = signal.detrend(C[:, i][~mask])
                res_j = signal.detrend(C[:, j][~mask])
                corr = np.ma.corrcoef(res_i, res_j)[0, 1]
            else:
                corr = -2
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr

def correlation_matrix(datas):
    frame = pd.DataFrame(datas)
    A = frame.corr(method='pearson', min_periods=1)
    corr = np.ma.corrcoef(A)
    mask = np.zeros_like(A)
    mask[np.triu_indices_from(mask)] = True

    return corr, mask

def Plot_TimeSeries(obs, mmod, site_id, col_num=0, site_name = None, score = None):

    print('Process on TimeSeries ' + 'No.' + str(site_id) + '!')
    x = np.ma.masked_invalid(obs.data[:, site_id])
    t = obs.time
    xx = x.compressed()
    if score is not None:
        score['h_mean'][site_id][-1] = xx.mean()
        # score['h_anom'][site_id][-1] = xx.anom()
        # score['h_median'][site_id][-1] = xx.median()
        score['h_std'][site_id][-1] = xx.std()
        score['h_var'][site_id][-1] = xx.var()
        score['Coffcoef'][site_id][-1] = np.corrcoef(xx)

    fig0 = plt.figure(figsize=(7, 4))
    # ax0 = fig0.add_subplot(1, 1, 1)
    ax0 = plt.gca()
    plt.suptitle(site_name)
    plt.subplots_adjust(top=0.9)
    ax0.plot(t[~x.mask], xx, 'k-', label='Observed')
    ax0.set_xlabel('Year(hourly)', fontsize=20)
    ax0.set_ylabel(obs.name+'('+obs.unit+')', fontsize=20)
    change_x_tick(x, t, site_id, ax0)

    zz,z = 0,0
    mods = []
    for i, mod in enumerate(mmod):
        y = np.ma.masked_invalid(mod.data[:, site_id])
        zz+=y
        z+=y
        yy = y[~x.mask]
        mods.append(yy)
        if col_num < 0:
            ax0.plot(t[~x.mask], yy, '-', label="Model " + str(i+1), color=col[i])
        else:
            ax0.plot(t[~x.mask], yy, '-', label="Model " + str(col_num+1), color=col[col_num])

    zz /=float(len(mmod))
    z/=float(len(mmod))
    if score is not None:
        score['h_mean'][site_id][col_num+1] = zz.mean()
        # score['h_anom'][site_id][col_num+1] = zz.anom()
        # score['h_median'][site_id][col_num+1] = zz.median()
        score['h_std'][site_id][col_num+1] = zz.std()
        score['h_var'][site_id][col_num+1] = zz.var()
        score['Coffcoef'][site_id][col_num+1]=np.corrcoef(x,z)[0][1]

    # fig0, samples0 = plot_Taylor_graph(xx, mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)
    figLegend = plt.figure(figsize=(3, 3))
    plt.figlegend(*ax0.get_legend_handles_labels(), loc='center', fontsize='xx-large')

    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])
    return fig0, figLegend

def Plot_TimeSeries_cycle(obs, mmod, site_id, cycle_length, col_num=0, site_name = None, score = None):

    if 'h' in obs.unit:
        if cycle_length ==1.0:
            obs0 = Variable(name=obs.name, unit=obs.unit.replace("h", "d"), time=obs.time, data=obs.data *24)
            mm = []
            for mod in mmod:
                mm.append(Variable(name=mod.name, unit=mod.unit.replace("h", "d"), time=mod.time, data=mod.data *24))
        elif cycle_length ==30.0:
            obs0 = Variable(name=obs.name, unit=obs.unit.replace("h", "m"), time=obs.time, data=obs.data *24*30)
            mm = []
            for mod in mmod:
                mm.append(Variable(name=mod.name, unit=mod.unit.replace("h", "m"), time=mod.time, data=mod.data *24*30))
        elif cycle_length ==365.0:
            obs0 = Variable(name=obs.name, unit=obs.unit.replace("h", "y"), time=obs.time, data=obs.data *24*365)
            mm = []
            for mod in mmod:
                mm.append(Variable(name=mod.name, unit=mod.unit.replace("h", "y"), time=mod.time, data=mod.data *24*365))
        elif cycle_length ==90.0:
            obs0 = Variable(name=obs.name, unit=obs.unit.replace("h", "season"), time=obs.time, data=obs.data *24*90)
            mm = []
            for mod in mmod:
                mm.append(Variable(name=mod.name, unit=mod.unit.replace("h", "season"), time=mod.time, data=mod.data *24*90))
    else:
        obs0 = obs
        mm = mmod

    print('Process on Cycle Means ' + 'No.' + str(site_id) + '!')
    odata, ot, otb = CycleReshape(obs0, cycle_length=cycle_length)

    x = np.ma.masked_invalid(odata[:, :, site_id])
    t = ot
    fig0 = plt.figure(figsize=(7, 4))
    xx = np.mean(x, axis=1)
    ax0 = fig0.add_subplot(111)
    ax0.plot(t, xx, 'k', label="Obs")
    plt.suptitle(site_name)
    # change_x_tick(xx, t, site_id, ax0)
    mods = []
    for i, mod in enumerate(mm):
        mdata, mt, mtb = CycleReshape(mod, cycle_length=cycle_length)
        y = mdata[:, :, site_id]
        yy = np.mean(y, axis=1)
        if col_num<0:
            ax0.plot(t, yy, label=" Model " + str(i + 1), color=col[i])
        else:
            ax0.plot(t, yy, label=" Model " + str(i + 1), color=col[col_num])
        mods.append(yy[~xx.mask])

    if int(cycle_length) == 1:
        ax0.set_xlabel('Year(daily)', fontsize=20)
    elif int(cycle_length) == 30:
        ax0.set_xlabel('Year(monthly)', fontsize=20)
    elif int(cycle_length) == 365:
        ax0.set_xlabel('Year(annualy)', fontsize=20)
    elif int(cycle_length) == 90:
        ax0.set_xlabel('Year(seasonly)', fontsize=20)
    else:
        ax0.set_xlabel('Year', fontsize=20)
    ax0.set_ylabel(obs.name + '(' + obs0.unit + ')', fontsize=20)

    # fix x-ticks
    nums = int(len(t)/5)
    ticks = t[::nums] - (t[::nums] % 365)
    ticklabels = (ticks / 365. + 1850.).astype(int)
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(ticklabels)
    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])
    return fig0

def Plot_TimeSeries_cycle_season(obs, mmod, site_id, cycle_length, col_num=0, s=1, site_name = None, score = None):
    if 'h' in obs.unit:
        if cycle_length ==1.0:
            obs0 = Variable(name=obs.name, unit=obs.unit.replace("h", "d"), time=obs.time, data=obs.data *24)
            mm = []
            for mod in mmod:
                mm.append(Variable(name=mod.name, unit=mod.unit.replace("h", "d"), time=mod.time, data=mod.data *24))
        elif cycle_length ==30.0:
            obs0 = Variable(name=obs.name, unit=obs.unit.replace("h", "m"), time=obs.time, data=obs.data *24*30)
            mm = []
            for mod in mmod:
                mm.append(Variable(name=mod.name, unit=mod.unit.replace("h", "m"), time=mod.time, data=mod.data *24*30))
        elif cycle_length ==365.0:
            obs0 = Variable(name=obs.name, unit=obs.unit.replace("h", "y"), time=obs.time, data=obs.data *24*365)
            mm = []
            for mod in mmod:
                mm.append(Variable(name=mod.name, unit=mod.unit.replace("h", "y"), time=mod.time, data=mod.data *24*365))
        elif cycle_length ==90.0:
            obs0 = Variable(name=obs.name, unit=obs.unit.replace("h", "season"), time=obs.time, data=obs.data *24*90)
            mm = []
            for mod in mmod:
                mm.append(Variable(name=mod.name, unit=mod.unit.replace("h", "season"), time=mod.time, data=mod.data *24*90))
    else:
        obs0 = obs
        mm = mmod

    print('Process on Cycle Means ' + 'No.' + str(site_id) + '!')
    odata, ot, otb = CycleReshape(obs0, cycle_length=cycle_length)
    x = np.ma.masked_invalid(odata[:, :, site_id])
    t = ot
    fig0 = plt.figure(figsize=(7, 4))
    plt.suptitle(site_name)
    xx = np.mean(x, axis=1)
    ax0 = fig0.add_subplot(111)
    ax0.plot(t, xx, 'k', label="Obs")
    # change_x_tick(xx, t, site_id, ax0)
    mods = []
    for i, mod in enumerate(mm):
        mdata, mt, mtb = CycleReshape(mod, cycle_length=cycle_length)
        y = mdata[:, :, site_id]
        yy = np.mean(y, axis=1)
        if col_num <= 0:
            ax0.plot(t, yy, label=" Model " + str(i + 1), color=col[i])
        else:
            ax0.plot(t, yy, label=" Model " + str(i + 1), color=col[col_num])
        mods.append(yy[~xx.mask])
    seasonlist = ['DJF', 'MAM', 'JJA', 'SON']
    ax0.set_xlabel('Year(annual,'+seasonlist[s-1]+')', fontsize=24)
    ax0.set_ylabel(obs.name+'('+obs.unit+')', fontsize=24)
    # fix x-ticks
    nums = int(len(t)/5)
    ticks = t[::nums] - (t[::nums] % 365)
    ticklabels = (ticks / 365. + 1850.).astype(int)
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(ticklabels)
    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])
    return fig0

def Plot_TimeSeries_cycle_reshape(obs, mmod, site_id, cycle_length,xname="Hours of a day", col_num=0, site_name = None, score = None):

    print('Process on Cycle Means Reshape ' + 'No.' + str(site_id) + '!')
    odata, ot, otb = CycleReshape(obs, cycle_length=cycle_length)
    x = odata[:, :, site_id]
    t = ot

    fig0 = plt.figure(figsize=(7, 4))
    plt.suptitle(site_name)
    ax0 = fig0.add_subplot(1, 1, 1)

    # x = np.ma.masked_where(mask1, x)
    ax0.plot(range(len(x.mean(axis=0))), x.mean(axis=0), 'k', label='Obs')
    ax0.fill_between(range(len(x.mean(axis=0))), x.mean(axis=0) - x.std(axis=0),
                     x.mean(axis=0) + x.std(axis=0), alpha=0.2, edgecolor='#1B2ACC',
                     linewidth=0.5, linestyle='dashdot', antialiased=True)

    xx = x.mean(axis=0)
    mods = []
    for i, mod in enumerate(mmod):
        mdata, mt, mtb = CycleReshape(mod, cycle_length=cycle_length)
        y = mdata[:, :, site_id]
        # mask1 = x.mask | y.mask
        # y = np.ma.masked_where(mask1, y)
        yy = y.mean(axis=0)

        mods.append(yy)
        if col_num<0:
            ax0.plot(range(len(y.mean(axis=0))), y.mean(axis=0), label='Mod ' + str(i + 1), color=col[i])
            ax0.fill_between(range(len(y.mean(axis=0))), y.mean(axis=0) - y.std(axis=0),
                             y.mean(axis=0) + y.std(axis=0), alpha=0.2, edgecolor='#1B2ACC',
                             linewidth=0.5, linestyle='dashdot', antialiased=True)

        else:

            ax0.plot(range(len(y.mean(axis=0))), y.mean(axis=0), label='Mod '+str(i+1), color=col[col_num])
            ax0.fill_between(range(len(y.mean(axis=0))), y.mean(axis=0) - y.std(axis=0),
                             y.mean(axis=0) + y.std(axis=0), alpha=0.2, edgecolor='#1B2ACC',
                             linewidth=0.5, linestyle='dashdot', antialiased=True)
    ax0.set_xlabel(xname, fontsize=20)
    ax0.set_ylabel(obs.name+'('+obs.unit+')', fontsize=20)


    # ax0.legend(bbox_to_anchor=(1.3, 0.5), shadow=False)

    if xname == 'Months of a year':
        plt.xticks(range(len(x.mean(axis=0))), calendar.month_name[1:13], rotation=20)
    elif xname == 'Seasons of a year':
        plt.xticks(range(len(x.mean(axis=0))), ('DJF', 'MAM', 'JJA', 'SON'))
    else:
        nums = int(len(x.mean(axis=0)) / 6)
        ticks = np.arange(len(x.mean(axis=0)))

        if nums >= 1:
            ticks = ticks[::nums]
        ticklabels = ticks.astype(int)
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(ticklabels)

    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])
    # fig0, samples0 = plot_Taylor_graph(xx, mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)

    return fig0

def Plot_TimeSeries_cycle_reshape_season(obs, mmod, site_id, cycle_length, xname="Hours of a day", col_num=0,s=1, site_name = None, score = None):

    print('Process on Cycle Means Reshape ' + 'No.' + str(site_id) + '!')
    odata, ot, otb = CycleReshape(obs, cycle_length=cycle_length)
    x = odata[:, :, site_id]
    t = ot

    fig0 = plt.figure(figsize=(7, 4))
    plt.suptitle(site_name)
    ax0 = fig0.add_subplot(1, 1, 1)

    # x = np.ma.masked_where(mask1, x)
    ax0.plot(range(len(x.mean(axis=0))), x.mean(axis=0), 'k', label='Obs')
    ax0.fill_between(range(len(x.mean(axis=0))), x.mean(axis=0) - x.std(axis=0),
                     x.mean(axis=0) + x.std(axis=0), alpha=0.2, edgecolor='#1B2ACC',
                     linewidth=0.5, linestyle='dashdot', antialiased=True)

    for i, mod in enumerate(mmod):
        mdata, mt, mtb = CycleReshape(mod, cycle_length=cycle_length)
        y = mdata[:, :, site_id]
        if col_num <=0:
            ax0.plot(range(len(y.mean(axis=0))), y.mean(axis=0), label='Mod ' + str(i + 1), color=col[i])
            ax0.fill_between(range(len(y.mean(axis=0))), y.mean(axis=0) - y.std(axis=0),
                             y.mean(axis=0) + y.std(axis=0), alpha=0.2, edgecolor='#1B2ACC',
                             linewidth=0.5, linestyle='dashdot', antialiased=True)
        else:
            ax0.plot(range(len(y.mean(axis=0))), y.mean(axis=0), label='Mod '+str(i+1), color=col[col_num])
            ax0.fill_between(range(len(y.mean(axis=0))), y.mean(axis=0) - y.std(axis=0),
                             y.mean(axis=0) + y.std(axis=0), alpha=0.2, edgecolor='#1B2ACC',
                             linewidth=0.5, linestyle='dashdot', antialiased=True)
    seasonlist = ['DJF','MAM','JJA','SON']
    ax0.set_xlabel(xname+'('+seasonlist[s-1]+')', fontsize=20)
    ax0.set_ylabel(obs.name+'('+obs.unit+')', fontsize=20)

    if xname == 'Months of a year':
        plt.xticks(range(len(x.mean(axis=0))), calendar.month_name[1:13], rotation=20)
    elif xname == 'Seasons of a year':
        plt.xticks(range(len(x.mean(axis=0))), ('DJF', 'MAM', 'JJA', 'SON'))
    else:
        nums = int(len(x.mean(axis=0)) / 6)
        ticks = np.arange(len(x.mean(axis=0)))

        if nums >= 1:
            ticks = ticks[::nums]
        ticklabels = ticks.astype(int)
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(ticklabels)

    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])

    return fig0

def Plot_TimeSeries_TaylorGram(obs, mmod, site_id, col_num=0, site_name = None, score = None):
    print('Process on TimeSeries Taylor Grams ' + 'No.' + str(site_id) + '!')
    odata1, ot, otb = CycleReshape(obs, cycle_length=1)
    x1 = np.ma.masked_invalid(odata1[:, :, site_id])
    data1 = np.mean(x1, axis=1)
    odata2, ot, otb = CycleReshape(obs, cycle_length=30)
    x2 = np.ma.masked_invalid(odata2[:, :, site_id])
    data2 = np.mean(x2, axis=1)
    odata3, ot, otb = CycleReshape(obs, cycle_length=90)
    x3 = np.ma.masked_invalid(odata3[:, :, site_id])
    data3 = np.mean(x3, axis=1)

    fig0 = plt.figure(figsize=(7, 7))
    # change_x_tick(xx, t, site_id, ax0)
    models1,models2,models3 = [],[],[]
    for i, mod in enumerate(mmod):
        mdata1, mt, mtb = CycleReshape(mod, cycle_length=1)
        y1 = mdata1[:, :, site_id]
        yy1 = np.mean(y1, axis=1)
        models1.append(yy1)
        mdata2, mt, mtb = CycleReshape(mod, cycle_length=30)
        y2 = mdata2[:, :, site_id]
        yy2 = np.mean(y2, axis=1)
        models2.append(yy2)
        mdata3, mt, mtb = CycleReshape(mod, cycle_length=90)
        y3 = mdata3[:, :, site_id]
        yy3 = np.mean(y3, axis=1)
        models3.append(yy3)
    if col_num <0:
        fig0, samples1, samples2, samples3 = plot_Taylor_graph_time_basic(data1, data2, data3, models1, models2,
                                                                          models3, fig0, rect=111, ref_times=10,
                                                                          bbox_to_anchor=(0.9,0.9))
    else:
        fig0, samples1, samples2, samples3 = plot_Taylor_graph_time_basic(data1, data2, data3, models1, models2, models3, fig0, rect=111, ref_times=10, bbox_to_anchor=(0.9, 0.9), modnumber=col_num+1)
    return fig0

def Plot_TimeSeries_TaylorGram_annual(obs, mmod, site_id, col_num=0, site_name = None, score = None):
    print('Process on TimeSeries Taylor Grams ' + 'No.' + str(site_id) + '!')
    obs1, mmod1 = GetSeasonMask(obs, mmod, site_id, 1)
    odata1, ot, otb = CycleReshape(obs1, cycle_length=365)
    x1 = np.ma.masked_invalid(odata1[:, :, 0])
    data1 = np.mean(x1, axis=1)
    obs2, mmod2 = GetSeasonMask(obs, mmod, site_id, 2)
    odata2, ot, otb = CycleReshape(obs2, cycle_length=365)
    x2 = np.ma.masked_invalid(odata2[:, :, 0])
    data2 = np.mean(x2, axis=1)
    obs3, mmod3 = GetSeasonMask(obs, mmod, site_id, 3)
    odata3, ot, otb = CycleReshape(obs3, cycle_length=365)
    x3 = np.ma.masked_invalid(odata3[:, :, 0])
    data3 = np.mean(x3, axis=1)
    obs4, mmod4 = GetSeasonMask(obs, mmod, site_id, 4)
    odata4, ot, otb = CycleReshape(obs4, cycle_length=365)
    x4 = np.ma.masked_invalid(odata4[:, :, 0])
    data4 = np.mean(x4, axis=1)

    odata0, ot, otb = CycleReshape(obs, cycle_length=365)
    x0 = np.ma.masked_invalid(odata0[:, :, site_id])
    data0 = np.mean(x0, axis=1)

    fig0 = plt.figure(figsize=(7, 7))
    # change_x_tick(xx, t, site_id, ax0)
    models1,models2,models3,models4,models5 = [],[],[],[],[]
    for i in range(len(mmod1)):
        mdata1, mt, mtb = CycleReshape(mmod1[i], cycle_length=365)
        y1 = mdata1[:, :, 0]
        yy1 = np.mean(y1, axis=1)
        models1.append(yy1)
        mdata2, mt, mtb = CycleReshape(mmod2[i], cycle_length=365)
        y2 = mdata2[:, :, 0]
        yy2 = np.mean(y2, axis=1)
        models2.append(yy2)
        mdata3, mt, mtb = CycleReshape(mmod3[i], cycle_length=365)
        y3 = mdata3[:, :, 0]
        yy3 = np.mean(y3, axis=1)
        models3.append(yy3)
        mdata4, mt, mtb = CycleReshape(mmod4[i], cycle_length=365)
        y4 = mdata4[:, :, 0]
        yy4 = np.mean(y4, axis=1)
        models4.append(yy4)

        mdata0, mt, mtb = CycleReshape(mmod[i], cycle_length=365)
        y0 = mdata0[:, :, site_id]
        yy0 = np.mean(y0, axis=1)
        models5.append(yy0)

    if col_num<=0:
        fig0, samples1, samples2, samples3, samples4, samples5 = plot_Taylor_graph_season_cycle(data1, data2, data3, data4,
                                                                                            data0, models1, models2,
                                                                                            models3, models4, models5,
                                                                                            fig0, rect=111,
                                                                                            ref_times=10,
                                                                                            bbox_to_anchor=(0.93, 0.9))
    else:
        fig0, samples1, samples2, samples3, samples4, samples5 = plot_Taylor_graph_season_cycle(data1, data2, data3,
                                                                                                data4,
                                                                                                data0, models1, models2,
                                                                                                models3, models4,
                                                                                                models5,
                                                                                                fig0, rect=111,
                                                                                                ref_times=10,
                                                                                                bbox_to_anchor=(
                                                                                                0.93,0.9),
                                                                                                modnumber=col_num + 1)

    return fig0

def Plot_TimeSeries_TaylorGram_hourofday(obs, mmod, site_id, col_num=0, site_name = None, score = None):
    print('Process on TimeSeries Taylor Grams ' + 'No.' + str(site_id) + '!')
    obs1, mmod1 = GetSeasonMask(obs, mmod, site_id, 1)
    odata1, ot, otb = CycleReshape(obs1, cycle_length=1)
    x1 = np.ma.masked_invalid(odata1[:, :, 0])
    data1 = np.mean(x1, axis=0)
    obs2, mmod2 = GetSeasonMask(obs, mmod, site_id, 2)
    odata2, ot, otb = CycleReshape(obs2, cycle_length=1)
    x2 = np.ma.masked_invalid(odata2[:, :, 0])
    data2 = np.mean(x2, axis=0)
    obs3, mmod3 = GetSeasonMask(obs, mmod, site_id, 3)
    odata3, ot, otb = CycleReshape(obs3, cycle_length=1)
    x3 = np.ma.masked_invalid(odata3[:, :, 0])
    data3 = np.mean(x3, axis=0)
    obs4, mmod4 = GetSeasonMask(obs, mmod, site_id, 4)
    odata4, ot, otb = CycleReshape(obs4, cycle_length=1)
    x4 = np.ma.masked_invalid(odata4[:, :, 0])
    data4 = np.mean(x4, axis=0)

    odata0, ot, otb = CycleReshape(obs, cycle_length=1)
    x0 = np.ma.masked_invalid(odata0[:, :, site_id])
    data0 = np.mean(x0, axis=0)


    fig0 = plt.figure(figsize=(7, 7))
    # change_x_tick(xx, t, site_id, ax0)
    models1,models2,models3,models4,models5 = [],[],[],[],[]
    for i in range(len(mmod1)):
        mdata1, mt, mtb = CycleReshape(mmod1[i], cycle_length=1)
        y1 = mdata1[:, :, 0]
        yy1 = np.mean(y1, axis=0)
        models1.append(yy1)
        mdata2, mt, mtb = CycleReshape(mmod2[i], cycle_length=1)
        y2 = mdata2[:, :, 0]
        yy2 = np.mean(y2, axis=0)
        models2.append(yy2)
        mdata3, mt, mtb = CycleReshape(mmod3[i], cycle_length=1)
        y3 = mdata3[:, :, 0]
        yy3 = np.mean(y3, axis=0)
        models3.append(yy3)
        mdata4, mt, mtb = CycleReshape(mmod4[i], cycle_length=1)
        y4 = mdata4[:, :, 0]
        yy4 = np.mean(y4, axis=0)
        models4.append(yy4)

        mdata0, mt, mtb = CycleReshape(mmod[i], cycle_length=1)
        y0 = mdata0[:, :, site_id]
        yy0 = np.mean(y0, axis=0)
        models5.append(yy0)

    if col_num <= 0:

        fig0, samples1, samples2, samples3, samples4, samples5 = plot_Taylor_graph_day_cycle(data1, data2, data3, data4,
                                                                                            data0, models1, models2,
                                                                                            models3, models4, models5,
                                                                                            fig0, rect=111,
                                                                                            ref_times=10,
                                                                                            bbox_to_anchor=(0.93, 0.9))
    else:
        fig0, samples1, samples2, samples3, samples4, samples5 = plot_Taylor_graph_day_cycle(data1, data2, data3, data4,
                                                                                             data0, models1, models2,
                                                                                             models3, models4, models5,
                                                                                             fig0, rect=111,
                                                                                             ref_times=10,
                                                                                             bbox_to_anchor=(
                                                                                                 0.93, 0.9),
                                                                                             modnumber=col_num + 1)
    return fig0

def Plot_TimeSeries_TaylorGram_cycles(obs, mmod, site_id, col_num=0, site_name = None, score = None):
    print('Process on TimeSeries Taylor Grams ' + 'No.' + str(site_id) + '!')
    odata01, ot, otb = CycleReshape(obs, cycle_length=1.)
    obs1_1 = Variable(name=obs.name,unit=obs.unit, time=ot,data=np.mean(odata01, axis=1))
    odata1, ot, otb = CycleReshape(obs1_1, cycle_length=365.)
    x1 = np.ma.masked_invalid(odata1[:, :, site_id])
    data1 = np.mean(x1, axis=0)

    odata02, ot, otb = CycleReshape(obs, cycle_length=30.)
    obs1_2 = Variable(name=obs.name, unit=obs.unit, time=ot, data=np.mean(odata02, axis=1))
    odata2, ot, otb = CycleReshape(obs1_2, cycle_length=365.)
    x2 = np.ma.masked_invalid(odata2[:, :, site_id])
    data2 = np.mean(x2, axis=0)

    odata03, ot, otb = CycleReshape(obs, cycle_length=90.)
    obs1_3 = Variable(name=obs.name, unit=obs.unit, time=ot, data=np.mean(odata03, axis=1))
    odata3, ot, otb = CycleReshape(obs1_3, cycle_length=365.)
    x3 = np.ma.masked_invalid(odata3[:, :, site_id])
    data3 = np.mean(x3, axis=0)

    fig0 = plt.figure(figsize=(7, 7))
    # change_x_tick(xx, t, site_id, ax0)
    models1,models2,models3 = [],[],[]
    for i, mod in enumerate(mmod):
        mdata01, mt, mtb = CycleReshape(mod, cycle_length=1.)
        mod1_1 = Variable(name=mod.name, unit=mod.unit, time=mt, data=np.mean(mdata01, axis=1))
        mdata1, ot, otb = CycleReshape(mod1_1, cycle_length=365.)
        y1 = mdata1[:, :, site_id]
        yy1 = np.mean(y1, axis=0)
        models1.append(yy1)

        mdata02, mt, mtb = CycleReshape(mod, cycle_length=30.)
        mod1_2 = Variable(name=mod.name, unit=mod.unit, time=mt, data=np.mean(mdata02, axis=1))
        mdata2, ot, otb = CycleReshape(mod1_2, cycle_length=365.)
        y2 = mdata2[:, :, site_id]
        yy2 = np.mean(y2, axis=0)
        models2.append(yy2)

        mdata03, mt, mtb = CycleReshape(mod, cycle_length=90.)
        mod1_3 = Variable(name=mod.name, unit=mod.unit, time=mt, data=np.mean(mdata03, axis=1))
        mdata3, ot, otb = CycleReshape(mod1_3, cycle_length=365.)
        y3 = mdata3[:, :, site_id]
        yy3 = np.mean(y3, axis=0)
        models3.append(yy3)

    if col_num<=0:
        fig0, samples1, samples2, samples3 = plot_Taylor_graph_three_cycle(data1, data2, data3, models1, models2,
                                                                           models3, fig0, rect=111, ref_times=10,
                                                                           bbox_to_anchor=(0.9, 0.9))

    else:

        fig0, samples1, samples2, samples3 = plot_Taylor_graph_three_cycle(data1, data2, data3, models1, models2, models3, fig0, rect=111, ref_times=10, bbox_to_anchor=(0.9, 0.9),modnumber=col_num+1)

    return fig0

def Plot_PDF_CDF(obs,mmod,site_id, col_num=0, site_name = None, score = None):
    print('Process on PDF&CDF ' + 'No.' + str(site_id) + '!')
    x = np.ma.masked_invalid(obs.data[:, site_id])
    t = obs.time

    fig0 = plt.figure(figsize=(7, 12))
    plt.suptitle(site_name)
    ax0 = fig0.add_subplot(2, 1, 1)
    ax1 = fig0.add_subplot(2, 1, 2)

    h_obs_sorted = np.ma.sort(x).compressed()
    p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
    p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)))  # bin it into n = N/10 bins
    x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
    ax0.plot(x_h, p_h / float(sum(p_h)), 'k-', label='Observed')
    ax1.plot(h_obs_sorted, p1_data, 'k-', label='Observed')

    for i, mod in enumerate(mmod):
        y = np.ma.masked_invalid(mod.data[:, site_id])
        h_obs_sorted = np.ma.sort(y).compressed()
        p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
        p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)))  # bin it into n = N/10 bins
        x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
        if col_num <0:
            ax0.plot(x_h, p_h / float(sum(p_h)), label="Model " + str(i + 1), color=col[i])
            ax1.plot(h_obs_sorted, p1_data, label="Model " + str(i + 1), color=col[i])
        else:
            ax0.plot(x_h, p_h / float(sum(p_h)), label="Model "+str(i+1), color=col[col_num])
            ax1.plot(h_obs_sorted, p1_data, label="Model "+str(i+1), color=col[col_num])

    ax0.set_xlabel('Hourly '+obs.name+'('+obs.unit+')',fontsize=20)
    ax1.set_xlabel('Hourly '+obs.name+'('+obs.unit+')',fontsize=20)
    ax0.set_ylabel('Probability Distribution(PDF)',fontsize=20)
    ax1.set_ylabel('Cumulative Distribution(CDF)',fontsize=20)
    # ax0.legend(bbox_to_anchor=(1.3, 0), shadow=False)
    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])

    return fig0

def Plot_PDF_CDF_one_mod_seasonal(obs,mmod,site_id, col_num=0, site_name = None, score = None):
    print('Process on PDF&CDF ' + 'No.' + str(site_id) + '!')
    x = np.ma.masked_invalid(obs.data[:, site_id])
    t = obs.time

    fig0 = plt.figure(figsize=(7, 7))
    # plt.suptitle('PDF and CDF')
    ax0 = fig0.add_subplot(2, 1, 1)
    ax1 = fig0.add_subplot(2, 1, 2)

    h_obs_sorted = np.ma.sort(x).compressed()
    p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
    p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)))  # bin it into n = N/10 bins
    x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
    ax0.plot(x_h, p_h / float(sum(p_h)), 'k-', label='Observed')
    ax1.plot(h_obs_sorted, p1_data, 'k-', label='Observed')

    for i, mod in enumerate(mmod):
        y = np.ma.masked_invalid(mod.data[:, site_id])
        h_obs_sorted = np.ma.sort(y).compressed()
        p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
        p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)))  # bin it into n = N/10 bins
        x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
        ax0.plot(x_h, p_h / float(sum(p_h)), label="Model "+str(i+1), color=col[col_num])
        ax1.plot(h_obs_sorted, p1_data, label="Model "+str(i+1), color=col[col_num])

    # ax0.legend(bbox_to_anchor=(1.3, 0), shadow=False)
    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])

    return fig0

def Plot_Wavelet(obs, obst, site_id, unit, model_name='Obs', col_num=0, site_name = None, score = None):
    print('Process on Wavelet ' + 'No.' + str(site_id) + '!')
    data = np.ma.masked_invalid(obs.data[:, site_id])
    time_data = obst[~data.mask]
    fig3 = plt.figure(figsize=(7, 6))
    result = waipy.cwt(data.compressed(), 1, 1, 0.125, 2, 4 / 0.125, 0.72, 6, mother='Morlet', name= model_name)
    ax1, ax2, ax3, ax5 = waipy.wavelet_plot(model_name, time_data, data.compressed(), 0.03125, result, fig3, unit=unit)
    change_x_tick(data, obst, site_id, ax2)

    if score is not None:
        globalpower = sorted(result['global_ws'])
        ind1 = np.where(result['global_ws'] == globalpower[-1])
        ind2 = np.where(result['global_ws'] == globalpower[-2])
        # print(model_name, result['period'][ind1],result['period'][ind2])
        if model_name == 'obs':
            score['FrequencyMax'][site_id][-1]=result['global_ws'][ind1]
            score['FrequencyTMax'][site_id][-1] =result['period'][ind1]
        else:
            score['FrequencyMax'][site_id][col_num] =result['global_ws'][ind1]
            score['FrequencyTMax'][site_id][col_num] =result['period'][ind1]
    return fig3

def Plot_IMF_one_mod(obs, ot, mmod, mt, site_id, col_num=0, site_name = None, score = None):
    print('Process on Decomposer_IMF_' + str(site_id) + '!')
    data0 = np.ma.masked_invalid(obs.data[:, site_id])
    data0 = np.ma.masked_where(data0==0.00, data0)

    # time = ot[~data.mask]
    d_mod = []
    for i, mod in enumerate(mmod):
        y = np.ma.masked_invalid(mod.data[:, site_id])
        time_y = mt[~y.mask]
        d_mod.append(y)

    fig0 = plt.figure(figsize=(5, 3))
    fig3 = plt.figure(figsize=(3, 3))
    data = np.ma.masked_invalid(obs.data[:, site_id])
    data = np.ma.masked_where(data==0.00, data)

    time = ot[~data.mask]
    data = data.compressed()
    eemd = EEMD(trials=5)
    # print(len(data), len(data0))
    if len(data) > 0:
        imfs = eemd.eemd(data)
        # print('obs',imfs.shape)
        if len(imfs) >= 1:
            ax0 = fig0.add_subplot(1, 2, 1)
            ax0.plot(time, (imfs[len(imfs) - 1]), 'k-', label='Observed')

            # d_d_obs = np.asarray([str(1850 + int(x) / 365) + (
            #     '0' + str(int(x) % 365 / 31 + 1) if int(x) % 365 / 31 < 9 else str(int(x) % 365 / 31 + 1)) for x
            #                       in
            #                       time])
            d_d_obs = np.asarray([str(1850 + 1 + int(x) / 365) for x
                                  in
                                  time])
            ax0.xaxis.set_ticks(
                [time[0], time[2 * len(time) / 5],
                 time[4 * len(time) / 5]])
            ax0.set_xticklabels(
                [d_d_obs[0], d_d_obs[2 * len(d_d_obs) / 5],
                 d_d_obs[4 * len(d_d_obs) / 5]])

        ## hht spectrum
        if len(imfs) >= 1:
            fig3, freq = hht(data, imfs, time, 1, fig3, inityear=1850)


    fig1 = plt.figure(figsize=(2 * (len(d_mod) + 1), 3))
    fig2 = plt.figure(figsize=(2 * (len(d_mod) + 1), 3))
    fig1.subplots_adjust(wspace=0.5, hspace=0.3)
    fig2.subplots_adjust(wspace=0.5, hspace=0.3)

    if len(data) > 0:
        if len(imfs) >= 1:
            fig1 = plot_imfs(data, imfs, time_samples=time, fig=fig1, no=1, m=len(d_mod),inityear=1850)
            fig2 = plot_frequency(data, freq.T, time_samples=time, fig=fig2, no=1, m=len(d_mod),inityear=1850)

        models1 = []
        datamask = []
        data1 = imfs[len(imfs) - 1]
        for m in range(len(d_mod)):
            ## hht spectrum
            eemd = EEMD(trials=5)
            fig4 = plt.figure(figsize=(7, 7))
            data2 = d_mod[m][~data0.mask]
            imfs = eemd.eemd(data2.compressed())
            # print('mod'+str(m), imfs.shape)
            if len(imfs) >= 1:
                fig4, freq = hht(data2.compressed(), imfs, time[~data2.mask], 1, fig4,inityear=1850)

            if len(imfs) >= 1:
                fig1 = plot_imfs(data2.compressed(), imfs, time_samples=time[~data2.mask], fig=fig1, no=m + 2,
                                 m=len(d_mod), inityear=1850)
                fig2 = plot_frequency(data2.compressed(), freq.T, time_samples=time[~data2.mask], fig=fig2, no=m + 2,
                                      m=len(d_mod),inityear=1850)
                ax0.plot(time[~data2.mask], (imfs[len(imfs) - 1]), '-', label='Model' + str(m + 1), c=col[col_num])
                models1.append(imfs[len(imfs) - 1])
                datamask.append(data2)

        ax0.set_xlabel('Time')
        # ax0.set_ylabel('(' + obs.unit + ')')
        ax0.yaxis.tick_right()
        ax0.yaxis.set_label_position("right")
        ax0.legend(bbox_to_anchor=(-0.05, 1), shadow=False, fontsize='medium')
        plot_Taylor_graph(data1, models1, fig0, 122, datamask=datamask)
    else:
        print("'Data's length is too short !")


    fig0.subplots_adjust(left=0.1, hspace=0.25, wspace=0.55)

    return fig0, fig1, fig2, fig3


def Plot_response2(obs1, mod1, obs2, mod2, site_id, col_num=0, site_name = None, s=None, score = None):
    print('Process on Response ' + 'No.' + str(site_id) + '!')
    x1 = np.ma.masked_invalid(obs1.data[:, site_id])
    x2 = np.ma.masked_invalid(obs2.data[:, site_id])

    fig0 = plt.figure(figsize=(7, 7))
    # plt.suptitle('Response 2 variables')
    ax0 = fig0.add_subplot(1, 1, 1)
    ax0.plot(x1, x2, 'k.', label='Obs')

    if col_num <=0:
        for i, mod11 in enumerate(mod1):
            y1 = np.ma.masked_invalid(mod1[i].data[:, site_id])
            y2 = np.ma.masked_invalid(mod2[i].data[:, site_id])

            ax0.plot(y1, y2,'.', label='mod'+str(i+1), color=col[i])
    else:
        for i, mod11 in enumerate(mod1):
            y1 = np.ma.masked_invalid(mod1[i].data[:, site_id])
            y2 = np.ma.masked_invalid(mod2[i].data[:, site_id])

            ax0.plot(y1, y2,'.', label='mod'+str(i+1), color=col[col_num])

    ax0.set_xlabel(obs1.name+'(' +obs1.unit+')', fontsize=20)
    ax0.set_ylabel(obs2.name+'(' +obs2.unit+')', fontsize=20)
    if s !=None:
        seasonlist = ['DJF', 'MAM', 'JJA', 'SON']
        plt.suptitle(site_name +' ('+seasonlist[s-1]+')', y=0.91)
    else:

        plt.suptitle(site_name, y=0.91)
    return fig0

def Plot_response2_error(obs1, mod1, obs2, mod2, site_id, col_num=0, site_name = None, s=None, score = None):
    print('Process on Response ' + 'No.' + str(site_id) + '!')
    x1= np.ma.masked_invalid(obs1.data[:, site_id])
    x2 = np.ma.masked_invalid(obs2.data[:, site_id])

    fig0 = plt.figure(figsize=(7, 7))
    # plt.suptitle('Response 2 variables')
    ax0 = fig0.add_subplot(1, 1, 1)
    binPlot(x1, x2, label='Observed', ax=ax0, numBins=15)

    if col_num <=0:
        for i, mod11 in enumerate(mod1):
            y1 = np.ma.masked_invalid(mod1[i].data[:, site_id])
            y2 = np.ma.masked_invalid(mod2[i].data[:, site_id])
            binPlot(y1, y2, ax=ax0, numBins=8,c=col[i])
    else:
        for i, mod11 in enumerate(mod1):
            y1 = np.ma.masked_invalid(mod1[i].data[:, site_id])
            y2 = np.ma.masked_invalid(mod2[i].data[:, site_id])
            binPlot(y1, y2, ax=ax0, numBins=8,c=col[col_num])

    ax0.set_xlabel(obs1.name + '(' + obs1.unit + ')', fontsize=20)
    ax0.set_ylabel(obs2.name + '(' + obs2.unit + ')', fontsize=20)
    if s != None:
        seasonlist = ['DJF', 'MAM', 'JJA', 'SON']
        plt.suptitle(site_name + ' (' + seasonlist[s - 1] + ')', y=0.91)
    else:

        plt.suptitle(site_name, y=0.91)

    return fig0

def Plot_response4(obs1, mod1, obs2, mod2, obs3, mod3, obs4, mod4, site_id, col_num=0, site_name=None, s=None, score = None):
    print('Process on Response4 ' + 'No.' + str(site_id) + '!')
    x1 = np.ma.masked_invalid(obs1.data[:, site_id])
    x1 = np.ma.masked_where(x1 == 0, x1)
    x2 = np.ma.masked_invalid(obs2.data[:, site_id])
    x2 = np.ma.masked_where(x2 == 0, x2)
    x3 = np.ma.masked_invalid(obs3.data[:, site_id])
    x3 = np.ma.masked_where(x3 == 0, x3)
    x4 = np.ma.masked_invalid(obs4.data[:, site_id])
    x4 = np.ma.masked_where(x4 == 0, x4)

    mask1 = x1.mask | x2.mask | x3.mask | x4.mask

    fig0 = plt.figure(figsize=(5, 4 * (len(mod1) + 1)))
    if s != None:
        seasonlist = ['DJF', 'MAM', 'JJA', 'SON']
        plt.suptitle(site_name + ' (' + seasonlist[s - 1] + ')', y=1.)
    else:
        plt.suptitle(site_name, y=1.)

    ax0 = fig0.add_subplot(len(mod1) + 1, 1, 1, projection='3d')

    colorax = ax0.scatter(x1[~mask1], x2[~mask1], x3[~mask1], c=x4[~mask1], cmap=plt.hot())

    ax0.set_title('Obs:' + obs4.name + '(' + obs4.unit + ')', fontsize=10)
    ax0.set_xlabel(obs1.name + '\n(' + obs1.unit + ')', fontsize=10)
    ax0.set_ylabel(obs2.name + '\n(' + obs2.unit + ')', fontsize=10)
    ax0.set_zlabel(obs3.name + '\n(' + obs3.unit + ')', fontsize=10)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_zticks([])

    #     fig0.subplots_adjust(right=0.7)
    z = x4[~mask1]
    axes = fig0.add_axes([1.1, 0.25, 0.02, 0.5])
    #     axes = [ax0]
    cbar = fig0.colorbar(colorax, ticks=[min(z), max(z)], orientation='vertical',
                         label=obs4.name, cax=axes)
    cbar.ax.set_yticklabels(['Low', 'High'])  # horizontal colorbar

    for i in range(len(mod1)):
        y1 = np.ma.masked_invalid(mod1[i].data[:, site_id])
        y1 = np.ma.masked_where(y1 == 0, y1)
        y2 = np.ma.masked_invalid(mod2[i].data[:, site_id])
        y2 = np.ma.masked_where(y2 == 0, y2)
        y3 = np.ma.masked_invalid(mod3[i].data[:, site_id])
        y3 = np.ma.masked_where(y3 == 0, y3)
        y4 = np.ma.masked_invalid(mod4[i].data[:, site_id])
        y4 = np.ma.masked_where(y4 == 0, y4)
        mask2 = y1.mask | y2.mask | y3.mask | y4.mask

        ax0 = fig0.add_subplot(len(mod1) + 1, 1, i + 2, projection='3d')

        colorax = ax0.scatter(y1[~mask2], y2[~mask2], y3[~mask2], c=y4[~mask2], cmap=plt.hot())
        if col_num < 0:
            ax0.set_title('Mod' + str(i + 1) + ':' + obs4.name + '(' + obs4.unit + ')', fontsize=10)
        else:
            ax0.set_title('Mod' + str(i + 1) + ':' + obs4.name + '(' + obs4.unit + ')', fontsize=10)
        ax0.set_xlabel(obs1.name + '\n(' + obs1.unit + ')', fontsize=10)
        ax0.set_ylabel(obs2.name + '\n(' + obs2.unit + ')', fontsize=10)
        ax0.set_zlabel(obs3.name + '\n(' + obs3.unit + ')', fontsize=10)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_zticks([])

    plt.tight_layout()
    return fig0

def correlation_matrix(datas):
    frame = pd.DataFrame(datas)
    A = frame.corr(method='pearson', min_periods=1)
    corr = np.ma.corrcoef(A)
    mask = np.zeros_like(A)
    mask[np.triu_indices_from(mask)] = True

    return corr, mask

def partial_corr(C):
#     """
    #     Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    #     for the remaining variables in C.
    #     Parameters
    #     ----------
    #     C : array-like, shape (n, p)
    #         Array with the different variables. Each column of C is taken as a variable
    #     Returns
    #     -------
    #     P : array-like, shape (p, p)
    #         P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
    #         for the remaining variables in C.

#     """
    # """
    # Partial Correlation in Python (clone of Matlab's partialcorr)
    # This uses the linear regression approach to compute the partial
    # correlation (might be slow for a huge number of variables).The code is adopted from
    # https://gist.github.com/fabianp/9396204419c7b638d38f
    # Date: Nov 2014
    # Author: Fabian Pedregosa-Izquierdo, f@bianp.net
    # Testing: Valentina Borghesani, valentinaborghesani@gmail.com
    # """

    C = np.column_stack([C, np.ones(C.shape[0])])

    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
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

def plot_4_variable_corr(obs1, mod1_1, obs2, mod2_1, obs3, mod3_1, obs4, mod4_1, site_id, col_num=0, site_name = None, score = None):
    print('Process on Corr4 ' + 'No.' + str(site_id) + '!')
    x1 = np.ma.masked_invalid(obs1.data[:, site_id])
    x1 = np.ma.masked_where(x1 == 0, x1)
    x2 = np.ma.masked_invalid(obs2.data[:, site_id])
    x2 = np.ma.masked_where(x2 == 0, x2)
    x3 = np.ma.masked_invalid(obs3.data[:, site_id])
    x3 = np.ma.masked_where(x3 == 0, x3)
    x4 = np.ma.masked_invalid(obs4.data[:, site_id])
    x4 = np.ma.masked_where(x4 == 0, x4)

    mask1 = x1.mask | x2.mask | x3.mask | x4.mask
    v0 = np.asarray([x1[~mask1],x2[~mask1],x3[~mask1],x4[~mask1]]).T
    corr0 = partial_corr(v0)
    # print('corr0:',corr0)
    fig0 = plt.figure(figsize=(5, 5))
    # plt.suptitle('Correlations 4 variables')
    ax0 = fig0.add_subplot(1, 1, 1, projection='3d')
    ax0.scatter(corr0[0, 1], corr0[0, 2], corr0[0, 3], label='Obs', marker='^',s=60)

    if col_num <=0:
        for i, mod1 in enumerate(mod1_1):
            y1 = np.ma.masked_invalid(mod1_1[i].data[:, site_id])
            y1 = np.ma.masked_where(y1 == 0, y1)
            y2 = np.ma.masked_invalid(mod2_1[i].data[:, site_id])
            y2 = np.ma.masked_where(y2 == 0, y2)
            y3 = np.ma.masked_invalid(mod3_1[i].data[:, site_id])
            y3 = np.ma.masked_where(y3 == 0, y3)
            y4 = np.ma.masked_invalid(mod4_1[i].data[:, site_id])
            y4 = np.ma.masked_where(y4 == 0, y4)
            mask2 = y1.mask | y2.mask | y3.mask | y4.mask
            v0 = np.asarray([y1[~mask2], y2[~mask2], y3[~mask2], y4[~mask2]]).T
            corr0 = partial_corr(v0)
            ax0.scatter(corr0[0, 1], corr0[0, 2], corr0[0, 3], label='Mod'+str(i+1), marker='^',s=60)
    else:
        for i, mod1 in enumerate(mod1_1):
            y1 = np.ma.masked_invalid(mod1_1[i].data[:, site_id])
            y1 = np.ma.masked_where(y1 == 0, y1)
            y2 = np.ma.masked_invalid(mod2_1[i].data[:, site_id])
            y2 = np.ma.masked_where(y2 == 0, y2)
            y3 = np.ma.masked_invalid(mod3_1[i].data[:, site_id])
            y3 = np.ma.masked_where(y3 == 0, y3)
            y4 = np.ma.masked_invalid(mod4_1[i].data[:, site_id])
            y4 = np.ma.masked_where(y4 == 0, y4)
            mask2 = y1.mask | y2.mask | y3.mask | y4.mask
            v0 = np.asarray([y1[~mask2], y2[~mask2], y3[~mask2], y4[~mask2]]).T
            corr0 = partial_corr(v0)
            ax0.scatter(corr0[0, 1], corr0[0, 2], corr0[0, 3], label='Mod' + str(col_num), marker='^', s=60)

    ax0.dist = 12
    ax0.tick_params(labelsize=4)
    ax0.legend(bbox_to_anchor=(1.03, 1), loc=2, fontsize=18)
    ax0.set_xlabel(obs1.name + '(' + obs1.unit + ')', fontsize=14)
    ax0.set_ylabel(obs2.name + '(' + obs2.unit + ')', fontsize=14)
    ax0.set_zlabel(obs3.name + '(' + obs3.unit + ')', fontsize=14)
    plt.suptitle(site_name, y=.92)
    ax0.xaxis.labelpad = 5
    ax0.yaxis.labelpad = 5
    ax0.zaxis.labelpad = 5
    return fig0

def plot_variable_matrix_trend_and_detrend(data, dtrend_data, variable_list, col_num=0, site_name = None, score = None):
    fig, axes = plt.subplots(len(variable_list), len(variable_list), sharex=True, sharey=True,
                             figsize=(6, 6))
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    plt.suptitle(site_name, y=.92)
    plt.figtext(0.99, 0.01, '* stands for the detrended data', horizontalalignment='right')
    ax_cb = fig.add_axes([.91, .3, .03, .4])

    for j in range(len(variable_list)):  # models
        for k in range(j, len(variable_list)):
            array = data[:, k, j]  ## Put n sites in one picture
            nam = ['Obs('+str(array[0])[:4]+')']
            if col_num<=0:
                nam.extend(['Mod'+str(i+1)+'('+str(array[i+1])[:4]+')' for i in range(len(array)-1)])
            else:
                nam.extend(['Mod'+str(col_num)+'('+str(array[i+1])[:4]+')' for i in range(len(array)-1)])
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
            nam = ['Obs*('+str(array[0])[:4]+')']
            if col_num<=0:
                nam.extend(['Mod'+str(i+1)+'*('+str(array[i+1])[:4]+')' for i in range(len(array)-1)])
            else:
                nam.extend(['Mod'+str(col_num)+'*('+str(array[i+1])[:4]+')' for i in range(len(array)-1)])
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

class ConfSite(Confrontation):
    """A confrontation for examining the site
    """

    def __init__(self, **keywords):

        # Calls the regular constructor
        super(ConfSite, self).__init__(**keywords)
        # Setup a html layout for generating web views of the results

        obs = Variable(filename=self.source,
                       variable_name=self.variable,
                       alternate_vars=self.alternate_vars)

        # Set the number of sites being considered and the number of models
        self.sitenumber = 3  # len(regions)
        # self.mmname = ["Model1", "Model2", "Models"]
        self.mmname = ["Models", "Model1", "Model2"]
        self.new_metrics = ['h_mean','h_std','h_var','FrequencyMax','FrequencyTMax','Coffcoef']
        # self.nam_metrics = ['Mean','Anom','Median','Std','Varance','Frequency','Correlation']
        self.uni_metrics = {'h_mean':obs.unit,'h_anom': obs.unit, 'h_median': obs.unit, 'h_var': obs.unit+'^2','FrequencyMax': 'Months','FrequencyTMax': 'Months','Coffcoef': '0-1', 'h_std': '0-1'}

        self.score = {}
        for metric in self.new_metrics:
            self.score[metric] = np.full([self.sitenumber, len(self.mmname)+1], np.nan)

        self.lats = obs.lat[:self.sitenumber]
        self.lons = obs.lon[:self.sitenumber]

        r = Regions()
        ###################
        # Setup the name of the sites and regions functions
        regions = []
        for i in range(len(self.lats)):
            # Define the site name in the webpage
            # r.addRegionLatLonBounds(("lat" + str(self.lats[i])[:5] + "lon" + str(self.lons[i])[:5]),
            #                         "lat" + str(self.lats[i])[:5] + "lon" + str(self.lons[i])[:5],
            #                         (self.lats[i] - 0.5, self.lats[i] + 0.5), (self.lons[i] - 0.5, self.lons[i] + 0.5))
            # regions.append("lat" + str(self.lats[i])[:5] + "lon" + str(self.lons[i])[:5])
            r.addRegionLatLonBounds(str(i), str(i), (self.lats[i] - 0.01, self.lats[i] + 0.01), (self.lons[i] - 0.01, self.lons[i] + 0.01))
            regions.append(str(i))


        # Define the organized regions, added for future districts

        self.lowerlatbound = []
        self.upperlatbound = []
        self.lowerlonbound = []
        self.upperlonbound = []

        r.addRegionLatLonBounds("global", "Globe", (-90, 90), (-180, 180))
        regions.append("global")
        self.lowerlatbound.append(-90)
        self.upperlatbound.append(80)
        self.lowerlonbound.append(-180)
        self.upperlonbound.append(180)
        #######################


        self.regions = regions


        pages = []

        pages.append(post.HtmlPage("MeanState", "Mean State"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections(["Time series","Time series(Annually)", "Cycles mean", "Cycles mean(seasonly)", "PDF CDF", "Frequency", "Response two variables", "Response four variables", "Correlations"])
        pages[-1].setRegions(self.regions) #self.regions
        # pages.append(post.HtmlAllModelsPage("AllModels","All Models"))
        # pages[-1].setHeader("CNAME / RNAME")
        # pages[-1].setSections(["Time series", "Cycles mean", "Frequency", "Response"])
        # pages[-1].setSections([])
        # pages[-1].setRegions(self.regions)
        pages.append(post.HtmlPage("DataInformation","Data Information"))
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
        mod = m.extractTimeSeries(self.variable,
                                  alt_vars=self.alternate_vars)
        # mod = mod.integrateInSpace().convert(obs.unit)

        obs, mod = il.MakeComparable(obs, mod, clip_ref=True)

        # capture the several sites for short test
        obs1 = Variable(name=obs.name,unit=obs.unit,time=obs.time,data=obs.data[:,0, 0:self.sitenumber])
        mod1 = Variable(name=mod.name, unit=mod.unit, time=mod.time, data=mod.data[:, 0, 0:self.sitenumber])

        return obs1, mod1

    def confront(self, m):

        # output_path = self.output_path
        output_path = '/Users/lli51/Documents/ILAMB_sample/test_output/' #local address
        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]

        # Grab the data, replicate the mods..
        obs, mod = self.stageData(m)

        obs1,obs2, obs3,obs4 = obs,obs,obs,obs
        mod1,mod2, mod3,mod4 = mod,mod,mod,mod

        mod11,mod12,mod13,mod14 = mod,mod,mod,mod
        mod21,mod22,mod23,mod24 = mod,mod,mod,mod
        mod31,mod32,mod33,mod34 = mod,mod,mod,mod
        mod41,mod42,mod43,mod44 = mod,mod,mod,mod

        mod1_1_1 = [mod11, mod12]
        mod2_1_1 = [mod21, mod22]
        mod3_1_1 = [mod31, mod32]
        mod4_1_1 = [mod41, mod42]


        # change the unit to hourly data, might need to be address for future
        if 's' in obs1.unit:
            obs1 = Variable(name=obs1.name, unit=obs1.unit.replace("s", "h"), time=obs1.time, data=obs1.data * 3600)
            mm = []
            for mod in mod1_1_1:
                mm.append(Variable(name=mod.name, unit=mod.unit.replace("s", "h"), time=mod.time, data=mod.data * 3600))
            mod1_1_1 = mm
        if 's' in obs2.unit:
            obs2 = Variable(name=obs2.name, unit=obs2.unit.replace("s", "h"), time=obs2.time, data=obs2.data * 3600)
            mm = []
            for mod in mod2_1_1:
                mm.append(Variable(name=mod.name, unit=mod.unit.replace("s", "h"), time=mod.time, data=mod.data * 3600))
            mod2_1_1 = mm
        if 's' in obs3.unit:
            obs3 = Variable(name=obs3.name, unit=obs3.unit.replace("s", "h"), time=obs3.time, data=obs3.data * 3600)
            mm = []
            for mod in mod3_1_1:
                mm.append(Variable(name=mod.name, unit=mod.unit.replace("s", "h"), time=mod.time, data=mod.data * 3600))
            mod3_1_1 = mm
        if 's' in obs4.unit:
            obs4 = Variable(name=obs4.name, unit=obs4.unit.replace("s", "h"), time=obs4.time, data=obs4.data * 3600)
            mm = []
            for mod in mod4_1_1:
                mm.append(Variable(name=mod.name, unit=mod.unit.replace("s", "h"), time=mod.time, data=mod.data * 3600))
            mod4_1_1 = mm

        variable_list = [obs1.name, obs2.name, obs3.name, obs4.name]

        for modnumber, mname in enumerate(self.mmname):
            if mname == "Models":
                mod1_1 = mod1_1_1
                mod2_1 = mod2_1_1
                mod3_1 = mod3_1_1
                mod4_1 = mod4_1_1
            else:
                mod1_1 = [mod1_1_1[modnumber - 1]]
                mod2_1 = [mod2_1_1[modnumber - 1]]
                mod3_1 = [mod3_1_1[modnumber - 1]]
                mod4_1 = [mod4_1_1[modnumber - 1]]

            ## all correlations' initals
            h_mod1 = [obs1] + mod1_1
            h_mod2 = [obs2] + mod2_1
            h_mod3 = [obs3] + mod3_1
            h_mod4 = [obs4] + mod4_1

            h1, h2, h3, h4 = [], [], [], []
            for i in range(len(h_mod1)):
                h1.append(np.ma.masked_where(obs1.data.mask, h_mod1[i].data))
                h2.append(np.ma.masked_where(obs2.data.mask, h_mod2[i].data))
                h3.append(np.ma.masked_where(obs3.data.mask, h_mod3[i].data))
                h4.append(np.ma.masked_where(obs4.data.mask, h_mod4[i].data))
            h = [h1, h2, h3, h4]
            h_data = np.ma.masked_where(np.asarray(h) >= 1.0e+18, np.asarray(h))
            ## finish correlations' initals

            for siteid in range(len(obs.data[0])):

                region = str(siteid)#"lat" + str(lats[siteid])[:5] + "lon" + str(lons[siteid])[:5]

                ######################### Time series of one model and models with legend and Taylor graph

                fig_TimeSeries, figLegend = Plot_TimeSeries(obs1, mod1_1, siteid, col_num=modnumber-1, site_name=region, score = self.score)
                fig_TimeSeries.savefig(os.path.join(output_path, "%s_%s_timeseries_hourly.png" % (mname, region)), bbox_inches='tight')
                figLegend.savefig(os.path.join(output_path, "%s_%s_timeseries_legend.png" % (mname, region)), bbox_inches='tight')

                fig_TimeSeries_daily = Plot_TimeSeries_cycle(obs1, mod1_1, siteid, 1., col_num=modnumber-1, site_name=region, score = self.score)
                fig_TimeSeries_daily.savefig(os.path.join(output_path, "%s_%s_timeseries_daily.png" % (mname, region)), bbox_inches='tight')
                #
                fig_TimeSeries_monthly = Plot_TimeSeries_cycle(obs1, mod1_1, siteid, 30., col_num=modnumber-1, site_name=region, score = self.score)
                fig_TimeSeries_monthly.savefig(os.path.join(output_path, "%s_%s_timeseries_monthly.png" % (mname, region)),bbox_inches='tight')
                #
                fig_TimeSeries_yearly = Plot_TimeSeries_cycle(obs1, mod1_1, siteid, 365., col_num=modnumber-1, site_name=region, score = self.score)
                fig_TimeSeries_yearly.savefig(os.path.join(output_path, "%s_%s_timeseries_yearly.png" % (mname, region)),bbox_inches='tight')
                #
        #         #
                fig_TimeSeries_seasonly = Plot_TimeSeries_cycle(obs1, mod1_1, siteid, 90., col_num=modnumber-1, site_name=region, score = self.score)
                fig_TimeSeries_seasonly.savefig(os.path.join(output_path, "%s_%s_timeseries_seasonly.png" % (mname, region)),bbox_inches='tight')

        #
                plt.close("all")
        #         ############################### Cycle means output
                fig_TimeSeries_cycle_hourofday = Plot_TimeSeries_cycle_reshape(obs1, mod1_1, siteid, 1., col_num=modnumber-1,site_name=region)
                fig_TimeSeries_cycle_hourofday.savefig(os.path.join(output_path, "%s_%s_timeseries_hourofday.png" % (mname, region)),bbox_inches='tight')

                odata, ot, otb = CycleReshape(obs1, cycle_length=1.)
                obs1_2 = Variable(name=obs1.name,unit=obs1.unit,time=ot,data=np.mean(odata, axis=1))
                mod1_2 = []
                for i, mod in enumerate(mod1_1):
                    mdata, mt, mtb = CycleReshape(mod, cycle_length=1.)
                    mod1_2.append(Variable(name=obs1.name,unit=obs1.unit,time=ot,data=np.mean(mdata, axis=1)))
                fig_TimeSeries_cycle_dayofyear = Plot_TimeSeries_cycle_reshape(obs1_2, mod1_2, siteid, 365., xname="Days of a year", col_num=modnumber-1,site_name=region)
                fig_TimeSeries_cycle_dayofyear.savefig(os.path.join(output_path, "%s_%s_timeseries_dayofyear.png" % (mname, region)),bbox_inches='tight')

                odata, ot, otb = CycleReshape(obs1, cycle_length=30.)
                obs1_2 = Variable(name=obs1.name,unit=obs1.unit,time=ot,data=np.mean(odata, axis=1))
                mod1_2 = []
                for i, mod in enumerate(mod1_1):
                    mdata, mt, mtb = CycleReshape(mod, cycle_length=30.)
                    mod1_2.append(Variable(name=obs1.name,unit=obs1.unit,time=ot,data=np.mean(mdata, axis=1)))
                fig_TimeSeries_cycle_monthofyear = Plot_TimeSeries_cycle_reshape(obs1_2, mod1_2, siteid, 365., xname="Months of a year", col_num=modnumber-1,site_name=region)
                fig_TimeSeries_cycle_monthofyear.savefig(os.path.join(output_path, "%s_%s_timeseries_monthofyear.png" % (mname, region)),bbox_inches='tight')

                odata, ot, otb = CycleReshape(obs1, cycle_length=90.)
                obs1_2 = Variable(name=obs1.name,unit=obs1.unit,time=ot,data=np.mean(odata, axis=1))
                mod1_2 = []
                for i, mod in enumerate(mod1_1):
                    mdata, mt, mtb = CycleReshape(mod, cycle_length=90.)
                    mod1_2.append(Variable(name=obs1.name,unit=obs1.unit,time=ot,data=np.mean(mdata, axis=1)))
                fig_TimeSeries_cycle_seasonofyear = Plot_TimeSeries_cycle_reshape(obs1_2, mod1_2, siteid, 365., xname="Seasons of a year", col_num=modnumber-1,site_name=region)
                fig_TimeSeries_cycle_seasonofyear.savefig(os.path.join(output_path, "%s_%s_timeseries_seasonofyear.png" % (mname, region)),bbox_inches='tight')
                plt.close("all")
                ############################### PDF and CDF output
                fig_PDF_CDF = Plot_PDF_CDF(obs1, mod1_1, siteid, col_num=modnumber-1,site_name=region, score = self.score)
                fig_PDF_CDF.savefig(os.path.join(output_path, "%s_%s_PDFCDF.png" % (mname, region)),bbox_inches='tight')
                plt.close("all")

                ############################# Wavelet obs
                if modnumber <= 0:
                    odata, ot, otb = CycleReshape(obs1, cycle_length=30.) # when it goes to 30 days with hourly data, there are totally 720 samples [292, 720, 34]
                    fig_Wavelet = Plot_Wavelet(np.mean(odata, axis=1), ot, siteid, obs1.unit, model_name='Obs:'+region, col_num=modnumber-1,site_name=region, score =self.score)
                    for mmmmname in self.mmname:
                        fig_Wavelet.savefig(os.path.join(output_path, "%s_%s_wavelet.png" % (mmmmname, region)),bbox_inches='tight')
                    list_im = []
                    for i, mod in enumerate(mod1_1):
                        mdata, mt, mtb = CycleReshape(mod, cycle_length=30.)
                        fig_Wavelet_mod = Plot_Wavelet(np.mean(mdata, axis=1), mt, siteid, obs1.unit,
                                                       model_name='Mod' + str(i + 1)+':'+region, col_num=i+1, site_name=region, score = self.score)
                        fig_Wavelet_mod.savefig(
                            os.path.join(output_path, ("%s_%s_wavelet_Mod0.png") % (self.mmname[i+1], region)),bbox_inches='tight')
                        list_im.append((output_path + "%s_%s_wavelet_Mod0.png") % (self.mmname[i+1], region))

                    imgs = [PIL.Image.open(i) for i in list_im]
                    x_axis_pictures_number = 1
                    while len(imgs) % x_axis_pictures_number != 0:
                        im = Image.new('RGB', (0, 0), color=tuple((np.random.rand(3) * 255).astype(np.uint8)))
                        imgs.append(im)
                    imgs_comb = pil_grid(imgs, x_axis_pictures_number)
                    # imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
                    imgs_comb.save((output_path + "/%s_%s_wavelet_Mod0.png") % (self.mmname[0], region))
                plt.close("all")
                # ############################## Response obs
                # fig2_variable = Plot_response2(obs1, mod1_1, obs2, mod2_1, siteid, col_num=modnumber-1,site_name=region, score = self.score)
                # fig2_variable.savefig(os.path.join(output_path, "%s_%s_response.png" % (mname, region)),bbox_inches='tight')
                #
                # fig2_variable_error = Plot_response2_error(obs1, mod1_1, obs2, mod2_1, siteid, col_num=modnumber-1, site_name=region)
                # fig2_variable_error.savefig(os.path.join(output_path, "%s_%s_response_error.png" % (mname, region)), bbox_inches='tight')
                #
                # fig4_variable = Plot_response4(obs1, mod1_1, obs2, mod2_1, obs3, mod3_1, obs4, mod4_1, siteid,
                #                                col_num=modnumber - 1, site_name=region, score = self.score)
                # fig4_variable.savefig(os.path.join(output_path, "%s_%s_response4.png" % (mname, region)),
                #                       bbox_inches='tight')
                #
                # fig_corr = plot_4_variable_corr(obs1, mod1_1, obs2, mod2_1, obs3, mod3_1, obs4, mod4_1, siteid, col_num=modnumber-1,site_name=region, score = self.score)
                # fig_corr.savefig(os.path.join(output_path, "%s_%s_corr4.png" % (mname, region)),bbox_inches='tight')
                # plt.close("all")
                #
                # ############################## timeseries_hourofday
                # obs2_s1, mmod2_s1 = GetSeasonMask(obs1, mod1_1, siteid, 1)
                # figannual_s1 = Plot_TimeSeries_cycle_season(obs2_s1, mmod2_s1, 0, 365., col_num=modnumber-1,s=1,site_name=region, score = self.score) # only one output is given
                # figannual_s1.savefig(os.path.join(output_path, "%s_%s_timeseries_s1.png" % (mname, region)), bbox_inches='tight')
                # fighourofday_s1 = Plot_TimeSeries_cycle_reshape_season(obs2_s1, mmod2_s1, 0, 1., col_num=modnumber-1, s=1,site_name=region, score = self.score)
                # fighourofday_s1.savefig(os.path.join(output_path, "%s_%s_timeseries_hourofday_s1.png" % (mname, region)),bbox_inches='tight')
                #
                # obs2_s2, mmod2_s2 = GetSeasonMask(obs1, mod1_1, siteid, 2)
                # figannual_s2 = Plot_TimeSeries_cycle_season(obs2_s2, mmod2_s2, 0, 365., col_num=modnumber-1,s=2,site_name=region, score = self.score) # only one output is given
                # figannual_s2.savefig(os.path.join(output_path, "%s_%s_timeseries_s2.png" % (mname, region)), bbox_inches='tight')
                # fighourofday_s2 = Plot_TimeSeries_cycle_reshape_season(obs2_s2, mmod2_s2, 0, 1., col_num=modnumber-1, s=2,site_name=region, score = self.score)
                # fighourofday_s2.savefig(os.path.join(output_path, "%s_%s_timeseries_hourofday_s2.png" % (mname, region)),bbox_inches='tight')
                #
                # obs2_s3, mmod2_s3 = GetSeasonMask(obs1, mod1_1, siteid, 3)
                # figannual_s3 = Plot_TimeSeries_cycle_season(obs2_s3, mmod2_s3, 0, 365., col_num=modnumber-1,s=3,site_name=region, score = self.score) # only one output is given
                # figannual_s3.savefig(os.path.join(output_path, "%s_%s_timeseries_s3.png" % (mname, region)), bbox_inches='tight')
                # fighourofday_s3 = Plot_TimeSeries_cycle_reshape_season(obs2_s3, mmod2_s3, 0, 1., col_num=modnumber-1, s=3,site_name=region, score = self.score)
                # fighourofday_s3.savefig(os.path.join(output_path, "%s_%s_timeseries_hourofday_s3.png" % (mname, region)),bbox_inches='tight')
                #
                # obs2_s4, mmod2_s4 = GetSeasonMask(obs1, mod1_1, siteid, 4)
                # figannual_s4 = Plot_TimeSeries_cycle_season(obs2_s4, mmod2_s4, 0, 365., col_num=modnumber-1,s=4,site_name=region, score = self.score) # only one output is given
                # figannual_s4.savefig(os.path.join(output_path, "%s_%s_timeseries_s4.png" % (mname, region)), bbox_inches='tight')
                # fighourofday_s4 = Plot_TimeSeries_cycle_reshape_season(obs2_s4, mmod2_s4, 0, 1., col_num=modnumber-1, s=4,site_name=region, score = self.score)
                # fighourofday_s4.savefig(os.path.join(output_path, "%s_%s_timeseries_hourofday_s4.png" % (mname, region)),bbox_inches='tight')
                # plt.close("all")
                #
                # ############################### Taylor graphs
                #
                # fig_TimeSeries_TaylorGram = Plot_TimeSeries_TaylorGram(obs1, mod1_1, siteid, col_num=modnumber-1,site_name=region, score = self.score)
                # fig_TimeSeries_TaylorGram.savefig(os.path.join(output_path, "%s_%s_timeseries_taylorgram.png" % (mname, region)))
                #
                # figannual_taylorgram = Plot_TimeSeries_TaylorGram_annual(obs1, mod1_1, siteid, col_num=modnumber-1,site_name=region, score = self.score)
                # figannual_taylorgram.savefig(os.path.join(output_path, "%s_%s_timeseries_taylorgram_annual.png" % (mname, region)))
                #
                # fighourofda_taylorgram = Plot_TimeSeries_TaylorGram_hourofday(obs1, mod1_1, siteid, col_num=modnumber-1,site_name=region, score = self.score)
                # fighourofda_taylorgram.savefig(os.path.join(output_path, "%s_%s_timeseries_taylorgram_hourofday.png" % (mname, region)))
                #
                # figcycles_taylorgram =  Plot_TimeSeries_TaylorGram_cycles(obs1, mod1_1, siteid, col_num=modnumber-1,site_name=region, score = self.score)
                # figcycles_taylorgram.savefig(os.path.join(output_path, "%s_%s_timeseries_taylorgram_cycles.png" % (mname, region)))
                #
                # ################################# response seasonal
                # obs2_1, mmod2_1 = GetSeasonMask(obs1, mod1_1, siteid, 1)
                # obs2_2, mmod2_2 = GetSeasonMask(obs2, mod2_1, siteid, 1)
                # obs2_3, mmod2_3 = GetSeasonMask(obs3, mod3_1, siteid, 1)
                # obs2_4, mmod2_4 = GetSeasonMask(obs4, mod4_1, siteid, 1)
                # fig10_s = Plot_response4(obs2_1, mmod2_1, obs2_2, mmod2_2, obs2_3, mmod2_3, obs2_4, mmod2_4, 0,
                #                          col_num=modnumber - 1, site_name=region, s=1)
                # fig10_s.savefig(os.path.join(output_path, "%s_%s_response4_s1.png" % (mname, region)),
                #                 bbox_inches='tight')
                #
                # fig9_s = Plot_response2(obs2_1, mmod2_1, obs2_2, mmod2_2, 0, col_num=modnumber-1,site_name=region, s=1, score = self.score)
                # fig9_s.savefig(os.path.join(output_path, "%s_%s_response_s1.png" % (mname, region)),bbox_inches='tight')
                # fig2_variable_error_s1 = Plot_response2_error(obs2_1, mmod2_1, obs2_2, mmod2_2, 0, col_num=modnumber-1, site_name=region, s=1)
                # fig2_variable_error_s1.savefig(os.path.join(output_path, "%s_%s_response_error_s1.png" % (mname, region)), bbox_inches='tight')
                #
                #
                # obs2_1, mmod2_1 = GetSeasonMask(obs1, mod1_1, siteid, 2)
                # obs2_2, mmod2_2 = GetSeasonMask(obs2, mod2_1, siteid, 2)
                # obs2_3, mmod2_3 = GetSeasonMask(obs3, mod3_1, siteid, 2)
                # obs2_4, mmod2_4 = GetSeasonMask(obs4, mod4_1, siteid, 2)
                # fig10_s = Plot_response4(obs2_1, mmod2_1, obs2_2, mmod2_2, obs2_3, mmod2_3, obs2_4, mmod2_4, 0,
                #                          col_num=modnumber - 1, site_name=region, s=2)
                # fig10_s.savefig(os.path.join(output_path, "%s_%s_response4_s2.png" % (mname, region)),
                #                 bbox_inches='tight')
                #
                # fig9_s = Plot_response2(obs2_1, mmod2_1, obs2_2, mmod2_2, 0, col_num=modnumber-1,site_name=region, s=2, score = self.score)
                # fig9_s.savefig(os.path.join(output_path, "%s_%s_response_s2.png" % (mname, region)),bbox_inches='tight')
                # fig2_variable_error_s2 = Plot_response2_error(obs2_1, mmod2_1, obs2_2, mmod2_2, 0, col_num=modnumber-1, site_name=region, s=2)
                # fig2_variable_error_s2.savefig(os.path.join(output_path, "%s_%s_response_error_s2.png" % (mname, region)), bbox_inches='tight')
                #
                #
                # obs2_1, mmod2_1 = GetSeasonMask(obs1, mod1_1, siteid, 3)
                # obs2_2, mmod2_2 = GetSeasonMask(obs2, mod2_1, siteid, 3)
                # obs2_3, mmod2_3 = GetSeasonMask(obs3, mod3_1, siteid, 3)
                # obs2_4, mmod2_4 = GetSeasonMask(obs4, mod4_1, siteid, 3)
                # fig10_s = Plot_response4(obs2_1, mmod2_1, obs2_2, mmod2_2, obs2_3, mmod2_3, obs2_4, mmod2_4, 0,
                #                          col_num=modnumber - 1, site_name=region, s=3)
                # fig10_s.savefig(os.path.join(output_path, "%s_%s_response4_s3.png" % (mname, region)),
                #                 bbox_inches='tight')
                #
                # fig9_s = Plot_response2(obs2_1, mmod2_1, obs2_2, mmod2_2, 0, col_num=modnumber-1,site_name=region, s=3, score = self.score)
                # fig9_s.savefig(os.path.join(output_path, "%s_%s_response_s3.png" % (mname, region)),bbox_inches='tight')
                # fig2_variable_error_s3 = Plot_response2_error(obs2_1, mmod2_1, obs2_2, mmod2_2, 0, col_num=modnumber-1, site_name=region, s=3)
                # fig2_variable_error_s3.savefig(os.path.join(output_path, "%s_%s_response_error_s3.png" % (mname, region)), bbox_inches='tight')
                #
                #
                # obs2_1, mmod2_1 = GetSeasonMask(obs1, mod1_1, siteid, 4)
                # obs2_2, mmod2_2 = GetSeasonMask(obs2, mod2_1, siteid, 4)
                # obs2_3, mmod2_3 = GetSeasonMask(obs3, mod3_1, siteid, 4)
                # obs2_4, mmod2_4 = GetSeasonMask(obs4, mod4_1, siteid, 4)
                # fig10_s = Plot_response4(obs2_1, mmod2_1, obs2_2, mmod2_2, obs2_3, mmod2_3, obs2_4, mmod2_4, 0,
                #                          col_num=modnumber - 1, site_name=region, s=4, score = self.score)
                # fig10_s.savefig(os.path.join(output_path, "%s_%s_response4_s4.png" % (mname, region)),
                #                 bbox_inches='tight')
                # #
                # fig9_s = Plot_response2(obs2_1, mmod2_1, obs2_2, mmod2_2, 0, col_num=modnumber-1,site_name=region, s=4, score = self.score)
                # fig9_s.savefig(os.path.join(output_path, "%s_%s_response_s4.png" % (mname, region)),bbox_inches='tight')
                # fig2_variable_error_s4 = Plot_response2_error(obs2_1, mmod2_1, obs2_2, mmod2_2, 0, col_num=modnumber-1, site_name=region, s=4)
                # fig2_variable_error_s4.savefig(os.path.join(output_path, "%s_%s_response_error_s4.png" % (mname, region)), bbox_inches='tight')
                #
                # ################################ Corrlation Metrix
                # all_corr_h = []
                # all_corr_h_d = []
                # for j in range(len(h_mod1)):
                #     corr_h, mask_h = correlation_matrix(h_data[:, j, :, siteid].T)
                #     detrend_corr_h = detrend_corr(h_data[:, j, :, siteid].T)
                #
                #     all_corr_h.append(corr_h)
                #     all_corr_h_d.append(detrend_corr_h)
                # fig = plot_variable_matrix_trend_and_detrend(np.asarray(all_corr_h), np.asarray(all_corr_h_d),
                #                                              variable_list, col_num=modnumber - 1, site_name=region, score = self.score)
                # fig.savefig(os.path.join(output_path, "%s_%s_correlation_box.png" % (mname, region)),
                #             bbox_inches='tight')

        #
        mmname = self.mmname[1:]+ [self.mmname[0]]
        for modnumber, mname in enumerate(mmname):
            for regionid, bigregion in enumerate(self.regions[self.sitenumber:]):
                timesseries = ['_timeseries_hourly.png', '_timeseries_daily.png', '_timeseries_monthly.png',
                               '_timeseries_yearly.png', '_timeseries_seasonly.png', "_timeseries_s1.png",
                               "_timeseries_s2.png", "_timeseries_s3.png", "_timeseries_s4.png"]
                Taylorgrams =['_timeseries_taylorgram.png', '_timeseries_taylorgram_annual.png','_timeseries_taylorgram_hourofday.png', '_timeseries_taylorgram_cycles.png']
                cycles = ['_timeseries_hourofday.png', '_timeseries_dayofyear.png', '_timeseries_monthofyear.png',
                          '_timeseries_seasonofyear.png', '_timeseries_hourofday_s1.png',
                          '_timeseries_hourofday_s2.png',
                          '_timeseries_hourofday_s3.png', '_timeseries_hourofday_s4.png']
                pdfcdf = ['_PDFCDF.png']
                frequency = ['_wavelet.png',
                             '_wavelet_Mod0.png']  # '_wavelet_Mod2.png', # '_wavelet_Mod3.png'  '_IMF1.png', '_IMF2.png', '_IMF3.png', '_IMF4.png'
                response = ['_response.png', '_response_s1.png', '_response_s2.png', '_response_s3.png',
                            '_response_s4.png', '_response_error_s1.png'
                    , '_response_error_s2.png', '_response_error_s3.png', '_response_error_s4.png',
                            '_response_error.png',
                            '_response4.png', '_response4_s1.png', '_response4_s2.png',
                            '_response4_s3.png', '_response4_s4.png', '_corr4.png', '_correlation_box.png']
                allmetric = timesseries + cycles + pdfcdf + frequency + response + Taylorgrams
                for metric in allmetric:
                    list_im = []
                    for siteid, siteregion in enumerate(self.regions[:self.sitenumber]):
                        siteregion = str(siteid)
                        # print('site lat, lon', self.lats[siteid], self.lons[siteid])
                        # print('lat bound', self.lowerlatbound[regionid], self.upperlatbound[regionid])
                        # print('lon bound', self.lowerlonbound[regionid], self.upperlonbound[regionid])

                        if self.lats[siteid] < self.upperlatbound[regionid] and self.lats[siteid] > self.lowerlatbound[
                            regionid] and self.lons[siteid] < self.upperlonbound[regionid] and self.lons[siteid] > \
                                self.lowerlonbound[regionid]:
                            # print('In region',siteid, siteregion, regionid, siteregion)
                            list_im.append((output_path + "%s_%s" + metric) % (mname, siteregion))
                    imgs = [PIL.Image.open(i) for i in list_im]
                    x_axis_pictures_number = 2
                    while len(imgs) % x_axis_pictures_number != 0:
                        im = Image.new('RGB', (0, 0), color=tuple((np.random.rand(3) * 255).astype(np.uint8)))
                        imgs.append(im)
                    imgs_comb = pil_grid(imgs, x_axis_pictures_number)
                    imgs_comb.save((output_path + "%s_%s" + metric) % (mname, bigregion))
                list_im = []
                list_im.append((output_path + "%s_%s" + '_timeseries_legend.png') % (mname, siteregion))
                imgs = [PIL.Image.open(i) for i in list_im]
                imgs_comb = pil_grid(imgs, x_axis_pictures_number)
                imgs_comb.save((output_path + "%s_%s" + '_timeseries_legend.png') % (mname, bigregion))


        page.addFigure("Time series", ('(Time Series)' + "Legend"), output_path + "MNAME_RNAME_timeseries_legend.png", legend=False)
        page.addFigure("Time series", ('(Time Series)' + "Hourly"), output_path + "MNAME_RNAME_timeseries_hourly.png", legend=False)
        page.addFigure("Time series", ('(Time Series)' + "Daily"), output_path + "MNAME_RNAME_timeseries_daily.png", legend=False)
        page.addFigure("Time series", ('(Time Series)' + "Monthly"), output_path + "MNAME_RNAME_timeseries_monthly.png", legend=False)
        page.addFigure("Time series", ('(Time Series)' + "Seasonly"), output_path + "MNAME_RNAME_timeseries_seasonly.png", legend=False)
        page.addFigure("Time series", ('(Time Series)' + "Taylor"), output_path + "MNAME_RNAME_timeseries_taylorgram.png",  legend=False)

        page.addFigure("Time series(Annually)", ('(Time Series)' + "Yearly"), output_path + "MNAME_RNAME_timeseries_yearly.png", legend=False)
        page.addFigure("Time series(Annually)", ('(Time Series)' + "Season 1"), output_path + "MNAME_RNAME_timeseries_s1.png", legend=False)
        page.addFigure("Time series(Annually)", ('(Time Series)' + "Season 2"), output_path + "MNAME_RNAME_timeseries_s2.png", legend=False)
        page.addFigure("Time series(Annually)", ('(Time Series)' + "Season 3"), output_path + "MNAME_RNAME_timeseries_s3.png",  legend=False)
        page.addFigure("Time series(Annually)", ('(Time Series)' + "Season 4"), output_path + "MNAME_RNAME_timeseries_s4.png", legend=False)
        page.addFigure("Time series(Annually)", ('(Time Series)' + "TaylorAnnual"), output_path + "MNAME_RNAME_timeseries_taylorgram_annual.png",  legend=False)

        page.addFigure("PDF CDF", ('(PDF & CDF)' + "PDF & CDF"), output_path + "MNAME_RNAME_PDFCDF.png", legend=False)

        page.addFigure("Cycles mean(seasonly)", ("Time Series"), output_path + "MNAME_RNAME_timeseries_hourofday.png", legend=False)
        page.addFigure("Cycles mean(seasonly)", ("Season 1"), output_path + "MNAME_RNAME_timeseries_hourofday_s1.png",  legend=False)
        page.addFigure("Cycles mean(seasonly)", ("Season 2"), output_path + "MNAME_RNAME_timeseries_hourofday_s2.png",  legend=False)
        page.addFigure("Cycles mean(seasonly)", ("Season 3"), output_path + "MNAME_RNAME_timeseries_hourofday_s3.png", legend=False)
        page.addFigure("Cycles mean(seasonly)", ("Season 4"), output_path + "MNAME_RNAME_timeseries_hourofday_s4.png", legend=False)
        page.addFigure("Cycles mean(seasonly)", ("Taylor Graph"), output_path + "MNAME_RNAME_timeseries_taylorgram_hourofday.png", legend=False)

        page.addFigure("Cycles mean", ('(Cycle 1)' + "Time Series"), output_path + "MNAME_RNAME_timeseries_dayofyear.png", legend=False)
        page.addFigure("Cycles mean", ('(Cycle 2)' + "Time Series"), output_path + "MNAME_RNAME_timeseries_monthofyear.png", legend=False)
        page.addFigure("Cycles mean", ('(Cycle 3)' + "Time Series"),   output_path + "MNAME_RNAME_timeseries_seasonofyear.png", legend=False)
        page.addFigure("Cycles mean", ('(Cycle 4)' + "Time Series"),  output_path + "MNAME_RNAME_timeseries_taylorgram_cycles.png", legend=False)

        page.addFigure("Frequency", ('(Wavelet)' + "Wavelet Obs"), output_path + "MNAME_RNAME_wavelet.png", legend=False)
        page.addFigure("Frequency", ('(Wavelet)' + "Wavelet Mod0"), output_path + "MNAME_RNAME_wavelet_Mod0.png", legend=False)

        page.addFigure("Response two variables", ('(Time Series)' + "Legend2"), output_path + "MNAME_RNAME_timeseries_legend.png", legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response"), output_path + "MNAME_RNAME_response.png", legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response_error"), output_path + "MNAME_RNAME_response_error.png",  legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(DJF)"), output_path + "MNAME_RNAME_response_s1.png", legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(DJF)1"),  output_path + "MNAME_RNAME_response_error_s1.png", legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(MAM)"), output_path + "MNAME_RNAME_response_s2.png", legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(MAM)2"), output_path + "MNAME_RNAME_response_error_s2.png",  legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(JJA)"), output_path + "MNAME_RNAME_response_s3.png", legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(JJA)2"),  output_path + "MNAME_RNAME_response_error_s3.png",  legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(SON)"), output_path + "MNAME_RNAME_response_s4.png", legend=False)
        page.addFigure("Response two variables", ('(Response)' + "Response(SON)2"), output_path + "MNAME_RNAME_response_error_s4.png", legend=False)

        page.addFigure("Response four variables", ('(Response)' + "Response 4 Variable"), output_path + "MNAME_RNAME_response4.png", legend=False)
        page.addFigure("Response four variables", ('(Response)' + "Response 4 Variable(DJF)"), output_path + "MNAME_RNAME_response4_s1.png", legend=False)
        page.addFigure("Response four variables", ('(Response)' + "Response 4 Variable(MAM)"), output_path + "MNAME_RNAME_response4_s2.png", legend=False)
        page.addFigure("Response four variables", ('(Response)' + "Response 4 Variable(JJA)"),  output_path + "MNAME_RNAME_response4_s3.png", legend=False)
        page.addFigure("Response four variables", ('(Response)' + "Response 4 Variable(SON)"),  output_path + "MNAME_RNAME_response4_s4.png",  legend=False)

        page.addFigure("Correlations", ('(Correlations)' + "Correlations"), output_path + "MNAME_RNAME_corr4.png",  legend=False)
        page.addFigure("Correlations", ('(Correlations)' + "Correlations_box"), output_path + "MNAME_RNAME_correlation_box.png", legend=False)



    def generateHtml(self):
        """Generate the HTML for the results of this confrontation.

        This routine opens all netCDF files and builds a table of
        metrics. Then it passes the results to the HTML generator and
        saves the result in the output directory. This only occurs on
        the confrontation flagged as master.

        """
        # only the master processor needs to do this
        output_path = self.output_path #'/Users/lli51/Documents/ILAMB_sample/'
        # output_path ='/Users/lli51/Documents/ILAMB_sample/'

        for j, modname in enumerate(self.mmname):#
            # output_path = '/Users/lli51/Documents/ILAMB_sample/'
            results = Dataset(os.path.join(output_path, "%s_%s.nc" % (self.name, modname)), mode="w")
            results.setncatts({"name": modname, "color": m.color})
            for siteid, region in enumerate(self.regions[:-1]):
                for metric in self.new_metrics:
                    Variable(name=(region+metric), unit=self.uni_metrics[metric], data=self.score[metric][siteid][j]).toNetCDF4(results, group="MeanState")
            for metric in self.new_metrics:
                Variable(name=('global' + metric), unit=self.uni_metrics[metric], data=self.score[metric].mean(axis=0)[j]).toNetCDF4(results, group="MeanState")
            results.close()

        if self.master:
            results = Dataset(os.path.join(output_path, "%s_Benchmark.nc" % (self.name)), mode="w")
            results.setncatts({"name": "Benchmark", "color": np.asarray([0.5, 0.5, 0.5])})
            for siteid, region in enumerate(self.regions[:-1]):
                for metric in self.new_metrics:
                    Variable(name=(region + metric), unit=self.uni_metrics[metric], data=self.score[metric][siteid][-1]).toNetCDF4(results, group="MeanState")
            for metric in self.new_metrics:
                Variable(name=('global' + metric), unit=self.uni_metrics[metric], data=self.score[metric].mean(axis=0)[-1]).toNetCDF4(
                    results, group="MeanState")
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
                        for region in self.regions:
                            if region in vname:
                                found = True
                                var = grp.variables[vname]
                                name = vname.replace(region, "")
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




#
data_file = '/Users/lli51/Documents/ILAMB_sample/DATA/rsus/CERES/rsus_0.5x0.5.nc'
m = ModelResult('/Users/lli51/Documents/ILAMB_sample/MODELS/', modelname='mod1')
#
# # # data_file = '/Users/lli51/Downloads/alldata/obs_FSH_model_ilamb.nc4'
# # # m = ModelResult('/Users/lli51/Downloads/alldata/171206_ELMv0_CN_FSH_model_ilamb.nc4', modelname='mod1')
# #
c = ConfSite(source=data_file, name='CERES', variable='rsus')
c.confront(m)
c.generateHtml()
