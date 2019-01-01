'''
Reads raw data from LAPUP and HOBOs
creates 10min timeseries with NaNs
concatenates to large dataframe
plots T or rh scatter
'''

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

csvfiles = glob.glob('*.csv')
csvfiles.sort()
hobonames = [i[:3] for i in csvfiles]


def make_ts(df, first=None, last=None, step='10min'):
    """Creates timeseries"""
    if first == None and last == None:
        first = df.iloc[0].name
        last = df.iloc[-1].name
    df = df[~df.index.duplicated(keep='first')]  # dumps index duplicate values
    dr = pd.date_range(first, last, freq=step, name='Date-time, UTC')
    df = df.reindex(dr)  # creates step index
    return df


def df_list():
    """ Reads hobo files,
    creates timeseries based on largest hobo df,
    appends dataframes to list"""
    dflist = []
    for fname, hname in zip(csvfiles, hobonames):
        df = pd.read_csv(fname, skiprows=1,
                         names=['Time', hname + 'T', hname + 'rh'],
                         usecols=(0, 1, 2), index_col=0,
                         parse_dates=True, dayfirst=True)
        df = df[np.isfinite(df[hname + "T"])]  # dumps rows with NaNs
        hobo = make_ts(df)
        dflist.append(hobo)
    lens = [len(i) for i in dflist]
    longest = lens.index(max(lens))  # Find largest hobo dataframe
    start = dflist[longest].iloc[0].name
    end = dflist[longest].iloc[-1].name
    df1 = pd.read_csv("Meteo_1min_2018_raw1.dat", names=['Time', 'T', 'RH'],
                      usecols=(0, 4, 5), index_col=0,
                      parse_dates=True, dayfirst=False)
    lapup = make_ts(df1, first=start, last=end)
    dflist = [lapup] + dflist
    return dflist


def plot_residuals():
    """ Creates residuals dataframe and plots"""
    res = pd.DataFrame()
    for h in hobonames:
        res[h] = large[h + 'T'] - large['T']
    res.plot(title='Residuals (Hobo - LapUp)', figsize=(16, 9), subplots=True, sharey=True, layout=(3, 3))


def corrfunc(x, y, **kws):
    """ Calculates Pearson's R and annotates axis
    see https://stackoverflow.com/questions/30942577/seaborn-correlation-coefficient-on-pairgrid
    Use on seaborn scatter matrix"""
    r, _ = stats.pearsonr(x, y)
    r2 = r ** 2
    ax = plt.gca()
    ax.annotate("$r^2$ = {:.2f}".format(r2),
                xy=(.1, .9), xycoords=ax.transAxes, fontsize='x-small')


def slope_intercept(x, y, **kws):
    """ Calculates slope + intercept and annotates axis
    Use on seaborn scatter matrix"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    ax = plt.gca()
    ax.annotate("y={0:.1f}x+{1:.1f}".format(slope, intercept),
                xy=(.1, .9), xycoords=ax.transAxes, fontsize='x-small')


def scatter_matrix_lower():
    grid = sns.PairGrid(data=temps, vars=list(temps), size=1)
    for i, j in zip(*np.triu_indices_from(grid.axes, k=0)):
        grid.axes[i, j].set_visible(False)
    grid = grid.map_lower(plt.scatter, s=0.2)
    grid.map_lower(corrfunc)
    grid.set(alpha=1)
    grid.fig.suptitle('Air Temperature (°C)')
    plt.rcParams["axes.labelsize"] = 11


def ref_scatters(df, ref=None):
    cols = list(df)
    cols.remove(ref)
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, figsize=(16, 9))
    for i, col in enumerate(cols):
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[ref].values, df[col].values)
        g = sns.regplot(x=df[col], y=ref, data=df, ax=axes[i // 3][i % 3],
                        scatter_kws={'s': 1}, truncate=True,
                        line_kws={'label': "$y={0:.1f}x+{1:.1f}$".format(slope, intercept)})
        g.legend()
    plt.show()


# plot_scatter(large, 'T', ax_max=35)
# plot_scatter(large,'rh',100) # H52 RH not good

df_list = df_list()
large = pd.concat(df_list, axis=1)  # concatenates dflist to large dataframe
large.rename(columns={'T': 'LapT'}, inplace=True)
large.rename(columns={'RH': 'LapRH'}, inplace=True)

large.columns = pd.MultiIndex.from_tuples([(c[:3], c[3:]) for c in large.columns])
temps = large.xs('T', axis=1, level=1, drop_level=True)
rh = large.xs('RH', axis=1, level=1, drop_level=True)

# g = scatter_matrix(temps, alpha = 0.2, figsize = (16,9), diagonal = 'hist')

# temps.plot(title = 'Temperature',figsize=(16,9), subplots=True,sharey=True,layout=(3,4))
# rh.plot(title = 'RH',figsize=(16,9), subplots=True,sharey=True,layout=(3,4))

temps = temps.dropna()
ref_scatters(temps, ref='H53')
