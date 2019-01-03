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


def df_list(read_lapup=True):
    """ Reads hobo files,
    creates timeseries based on largest hobo df,
    appends dataframes to list"""
    dflist = []
    for fname, hname in zip(csvfiles, hobonames):
        df = pd.read_csv(fname, skiprows=1,
                         names=['Time', hname + 'T', hname + 'RH'],
                         usecols=(0, 1, 2), index_col=0,
                         parse_dates=True, dayfirst=True)
        df = df[np.isfinite(df[hname + "T"])]  # dumps rows with NaNs
        hobo = make_ts(df)
        dflist.append(hobo)
    lens = [len(i) for i in dflist]
    longest = lens.index(max(lens))  # Find largest hobo dataframe
    start = dflist[longest].iloc[0].name
    end = dflist[longest].iloc[-1].name
    if read_lapup is True:
        df1 = pd.read_csv("Meteo_1min_2018_raw1.dat",
                          names=['Time', 'LapT', 'LapRH'],
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
    res.plot(title='Residuals (Hobo - LapUp)', figsize=(16, 9),
             subplots=True, sharey=True, layout=(3, 3))


def corrfunc(x, y, **kwargs):
    """ Calculates Pearson's R and annotates axis
    see https://stackoverflow.com/questions/30942577/seaborn-correlation-coefficient-on-pairgrid
    Use on seaborn scatter matrix"""
    r, _ = stats.pearsonr(x, y)
    r2 = r ** 2
    ax = plt.gca()
    ax.annotate("$r^2$ = {:.2f}".format(r2),
                xy=(.1, .9), xycoords=ax.transAxes, fontsize='x-small')


def slope_intercept(x, y, **kwargs):
    """ Calculates slope + intercept and annotates axis
    Use on seaborn scatter matrix"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    ax = plt.gca()
    ax.annotate("y={0:.1f}x+{1:.1f}".format(slope, intercept),
                xy=(.1, .9), xycoords=ax.transAxes, fontsize='x-small')


def scatter_matrix_lower():
    """ Plots lower triangle of scatter matrix """
    grid = sns.PairGrid(data=temps, vars=list(temps), size=1)
    for i, j in zip(*np.triu_indices_from(grid.axes, k=0)):
        grid.axes[i, j].set_visible(False)
    grid = grid.map_lower(plt.scatter, s=0.2)
    grid.map_lower(corrfunc)
    grid.set(alpha=1)
    grid.fig.suptitle('Air Temperature (°C)')
    # plt.rcParams["axes.labelsize"] = 11


def ref_scatters(df, ref=None, figtitle=None):
    cols = list(df)
    cols.remove(ref)
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, figsize=(16, 9))
    for i, col in enumerate(cols):
        xval = df[col].values
        yval = df[ref].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(xval,
                                                                       yval)
        g = sns.regplot(x=df[col], y=ref, data=df, ax=axes[i // 4][i % 4],
                        scatter_kws={'s': 1}, truncate=True,
                        line_kws={'label': "$y={0:.1f}x+{1:.1f}$".format(slope,
                                                                         intercept)})
        g.legend()
    fig.suptitle(figtitle, fontsize=16)
    plt.show()


def calc_resids(df, ref="H53"):
    resids = pd.DataFrame()
    others = list(df)
    others.remove(ref)
    for j in others:
        resids[j] = df[j] - df[ref]
    return resids


def plot_diurnal(df, figtitle=None, ylab=None):
    hour = pd.to_timedelta(df.index.hour, unit='H')
    hour.name = 'Hour'
    dfout = df.groupby(hour).mean()
    ax = dfout.plot(title=figtitle, figsize=(16, 9), subplots=True,
                    sharey=True,
                    layout=(2, 4))
    ax[0][0].set_ylabel(ylab)
    ax[1][0].set_ylabel(ylab)




df_list = df_list(read_lapup=False)
large = pd.concat(df_list, axis=1)  # concatenates dflist to large dataframe

large.columns = pd.MultiIndex.from_tuples([(c[:3], c[3:]) for c in large.columns])
large = large.dropna()
temps = large.xs('T', axis=1, level=1, drop_level=True)
rh = large.xs('RH', axis=1, level=1, drop_level=True)

# temp_res = calc_resids(temps)


others = list(temps)
others.remove('H53')
# for j in others:
#     temps[j + 'res'] = temps[j] - temps['H53']

tempst = temps.copy()
# temps.plot(title = 'Temperature',figsize=(16,9), subplots=True,sharey=True,layout=(3,4))
# rh.plot(title = 'RH',figsize=(16,9), subplots=True,sharey=True,layout=(3,4))

# ref_scatters(temps, ref='H53', figtitle='Air Temperature (°C)')


# # Linear regression for all hobos, ref = H53
# for k in list(temp_res):
#     slope, intercept, r_value, p_value, std_err = \
#     stats.linregress(temps[k], temps['H53'])
#     print(slope, intercept, r_value, p_value, std_err)


# temp_res.plot(title='Temperature', figsize=(16, 9), subplots=True, sharey=True,
#               layout=(2, 4))

# plot_diurnal(temp_res, figtitle='Diurnal variation of residuals (Tref: H53)', ylab="Tref - T")

# Working
sns.set(color_codes=True)
# for j in others:
#     q75 = np.percentile(temps[j + 'res'], 75)
#     q25 = np.percentile(temps[j + 'res'], 25)
#     iqr = q75 - q25
#     temps[j + 'clean'] = ((temps[j + 'res'] > (q25 - 1.5 * iqr)) & (
#             temps[j + 'res'] < (q75 + 1.5 * iqr)))
#
# for h in others:
#     g = sns.lmplot(x=h, y='H53', hue=h + "clean", palette=['r', 'k'],
#                    data=temps,
#                    scatter_kws={'alpha': 0.3})
#     plt.show()
#
# for h in others:
#     print(h, len(temps[temps[h + 'clean'] == False]) / len(temps))

# TODO calculate linear regression before and after cleaning

import thompson as t


# # Working
# for k in others:
#     thompson_t = t.mtt(tempst[k + 'res'].tolist())
#     tempst[k + 'clean'] = tempst[k + 'res'].isin(thompson_t)
# for h in others:
#     g = sns.lmplot(x=h, y='H53', hue=h + "clean", palette=['r', 'k'],
#                    data=tempst,
#                    scatter_kws={'alpha': 0.3})
#     # plt.show()
#     plt.savefig(h + '_thom.png')
def clean_thompson(df):
    serlist = [df['H53']]
    for i in others:
        res = df[i] - df['H53']
        reslist = res.tolist()
        thompson_t = t.mtt(reslist)
        good = df[i][res.isin(thompson_t)]
        serlist.append(good)
    clean = pd.concat(serlist, axis=1)
    return clean


clean_thom = clean_thompson(temps)
clean_thom = clean_thom.dropna()
# ref_scatters(clean_thom,ref='H53',figtitle="Air temperature (°C)")

def clean_iqr(df):
    serlist = [df['H53']]
    for i in others:
        res = df[i] - df['H53']
        q75 = np.percentile(res, 75)
        q25 = np.percentile(res, 25)
        iqr = q75 - q25
        good = df[i][((res > (q25 - 1.5 * iqr)) & (res < (q75 + 1.5 * iqr)))]
        serlist.append(good)
    clean = pd.concat(serlist, axis=1)
    return clean


clean_i = clean_iqr(temps)
clean_i = clean_i.dropna()
# ref_scatters(clean_i, ref='H53', figtitle="Air temperature (°C)")


idx = pd.MultiIndex.from_product([others,
                                  ['raw', 'IQR', 'thom']],
                                 names=['sensor', 'data'])
col = ['slope', 'intercept', 'r_value', 'p_value', 'std_err']

reg_df = pd.DataFrame('-', idx, col)
print(reg_df)


def regdf(df, rowname=None):
    for j in others:
        x = df[j].values
        y = df['H53'].values
        reg_df.loc[j, rowname] = stats.linregress(x, y)
        print(len(df[j]) / len(temps[j]))


regdf(temps, rowname='raw')
regdf(clean_i, rowname='IQR')
regdf(clean_thom, rowname='thom')
