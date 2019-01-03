"""
Reads raw data from HOBOs
creates 10min timeseries with NaNs
concatenates to large dataframe
removes outliers by boxplot rule (IQR) and Thompson test
compares outlier removal methods
"""
# TODO write better comments
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import thompson as t  # from cwd

csvfiles = glob.glob('*.csv')
csvfiles.sort()
hobonames = [i[:3] for i in csvfiles]

sns.set(color_codes=True)


def make_ts(df, first=None, last=None, step='10min'):
    """Creates timeseries"""
    if first is None:
        if last is None:
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
                         names=['Time', hname + 'T', hname + 'RH'],
                         usecols=(0, 1, 2), index_col=0,
                         parse_dates=True, dayfirst=True)
        df = df[np.isfinite(df[hname + "T"])]  # dumps rows with NaNs
        dflist.append(df)
    return dflist


def corrfunc(x, y, **kwargs):
    """ Calculates Pearson's R and annotates axis
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
    grid = sns.PairGrid(data=temps, vars=list(temps), height=1)
    for m, n in zip(*np.triu_indices_from(grid.axes, k=0)):
        grid.axes[m, n].set_visible(False)
    grid = grid.map_lower(plt.scatter, s=0.2)
    grid.map_lower(corrfunc)
    grid.set(alpha=1)
    grid.fig.suptitle('Air Temperature (°C)')
    # plt.rcParams["axes.labelsize"] = 11


def ref_scatters(df, ref=None, figtitle=None):
    cols = list(df)
    cols.remove(ref)
    # noinspection PyTypeChecker
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


def calc_iqr():
    """ Finds outliers according to the boxplot rule (IQR).
    Clean data are marked as True or False in "*clean" column"""
    for i in others:
        q75 = np.percentile(temps[i + 'res'], 75)
        q25 = np.percentile(temps[i + 'res'], 25)
        iqr = q75 - q25
        temps[i + 'clean'] = ((temps[i + 'res'] > (q25 - 1.5 * iqr)) & (
                temps[i + 'res'] < (q75 + 1.5 * iqr)))
    return temps


def calc_thom():
    """ Finds outliers according to modified Thompson test.
    Clean data are marked as True or False in "*clean" column"""
    tempst = temps.copy()
    for k in others:
        thompson_t = t.mtt(tempst[k + 'res'].tolist())
        tempst[k + 'clean'] = tempst[k + 'res'].isin(thompson_t)
    return tempst


def lmplot_outliers(df, filend=None):
    """ Plots  scatter with 2 lines for outliers and good data.
    lmplot cannot be combined with subplots, so multiple figures are created"""
    for h in others:
        sns.lmplot(x=h, y='H53', hue=h + "clean", palette=['r', 'k'],
                   data=df, scatter_kws={'alpha': 0.3})
        plt.show()
        # plt.savefig(h+ filend)


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


def calc_reg():
    """ Calculates regression parameters
    before cleaning (raw)
    and after (IQR and Thompson)"""
    idx = pd.MultiIndex.from_product([others, ['raw', 'IQR', 'thom']],
                                     names=['sensor', 'data'])
    col = ['slope', 'intercept', 'r_value', 'p_value', 'std_err']
    reg_df = pd.DataFrame('-', idx, col)

    def regdf(df, rowname=None):
        for k in others:
            x = df[k].values
            y = df['H53'].values
            reg_df.loc[k, rowname] = stats.linregress(x, y)

    regdf(temps, rowname='raw')
    regdf(clean_i, rowname='IQR')
    regdf(clean_thom, rowname='thom')
    return reg_df


df_list = df_list()
large = pd.concat(df_list, axis=1)  # concatenates df_list to large dataframe
large = make_ts(large)
large.columns = pd.MultiIndex.from_tuples(
    [(c[:3], c[3:]) for c in large.columns])
large = large.dropna()

temps = large.xs('T', axis=1, level=1, drop_level=True)
rh = large.xs('RH', axis=1, level=1, drop_level=True)

# temp_res = calc_resids(temps)
# temp_res.plot(title='Temperature residuals $(H* - H53) = f(time)$',
#               figsize=(16, 9), subplots=True, sharey=True,layout=(2, 4))

# plot_diurnal(temp_res, ylab="$T - T_{ref}$",
#              figtitle='Diurnal variation of residuals ($T_{ref}$: H53)')


# ref_scatters(temps, ref='H53', figtitle='Air Temperature (°C)')

others = list(temps)
others.remove('H53')
for j in others:
    temps[j + 'res'] = temps[j] - temps['H53']

# tempsiqr = calc_iqr()
# lmplot_outliers(tempsiqr, filend='_iqr.png')
#
# tempsthom = calc_thom()
# lmplot_outliers(tempsthom, filend='_thom.png')


clean_i = clean_iqr(temps)
clean_i = clean_i.dropna()
# ref_scatters(clean_i, ref='H53', figtitle="Air temperature (°C)")

clean_thom = clean_thompson(temps)
clean_thom = clean_thom.dropna()
# ref_scatters(clean_thom,ref='H53',figtitle="Air temperature (°C)")

# reg_results = calc_reg()
# reg_results.to_excel('regression_results.xlsx')
