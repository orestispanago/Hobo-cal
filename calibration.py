"""Compares IQR and modified Thompson test outlier removal methods"""
import os
import glob
import time
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as sm
from statsmodels.api import add_constant
import seaborn as sns
import matplotlib.pyplot as plt


def make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return f"{os.getcwd()}/{dirname}/"

def load_dataset():
    """ Reads hobo files to list of multiindex dataframes, concatenates them
    and makes timeseries"""
    dflist = []
    for fname, hname in zip(csvfiles, hobonames):
        df = pd.read_csv(fname, usecols=(0, 1, 2), index_col=0,
                         parse_dates=True, dayfirst=True)
        df.columns = pd.MultiIndex.from_tuples([(hname, 'T'), (hname, 'RH')])
        dflist.append(df)
    dfout = pd.concat(dflist, axis=1)
    dfout = dfout.dropna()  # dumps rows with NaNs
    dfout = dfout.asfreq('10min')
    return dfout


def mtt(datain, siglvl=0.95):
    """
    Mofified Thompson test

    :param list datain: must not have NaNs
    :param float siglvl: significance level (0 to 1)
    :returns: non outliers
    :rtype: list
    """

    def calc_TS(data):
        S = np.std(data)
        xbar = np.mean(data)
        # Thompson tau
        ta = stats.t.ppf((1 + siglvl) / 2, n - 2)
        thompson_tau = ta * (n - 1) / (np.sqrt(n) * np.sqrt(n - 2 + ta ** 2))
        # Thompson tau statistic
        TS = thompson_tau * S
        return TS, xbar

    n = len(datain)  # Determine the number of samples in datain
    if n < 3:
        print("ERROR: There must be at least 3 samples in the dataset for the "
              "Modified Thompson test")
    elif n >= 3:
        TS, xbar = calc_TS(datain)
        datain.sort(reverse=True)
        dataout = datain[:]  # copy of data
        # Compare the values of extreme high data points to TS
        while abs(max(dataout) - xbar) > TS:
            dataout.pop(0)
            # Determine the NEW value of S times tau
            TS, xbar = calc_TS(dataout)
        # Compare the values of extreme low data points to TS.
        # Begin by determining the NEW value of S times tau
        TS, xbar = calc_TS(dataout)
        while abs(min(dataout) - xbar) > TS:
            dataout.pop(len(dataout) - 1)
            TS, xbar = calc_TS(dataout)
    return dataout


# Left as separate functions for possible use in other script.
# Can be one function like mark_outliers()
def clean_thompson(df):
    """Cleans data using Thompson test """
    serlist = [df[ref]]
    for i in others:
        resids = df[i] - df[ref]
        thompson_t = mtt(resids.tolist())
        good = df[i][resids.isin(thompson_t)]
        serlist.append(good)
    clean = pd.concat(serlist, axis=1)
    return clean


def clean_iqr(df):
    """ Cleans data using boxplot rule (IQR)"""
    serlist = [df[ref]]
    for i in others:
        resids = df[i] - df[ref]
        q75 = np.percentile(resids, 75)
        q25 = np.percentile(resids, 25)
        iqr = q75 - q25  # InterQuantileRange
        good = df[i][
            ((resids > (q25 - 1.5 * iqr)) & (resids < (q75 + 1.5 * iqr)))]
        serlist.append(good)
    clean = pd.concat(serlist, axis=1)
    return clean


def calc_reg():
    """ Calculates regression parameters
    before cleaning (raw)
    and after (IQR and Thompson)"""
    idx = pd.MultiIndex.from_product([others, ['raw', 'IQR', 'thom']],
                                     names=['sensor', 'data'])
    col = regparams
    reg_df = pd.DataFrame('-', idx, col)

    def regdf(df, rowname=None):
        for k in others:
            df1 = df[[ref, k]]
            df1 = df1.dropna()
            x = df1[k]
            y = df1[ref]
            nanperc = (1 - df[k].count() / df[ref].count()) * 100

            X = add_constant(x)  # include constant (intercept) in ols model
            mod = sm.OLS(y, X)
            results = mod.fit()
            intercept, slope = results.params
            intercept_stderr, slope_stderr = results.bse
            rsquared = results.rsquared
            regstats = [slope, intercept, slope_stderr, intercept_stderr,
                        rsquared, nanperc]
            reg_df.loc[k, rowname] = regstats

    regdf(temps, rowname='raw')
    regdf(tempsc, rowname='thom')
    regdf(tempsi, rowname='IQR')
    return reg_df


def mark_outliers(df, method="Thompson"):
    """ Marks outliers as True or False in 'clean' column. Use for lmplot"""
    df1 = df.copy()
    for i in others:
        resids = df[i] - df[ref]
        if method == "Thompson":
            thompson_t = mtt(resids.tolist())
            df1[i + 'clean'] = resids.isin(thompson_t)  # True - False column
        elif method == "IQR":
            q75 = np.percentile(resids, 75)
            q25 = np.percentile(resids, 25)
            iqr = q75 - q25
            df1[i + 'clean'] = ((resids > (q25 - 1.5 * iqr)) & (
                    resids < (q75 + 1.5 * iqr)))  # True - False column
    return df1


def scatter_matrix_lower(df, folder=None, filename=None):
    """ Plots lower triangle of scatter matrix """

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

    grid = sns.PairGrid(data=df, vars=list(df), height=1)
    for m, n in zip(*np.triu_indices_from(grid.axes, k=0)):
        grid.axes[m, n].set_visible(False)
    grid = grid.map_lower(plt.scatter, s=0.2)
    grid.map_lower(corrfunc)
    grid.set(alpha=1)
    grid.fig.suptitle('Air Temperature (°C)')
    # plt.rcParams["axes.labelsize"] = 11
    if folder and filename:
        grid.savefig(f"{make_dir(folder)}{filename}")


def ref_scatters(df, figtitle=None, folder=None, filename=None):
    # noinspection PyTypeChecker
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, figsize=(16, 9))
    for i, col in enumerate(others):
        dfeach = df[[col,ref]].dropna()
        xval = dfeach[col].values
        yval = dfeach[ref].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(xval,
                                                                       yval)
        g = sns.regplot(x=ref, y=df[col], data=df, ax=axes[i // 4][i % 4],
                        scatter_kws={'s': 1}, truncate=True,
                        line_kws={'label': "$y={0:.1f}x+{1:.1f}$".format(slope,
                                                                         intercept)})
        g.legend()
    fig.suptitle(figtitle, fontsize=16)
    plt.show()
    if folder and filename:
        plt.savefig(f"{make_dir(folder)}{filename}")


def plot_diurnal(df, figtitle=None, ylab=None):
    """ Groups dataframe by hour and plots its columns on subplots"""
    hour = pd.to_timedelta(df.index.hour, unit='H')
    hour.name = 'Hour'
    dfout = df.groupby(hour).mean()
    ax = dfout.plot(title=figtitle, figsize=(16, 9), subplots=True,
                    sharey=True,
                    layout=(3, 3))
    ax[0][0].set_ylabel(ylab)
    ax[1][0].set_ylabel(ylab)


def lmplot_outliers(df, folder=None):
    """ Plots  scatter with 2 lines for outliers and good data.
    lmplot cannot be combined with subplots, so multiple figures are created"""
    for i in others:
        sns.lmplot(x=ref, y=i, hue=i + "clean", palette=['r', 'k'],
                   data=df, scatter_kws={'alpha': 0.3})
        plt.show()
        if folder:
            plt.savefig(f"{make_dir(folder)}{i}.png")


csvfiles = glob.glob('raw/*cal.csv')
hobonames = [os.path.split(i)[1][:3] for i in csvfiles]
ref = "H53"
others = hobonames[:]
others.remove(ref)
regparams = ['slope', 'intercept', 'sl_stderr', 'int_stderr', 'r2', 'out_perc']

start = time.time()
large = load_dataset()
temps = large.xs('T', axis=1, level=1, drop_level=True)
# rh = large.xs('RH', axis=1, level=1, drop_level=True)

temps = temps.dropna()
tempsc = clean_thompson(temps)
tempsi = clean_iqr(temps)
reg = calc_reg()
# ref_scatters(temps, folder='scatters', filename='ref_raw.png')
# ref_scatters(tempsc, folder='scatters', filename='ref_clean_thom.png')
# ref_scatters(tempsi, folder='scatters', filename='ref_clean_iqr.png')

# reg.to_excel(make_dir('reg')+'regresults.xlsx')

# reg = reg.reset_index()
# for i, j in enumerate(regparams):
#     fg = sns.catplot(x='sensor', y=j, hue='data', data=reg, kind='bar')
#     fg.savefig(make_dir('reg')+j+'.png')

# lmplot_outliers(mark_outliers(temps, method='Thompson'), folder='outliers/Thompson')
# lmplot_outliers(mark_outliers(temps, method='IQR'), folder='outliers/IQR')
# scatter_matrix_lower(temps, folder='scatters', filename='matrix_raw.png')


print(time.time() - start)
