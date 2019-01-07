"""
Simpler/tidier version of calibration.py
"""
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import t
import statsmodels.formula.api as sm
from statsmodels.api import add_constant
import seaborn as sns

cwd = os.getcwd()
outdir = cwd + '/output/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

csvfiles = glob.glob(cwd + '/raw/*cal.csv')
hobonames = [os.path.split(i)[1][:3] for i in csvfiles]
ref = "H53"
others = hobonames[:]
others.remove(ref)
regparams = ['slope', 'intercept', 'sl_stderr', 'int_stderr', 'r2', 'nan_perc']


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
    dfout = make_ts(dfout)
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
        ta = t.ppf((1 + siglvl) / 2, n - 2)
        thompson_tau = ta * (n - 1) / (np.sqrt(n) * np.sqrt(n - 2 + ta ** 2))
        # Thompson tau statistic
        TS = thompson_tau * S
        return TS, xbar

    n = len(datain)  # Determine the number of samples in datain
    if n < 3:
        print(
            'ERROR: There must be at least 3 samples in the data set for the Modified Thompson test')
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


def clean_thompson(df, ref=ref):
    """Cleans data using Thompson test"""
    serlist = [df[ref]]
    for i in others:
        resids = df[i] - df[ref]
        residlist = resids.tolist()
        thompson_t = mtt(residlist)
        good = df[i][resids.isin(thompson_t)]
        serlist.append(good)
    clean = pd.concat(serlist, axis=1)
    return clean


def clean_iqr(df):
    """ Cleans data using boxplot rule"""
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
            x = df1[k].values
            y = df1[ref].values
            nanperc = (1 - df[k].count() / df[ref].count()) * 100

            X = add_constant(x)  # include constant (intercept) in ols model
            mod = sm.OLS(y, X)
            results = mod.fit()
            slope = results.params[1]
            intercept = results.params[0]
            slope_stderr = results.bse[1]
            intercept_stderr = results.bse[0]
            rsquared = results.rsquared
            regstats = [slope, intercept, slope_stderr, intercept_stderr,
                        rsquared, nanperc]
            reg_df.loc[k, rowname] = regstats

    regdf(temps, rowname='raw')
    regdf(tempsc, rowname='thom')
    regdf(tempsi, rowname='IQR')
    return reg_df


large = load_dataset()
temps = large.xs('T', axis=1, level=1, drop_level=True)
# rh = large.xs('RH', axis=1, level=1, drop_level=True)

# temps = temps.dropna()
tempsc = clean_thompson(temps)
tempsi = clean_iqr(temps)
reg = calc_reg()

# reg.to_excel(outdir+'reg.xlsx')

reg = reg.reset_index()

for i, j in enumerate(regparams):
    fg = sns.catplot(x='sensor', y=j, hue='data', data=reg, kind='bar')
    # fg.savefig(outdir+j+'.png')

