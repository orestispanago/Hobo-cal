"""
Simpler/tidier version of calibration.py
"""
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import t
from scipy import stats

cwd = os.getcwd()
outdir = cwd + '/output/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

csvfiles = glob.glob(cwd + '/raw/*cal.csv')
hobonames = [os.path.split(i)[1][:3] for i in csvfiles]
ref = 'H53'
others = hobonames[:]
others.remove(ref)


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


# data is a list (no missing values)
def mtt(datain, siglvl=0.95):
    """ Modified Thompson test"""

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


def clean_thompson(df):
    serlist = [df['H53']]
    for i in others:
        res = df[i] - df['H53']
        reslist = res.tolist()
        thompson_t = mtt(reslist)
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
    col = ['slope', 'intercept', 'r_value', 'p_value', 'std_err', 'nan_perc']
    reg_df = pd.DataFrame('-', idx, col)

    def regdf(df, rowname=None):
        for k in others:
            df1 = df[['H53', k]]
            df1 = df1.dropna()
            x = df1[k].values
            y = df1['H53'].values
            nanperc = (1 - df[k].count() / df[ref].count()) * 100
            regstats = [s for s in stats.linregress(x, y)]
            regstats.append(nanperc)
            reg_df.loc[k, rowname] = regstats

    regdf(temps, rowname='raw')
    regdf(tempsc, rowname='thom')
    regdf(tempsi, rowname='IQR')
    return reg_df


large = load_dataset()
temps = large.xs('T', axis=1, level=1, drop_level=True)
# rh = large.xs('RH', axis=1, level=1, drop_level=True)


tempsc = clean_thompson(temps)
tempsi = clean_iqr(temps)
reg = calc_reg()

# TODO add barplot ?
# TODO add classes ?
