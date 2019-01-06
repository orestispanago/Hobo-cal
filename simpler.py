"""
Simpler/tidier version of calibration.py
"""
import os
import glob
import pandas as pd
import numpy as np

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
        df = df.dropna()  # dumps rows with NaNs
        dflist.append(df)
    dfout = pd.concat(dflist, axis=1)
    dfout = make_ts(dfout)
    return dfout


large = load_dataset()
# temps = large.xs('T', axis=1, level=1, drop_level=True)
# rh = large.xs('RH', axis=1, level=1, drop_level=True)


# TODO calculate regression - store to dataframe
# TODO drop outliers with IQR and thompson
# TODO calculate regression (dropna on each column) - store to dataframe, same or separate?
