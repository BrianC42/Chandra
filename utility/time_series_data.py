'''
Created on Jul 20, 2018

Based on 'Machine Learning Mastery' by Jason Brownlee

@author: Brian
'''
import logging
from pandas import DataFrame
from pandas import concat
 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    logging.debug('')
    logging.debug('===>========================================')
    logging.debug('===> series_to_supervised(\n%s\n, %s, %s, %s)', data, n_in, n_out, dropnan)
    logging.debug('===> data type: %s', type(data))
    logging.debug('===>========================================')

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    logging.debug('<-------------------------------------------')
    logging.debug('<--- series_to_supervised:\nshape %s', agg.shape)
    logging.debug('<--- head\n%s', agg.head())
    logging.debug('<--- tail\n%s', agg.tail())
    logging.debug('<-------------------------------------------')
    return agg
