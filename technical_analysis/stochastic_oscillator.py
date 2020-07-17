'''
Created on Jan 31, 2018

@author: Brian
http://www.investopedia.com/terms/s/stochasticoscillator.asp

What is the 'Stochastic Oscillator'
The stochastic oscillator is a momentum indicator comparing the closing price of a security to the 
range of its prices over a certain period of time. The sensitivity of the oscillator to market 
movements is reducible by adjusting that time period or by taking a moving average of the result.

BREAKING DOWN 'Stochastic Oscillator'
The stochastic oscillator is calculated using the following formula:
%K = 100(C - L14)/(H14 - L14)

Where:
C = the most recent closing price
L14 = the low of the 14 previous trading sessions
H14 = the highest price traded during the same 14-day period
%K = the current market rate for the currency pair
%D = 3-period moving average of %K

The general theory serving as the foundation for this indicator is that in a market trending upward, 
prices will close near the high, and in a market trending downward, prices close near the low. 
Transaction signals are created when the %K crosses through a three-period moving average, 
which is called the %D.

History
The stochastic oscillator was developed in the late 1950s by George Lane. As designed by Lane, 
the stochastic oscillator presents the location of the closing price of a stock in relation 
to the high and low range of the price of a stock over a period of time, typically a 14-day period. 
Lane, over the course of numerous interviews, has said that the stochastic oscillator does not 
follow price or volume or anything similar. He indicates that the oscillator follows the speed 
or momentum of price. Lane also reveals in interviews that, as a rule, the momentum or speed of 
the price of a stock changes before the price changes itself. In this way, the stochastic oscillator 
can be used to foreshadow reversals when the indicator reveals bullish or bearish divergences. 
This signal is the first, and arguably the most important, trading signal Lane identified.

Overbought vs Oversold
Lane also expressed the important role the stochastic oscillator can play in identifying 
overbought and oversold levels, because it is range bound. 
This range - from 0 to 100 - will remain constant, no matter how quickly or slowly a security 
advances or declines. Considering the most traditional settings for the oscillator, 
20 is typically considered the oversold threshold and 80 is considered the overbought threshold. 
However, the levels are adjustable to fit security characteristics and analytical needs. Readings 
above 80 indicate a security is trading near the top of its high-low range; readings below 20 
indicate the security is trading near the bottom of its high-low range.
'''
from numpy import NaN

from technical_analysis_utilities import add_results_index
from technical_analysis_utilities import find_sample_index

def eval_stochastic_oscillator(df_data, eval_results):
    so_index = 'Stochastic Oscillator, SI<20, 10 day'
    cross_index = 'Stochastic Oscillator, SIcrossSO SMA3, 10 day'
    if not so_index in eval_results.index:
        eval_results = eval_results.append(add_results_index(eval_results, so_index))
        eval_results = eval_results.append(add_results_index(eval_results, cross_index))

    ndx = 0
    rows = df_data.iterrows()
    for nrow in rows:
        if ndx == 0:
            prior = nrow
        if nrow[1]['Stochastic Oscillator'] < 20:
            cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
            eval_results.at[so_index, cat_str] += 1
        if (nrow[1]['Stochastic Oscillator'] >= nrow[1]['SO SMA3']) and \
            (prior[1]['Stochastic Oscillator'] < prior[1]['SO SMA3']):
            cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
            eval_results.at[cross_index, cat_str] += 1
        prior = nrow
        ndx += 1
    return eval_results

def add_stochastic_oscillator_fields(df_data=None):
    df_data.insert(loc=0, column='Stochastic Oscillator', value=NaN)
    df_data.insert(loc=0, column='SO SMA3', value=NaN)
    return df_data

def stochastic_oscillator(df_data=None):
    add_stochastic_oscillator_fields(df_data)
    
    data_ndx = 14
    while data_ndx < len(df_data):
        trading_range = df_data.at[data_ndx - 13, '14 day max'] - df_data.at[data_ndx - 13, '14 day min']
        
        if trading_range == 0:
            df_data.at[data_ndx, 'Stochastic Oscillator'] = 0.0
        else:
            df_data.at[data_ndx, 'Stochastic Oscillator'] = 100 * (df_data.at[data_ndx, 'Close'] - df_data.at[data_ndx - 13, '14 day min']) / trading_range
        if data_ndx >= 16:
            df_data.at[data_ndx, 'SO SMA3'] = df_data.at[data_ndx-2, 'Stochastic Oscillator'] + \
                                              df_data.at[data_ndx-1, 'Stochastic Oscillator'] + \
                                              df_data.at[data_ndx, 'Stochastic Oscillator'] / 3
        data_ndx += 1
    
    return df_data