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
from tda_api_library import format_tda_datetime

from technical_analysis_utilities import add_results_index
from technical_analysis_utilities import find_sample_index

def stochastic_oscillator_oversold(guidance, symbol, df_data):
    OVERSOLD = 10

    trigger_status = ""
    trade = False
    trigger_date = ""
    close = 0.0
        
    # Oversold indicator
    idx = len(df_data) - 1
    if idx >= 0:
        if df_data.at[idx, 'Stochastic Oscillator'] < OVERSOLD:
            trade = True
            close = df_data.at[idx, 'Close']
            trigger_status = "probable upward price move coming"
            trigger_date = format_tda_datetime(df_data.at[df_data.shape[0] - 1, 'DateTime'])

    if trade:
        guidance = guidance.append([[trade, symbol, 'SO Oversold', trigger_date, trigger_status, close]])
    return guidance

def trade_on_stochastic_oscillator(guidance, symbol, df_data):
    guidance = stochastic_oscillator_oversold(guidance, symbol, df_data)
    return guidance

def eval_stochastic_oscillator(df_data, eval_results):
    OVERSOLD = 5
    OVERBOUGHT = 95
    
    so1_index = 'Stochastic Oscillator, oversold, 10 day'
    so2_index = 'Stochastic Oscillator, reversal up, 10 day'
    so3_index = 'Stochastic Oscillator, overbought, 10 day'
    so4_index = 'Stochastic Oscillator, reversal down, 10 day'
    if not so1_index in eval_results.index:
        eval_results = eval_results.append(add_results_index(eval_results, so1_index))
        eval_results = eval_results.append(add_results_index(eval_results, so2_index))
        eval_results = eval_results.append(add_results_index(eval_results, so3_index))
        eval_results = eval_results.append(add_results_index(eval_results, so4_index))

    ndx = 1
    while ndx < len(df_data):
        if df_data.at[ndx, 'Stochastic Oscillator'] < OVERSOLD:
            eval_results.at[so1_index, find_sample_index(eval_results, df_data.at[ndx, '10 day change'])] += 1
            
        if df_data.at[ndx, 'Stochastic Oscillator'] > OVERBOUGHT:
            eval_results.at[so3_index, find_sample_index(eval_results, df_data.at[ndx, '10 day change'])] += 1
            
        if (df_data.at[ndx, 'Stochastic Oscillator'] >= df_data.at[ndx, 'SO SMA3']) and \
            (df_data.at[ndx-1, 'Stochastic Oscillator'] < df_data.at[ndx-1, 'SO SMA3']):
            eval_results.at[so2_index, find_sample_index(eval_results, df_data.at[ndx, '10 day change'])] += 1
            
        if (df_data.at[ndx, 'Stochastic Oscillator'] <= df_data.at[ndx, 'SO SMA3']) and \
            (df_data.at[ndx-1, 'Stochastic Oscillator'] > df_data.at[ndx-1, 'SO SMA3']):
            eval_results.at[so4_index, find_sample_index(eval_results, df_data.at[ndx, '10 day change'])] += 1
            
        ndx += 1
    return eval_results

def stochastic_oscillator(df_data=None):
    df_data.insert(loc=0, column='Stochastic Oscillator', value=0.0)
    df_data.insert(loc=0, column='SO SMA3', value=0.0)
    
    data_ndx = 14
    while data_ndx < len(df_data):
        trading_range = df_data.at[data_ndx - 14, '14 day max'] - df_data.at[data_ndx - 14, '14 day min']
        
        if trading_range == 0.0:
            df_data.at[data_ndx, 'Stochastic Oscillator'] = 0.0
        else:
            df_data.at[data_ndx, 'Stochastic Oscillator'] = (100 * (df_data.at[data_ndx, 'Close'] - df_data.at[data_ndx - 14, '14 day min'])) / trading_range
            
        if data_ndx >= 16:
            df_data.at[data_ndx, 'SO SMA3'] = (df_data.at[data_ndx-2, 'Stochastic Oscillator'] + \
                                              df_data.at[data_ndx-1, 'Stochastic Oscillator'] + \
                                              df_data.at[data_ndx, 'Stochastic Oscillator']) / 3
        data_ndx += 1
    
    return df_data