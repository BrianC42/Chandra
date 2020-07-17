'''
Created on Jan 31, 2018

@author: Brian

What is the 'Moving Average Convergence Divergence - MACD'
http://www.investopedia.com/terms/m/macd.asp?ad=dirN&qo=investopediaSiteSearch&qsrc=0&o=40186

Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows 
the relationship between two moving averages of prices. The MACD is calculated by subtracting 
the 26-day exponential moving average (EMA) from the 12-day EMA. A nine-day EMA of the MACD, 
called the "signal line", is then plotted on top of the MACD, functioning as a trigger for 
buy and sell signals.

BREAKING DOWN 'Moving Average Convergence Divergence - MACD'
There are three common methods used to interpret the MACD:

1. Crossovers - As shown in the chart above, when the MACD falls below the signal line, 
it is a bearish signal, which indicates that it may be time to sell. Conversely, when the 
MACD rises above the signal line, the indicator gives a bullish signal, which suggests 
that the price of the asset is likely to experience upward momentum. Many traders wait for a 
confirmed cross above the signal line before entering into a position to avoid getting getting 
"faked out" or entering into a position too early, as shown by the first arrow.

2. Divergence - When the security price diverges from the MACD. It signals the end of the current trend.

3. Dramatic rise - When the MACD rises dramatically - that is, the shorter moving average pulls 
away from the longer-term moving average - it is a signal that the security is overbought and 
will soon return to normal levels.

Traders also watch for a move above or below the zero line because this signals the position of the short-term average relative to the long-term average. 
When the MACD is above zero, the short-term average is above the long-term average, which signals upward momentum. 
The opposite is true when the MACD is below zero. As you can see from the chart above, the zero line often acts as an area of support and resistance for the indicator.

'''
import numpy as np
from numpy import NaN
import sys

from moving_average import exponential_moving_average
from technical_analysis_utilities import add_results_index
from technical_analysis_utilities import find_sample_index

def return_macd_flags():
    #MACD_flags = dict(buy='buy', sell='sell', neutral='neutral')
    MACD_flags = dict(buy='1', sell='-1', neutral='0')
    return (MACD_flags)

def return_macd_ind():
    MACD_ind = dict(buy=1, sell=-1, neutral=0)
    return (MACD_ind)

def eval_macd_positive_cross(df_data, eval_results):
    r1_index = 'MACD Positive Cross - 1 day'
    r5_index = 'MACD Positive Cross - 5 day'
    r10_index = 'MACD Positive Cross - 10 day'
    r20_index = 'MACD Positive Cross - 20 day'
    if not r10_index in eval_results.index:
        #MACD_results = add_results_index(eval_results, r10_index)
        eval_results = eval_results.append(add_results_index(eval_results, r1_index))
        eval_results = eval_results.append(add_results_index(eval_results, r5_index))
        eval_results = eval_results.append(add_results_index(eval_results, r10_index))
        eval_results = eval_results.append(add_results_index(eval_results, r20_index))
        
    rows = df_data.iterrows()
    for nrow in rows:
        if nrow[1]['MACD_Buy'] == True:
            if not np.isnan(nrow[1]['MACD']):
                #cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
                eval_results.at[r1_index, find_sample_index(eval_results, nrow[1]['1 day change'])] += 1
                eval_results.at[r5_index, find_sample_index(eval_results, nrow[1]['5 day change'])] += 1
                eval_results.at[r10_index, find_sample_index(eval_results, nrow[1]['10 day change'])] += 1
                eval_results.at[r20_index, find_sample_index(eval_results, nrow[1]['20 day change'])] += 1
    return eval_results

def add_macd_fields(df_data):
    df_data.insert(loc=0, column='MACD_Buy', value=False)
    df_data.insert(loc=0, column='MACD_Sell', value=False)
    df_data.insert(loc=0, column='MACD_flag', value=NaN)
    df_data.insert(loc=0, column='MACD', value=NaN)
    df_data.insert(loc=0, column='MACD_Signal', value=NaN)
    return (df_data)

def macd(df_src=None, value_label=None):
        
    sys.path.append("../Utilities/")

    #str_prediction = prediction_interval + ' day change'
    add_macd_fields(df_src)
    MACD_flags = return_macd_flags()
    '''
    Upward cross: short duration EMA changes from less than to greater than long duration EMA
                    compared to previous difference
                'MACD_Buy' = False
    Downward cross: short duration EMA changes from greater than to less than long duration EMA
                    compared to previous difference
                'MACD_Sell' = True
    '''
    idx = 1 ### EMA calculation forces 1st entries to be equal, will always show as crossover on 2nd row
    while idx < len(df_src):
        df_src.at[idx,'MACD'] = df_src.at[idx, "EMA12"] - df_src.at[idx, "EMA26"]
        idx += 1
    df_src = exponential_moving_average(df_src[:], value_label='MACD', interval=9, EMA_data_label='MACD_Signal')  
    idx = 1 ### EMA calculation forces 1st entries to be equal, will always show as crossover on 2nd row
    while idx < len(df_src):
        df_src.at[idx, 'MACD_flag'] = MACD_flags.get('neutral')
        if df_src.at[idx, 'MACD'] < df_src.at[idx, 'MACD_Signal']:
            if df_src.at[idx-1, 'MACD'] >= df_src.at[idx-1, 'MACD_Signal']:
                ### Downward crossover
                df_src.at[idx, 'MACD_Sell'] = True
                df_src.at[idx, 'MACD_flag'] = MACD_flags.get('sell')
        if df_src.at[idx, 'MACD'] > df_src.at[idx, 'MACD_Signal']:
            if df_src.at[idx-1, 'MACD'] <= df_src.at[idx-1, 'MACD_Signal']:
                ### Upward crossover
                df_src.at[idx, 'MACD_Buy'] = True
                df_src.at[idx, 'MACD_flag'] = MACD_flags.get('buy')   
        idx += 1
    
    return df_src

def add_macd_profile_fields(df_data, short_data=None, mid_data=None, long_data=None):
    df_data.insert(loc=0, column=mid_data, value=NaN)
    df_data.insert(loc=0, column=short_data, value=NaN)
    df_data.insert(loc=0, column=long_data, value=NaN)
    df_data.insert(loc=0, column='momentum', value=NaN)
    df_data.insert(loc=0, column='MACD_flag', value=NaN)

    return (df_data)

def macd_profile(df_src=None, \
         value_label=None, \
         short_interval=None, short_data=None, \
         mid_interval=None, mid_data=None, \
         long_interval=None, long_data=None, \
         date_label=None, \
         prediction_interval=None):
        
    sys.path.append("../Utilities/")

    add_macd_profile_fields(df_src, short_data, mid_data, long_data)
    MACD_flags = return_macd_ind()
    df_src = exponential_moving_average(df_src[:], value_label, date_label, short_interval, short_data)
    df_src = exponential_moving_average(df_src[:], value_label, date_label, mid_interval, mid_data)
    df_src = exponential_moving_average(df_src[:], value_label, date_label, long_interval, long_data)
    
    '''
    Upward cross: short duration EMA changes from less than to greater than long duration EMA
                    compared to previous difference
                'MACD_Buy' = False
    Downward cross: short duration EMA changes from greater than to less than long duration EMA
                    compared to previous difference
                'MACD_Sell' = True
    '''
    idx = 1 ### EMA calculation forces 1st entries to be equal, will always show as crossover on 2nd row
    EMA_momentum = 0.0
    
    while idx < len(df_src):
        EMA_momentum = df_src.ix[idx, "EMA12"] - df_src.ix[idx, "EMA26"]
        df_src.ix[idx,'momentum'] = EMA_momentum
        df_src.ix[idx, 'MACD_flag'] = MACD_flags.get('neutral')


        if df_src.ix[idx,'momentum'] < df_src.ix[idx, 'EMA9']:
                df_src.ix[idx, 'MACD_flag'] = MACD_flags.get('sell')
                    
        if df_src.ix[idx,'momentum'] > df_src.ix[idx, 'EMA9']:
                df_src.ix[idx, 'MACD_flag'] = MACD_flags.get('buy')
                
        idx += 1
    
    return df_src