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
import pandas as pd
from numpy import NaN
import sys

from tda_api_library import format_tda_datetime

from moving_average import exponential_moving_average
from technical_analysis_utilities import add_results_index
from technical_analysis_utilities import find_sample_index

def return_macd_flags():
    MACD_flags = dict(buy='1', sell='-1', neutral='0')
    return (MACD_flags)

def return_macd_ind():
    MACD_ind = dict(buy=1, sell=-1, neutral=0)
    return (MACD_ind)

def trade_on_macd(guidance, symbol, df_data):
    day1 = df_data.shape[0]-3
    day2 = df_data.shape[0]-2
    day3 = df_data.shape[0]-1
    trigger_status = ""
    trade = False
    trigger_date = ""
    close = 0.0
    if day1 >= 0:
        trigger_date = format_tda_datetime(df_data.at[day3, 'DateTime'])
        if df_data.at[day3, 'MACD_Buy'] == True:
            trade = True
            trigger_status = "Potential recommendation in 2 trading days"
        if df_data.at[day2, 'MACD_Buy'] == True:
            if df_data.at[day3, 'MACD_Signal'] > 0:
                trade = True
                trigger_status = "Potential recommendation on next trading day"
        if df_data.at[day1, 'MACD_Buy'] == True:
            if df_data.at[day2, 'MACD_Signal'] > 0:
                if df_data.at[day3, 'MACD_Signal'] > 0:
                    trade = True
                    close = df_data.at[day3, 'Close']
                    trigger_status = "Recommend buy for 10 days target = 100% gain"
    
    if trade:
        guidance = guidance.append([[trade, symbol, 'MACD condition 1', trigger_date, trigger_status, close]])
    return guidance

def eval_macd_positive_cross(df_data, eval_results):
    index_1 = 'MACD confirmed positive Cross - 10 day max'
    if not index_1 in eval_results.index:
        eval_results = eval_results.append(add_results_index(eval_results, index_1))

    ndx = 0
    df_priors = pd.DataFrame()
    while ndx < df_data.shape[0]:
        df_priors = df_priors.append(df_data.iloc[[ndx]])
        if ndx >= 10 and ndx < (df_data.shape[0] - 1):
            if df_priors.at[ndx-1, 'MACD_Buy'] == True and \
                df_data.at[ndx, 'MACD_Signal'] > 0 and \
                df_data.at[ndx+1, 'MACD_Signal'] > 0:
                eval_results.at[index_1, find_sample_index(eval_results, df_data.at[ndx, '10 day max'])] += 1
        ndx += 1
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